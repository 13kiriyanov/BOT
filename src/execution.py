"""
Управление ордерами: постановка, отмена, cancel/replace, отслеживание филлов.

Принципы:
 * Diff-based. Мы не отменяем и не переставляем всё подряд каждый цикл —
   это самый быстрый способ упереться в rate limit и потерять место в
   очереди. Перевыставляем только то, что реально сдвинулось.
 * Батчинг. post_orders() отправляет пачку подписанных ордеров одним
   запросом; cancel_orders() отменяет пачку по списку id.
 * post_only=True на всех котировках. Мы мейкер. Ордер, который пересёк бы
   спред, должен быть отклонён, а не исполнен как тейкер.
 * dry_run — полная симуляция без единого сетевого вызова на запись.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict, deque
from decimal import Decimal
from typing import Awaitable, Callable

from polymarket import AsyncSecureClient
from polymarket.streams import UserSpec

from .logging_setup import log_event
from .models import MIN_GTD_TTL_S, ZERO, Fill, LiveOrder, Quote

log = logging.getLogger("polybot.exec")

FillCallback = Callable[[Fill], Awaitable[None]]
"""Колбэк на каждый НАШ филл, уже приведённый к нашей перспективе."""

# Сколько последних trade id держим для дедупликации. User-stream шлёт
# отдельное событие на каждую смену статуса трейда (MATCHED -> MINED ->
# CONFIRMED) и может повторить историю после переподключения — без дедупа
# каждый филл задваивается или затраивается.
SEEN_TRADES_MAX = 4096

# Сколько своих order id помним. Нужно больше, чем живых ордеров: order-
# событие со статусом MATCHED может снять ордер с учёта РАНЬШЕ, чем придёт
# trade-событие, и атрибуция по живым ордерам провалилась бы.
KNOWN_ORDERS_MAX = 4096


class RateLimiter:
    """Token bucket: не больше `rate` операций за `per` секунд."""

    def __init__(self, rate: int, per: float) -> None:
        self.rate = rate
        self.per = per
        self._events: deque[float] = deque()

    async def acquire(self) -> None:
        while True:
            now = time.time()
            while self._events and now - self._events[0] > self.per:
                self._events.popleft()
            if len(self._events) < self.rate:
                self._events.append(now)
                return
            await asyncio.sleep(max(0.01, self.per - (now - self._events[0])))


class OrderManager:
    """Единственная точка, через которую бот трогает ордера."""

    def __init__(
        self,
        client: AsyncSecureClient,
        *,
        dry_run: bool,
        requote_threshold_ticks: int,
        order_ttl_s: int,
        on_fill: FillCallback | None = None,
    ) -> None:
        self.client = client
        self.dry_run = dry_run
        self.requote_ticks = requote_threshold_ticks
        if 0 < order_ttl_s < MIN_GTD_TTL_S:
            log.warning(
                "order_ttl_s=%d ниже минимума GTD у биржи — поднимаю до %d "
                "(0 = GTC, если TTL не нужен)",
                order_ttl_s, MIN_GTD_TTL_S,
            )
            order_ttl_s = MIN_GTD_TTL_S
        self.order_ttl_s = order_ttl_s
        self.on_fill = on_fill

        # (token_id, side) -> LiveOrder
        self._live: dict[tuple[str, str, int], LiveOrder] = {}
        # order_id -> (token_id, side) для быстрого поиска при апдейтах.
        self._by_id: dict[str, tuple[str, str, int]] = {}
        # token_id -> condition_id, чтобы филлы попадали в нужную позицию.
        self._token_market: dict[str, str] = {}
        # LRU всех своих order id (живых и недавно снятых) — для атрибуции
        # trade-событий, и LRU обработанных trade id — для дедупликации.
        self._known_orders: OrderedDict[str, None] = OrderedDict()
        self._seen_trades: OrderedDict[str, None] = OrderedDict()

        self._post_limiter = RateLimiter(rate=25, per=1.0)
        self._cancel_limiter = RateLimiter(rate=25, per=1.0)
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._dry_seq = 0

    # ------------------------------------------------------------ учёт

    def register_market(self, condition_id: str, *token_ids: str) -> None:
        for t in token_ids:
            self._token_market[t] = condition_id

    @property
    def open_count(self) -> int:
        return len(self._live)

    def live_orders(self) -> list[LiveOrder]:
        return list(self._live.values())

    def orders_for_market(self, token_ids: set[str]) -> list[LiveOrder]:
        return [o for o in self._live.values() if o.token_id in token_ids]

    def _track(self, order: LiveOrder) -> None:
        key = (order.token_id, order.side, order.level)
        self._live[key] = order
        self._by_id[order.order_id] = key
        self._remember_order(order.order_id)

    def _remember_order(self, order_id: str) -> None:
        if not order_id:
            return
        self._known_orders[order_id] = None
        self._known_orders.move_to_end(order_id)
        while len(self._known_orders) > KNOWN_ORDERS_MAX:
            self._known_orders.popitem(last=False)

    def _remember_trade(self, trade_id: str) -> bool:
        """True, если trade id уже видели (дубль). Иначе запоминает его."""
        if not trade_id:
            return False  # без id дедупликация невозможна — обрабатываем
        if trade_id in self._seen_trades:
            self._seen_trades.move_to_end(trade_id)
            return True
        self._seen_trades[trade_id] = None
        while len(self._seen_trades) > SEEN_TRADES_MAX:
            self._seen_trades.popitem(last=False)
        return False

    def _untrack(self, order_id: str) -> None:
        key = self._by_id.pop(order_id, None)
        if key and self._live.get(key) and self._live[key].order_id == order_id:
            self._live.pop(key, None)

    # ------------------------------------------------- принятие решений

    def needs_replace(self, quote: Quote, tick: Decimal) -> bool:
        """Нужно ли снимать существующий ордер ради этой котировки."""
        existing = self._live.get(quote.key())
        if existing is None:
            return True
        # Цена сдвинулась больше порога?
        if abs(existing.price - quote.price) >= tick * self.requote_ticks:
            return True
        # Ордер почти выбран — обновляем размер.
        if existing.remaining < quote.size * Decimal("0.4"):
            return True
        # GTD-ордер скоро протухнет.
        if self.order_ttl_s > 0 and (time.time() - existing.created_at) > self.order_ttl_s * 0.8:
            return True
        return False

    # ------------------------------------------------------- операции

    async def cancel(self, order_ids: list[str]) -> int:
        if not order_ids:
            return 0
        if self.dry_run:
            for oid in order_ids:
                self._untrack(oid)
            log.debug("[DRY] Отмена %d ордеров", len(order_ids))
            return len(order_ids)

        await self._cancel_limiter.acquire()
        try:
            resp = await self.client.cancel_orders(order_ids=order_ids)
            cancelled = list(getattr(resp, "canceled", None) or [])
            for oid in order_ids:
                self._untrack(oid)
            not_cancelled = getattr(resp, "not_canceled", None) or {}
            if not_cancelled:
                log.debug("Не отменены: %s", not_cancelled)
            log_event("cancel", count=len(cancelled), ids=cancelled[:10])
            return len(cancelled)
        except Exception as exc:  # noqa: BLE001
            log.error("Ошибка отмены ордеров: %s", exc)
            # Считаем их мёртвыми: пусть лучше реальность окажется лучше учёта.
            for oid in order_ids:
                self._untrack(oid)
            return 0

    async def cancel_all(self) -> int:
        """Аварийная отмена ВСЕГО. Вызывается при halt и при выходе."""
        if self.dry_run:
            n = len(self._live)
            self._live.clear()
            self._by_id.clear()
            log.warning("[DRY] cancel_all: %d ордеров", n)
            return n
        try:
            resp = await self.client.cancel_all()
            n = len(list(getattr(resp, "canceled", None) or []))
            log.warning("cancel_all выполнен: отменено %d", n)
            log_event("cancel_all", count=n)
        except Exception as exc:  # noqa: BLE001
            log.error("cancel_all не удался: %s", exc)
            n = 0
        self._live.clear()
        self._by_id.clear()
        return n

    async def place_batch(self, quotes: list[Quote]) -> int:
        """Подписать и отправить пачку лимитных post-only ордеров."""
        if not quotes:
            return 0

        if self.dry_run:
            for q in quotes:
                self._dry_seq += 1
                oid = f"dry-{self._dry_seq}"
                self._track(
                    LiveOrder(oid, q.token_id, q.side, q.price, q.size, level=q.level)
                )
                log.info(
                    "[DRY] %s %s %s @ %s (%s) L%d",
                    q.side, q.outcome, q.size, q.price, q.token_id[:10], q.level,
                )
                log_event(
                    "place_dry", token=q.token_id, side=q.side,
                    outcome=q.outcome, price=q.price, size=q.size,
                )
            return len(quotes)

        expiration = int(time.time()) + self.order_ttl_s if self.order_ttl_s > 0 else None

        signed = []
        prepared: list[Quote] = []
        for q in quotes:
            try:
                order = await self.client.create_limit_order(
                    token_id=q.token_id,
                    price=q.price,
                    size=q.size,
                    side=q.side,
                    post_only=True,       # мы всегда мейкер
                    expiration=expiration,
                )
                signed.append(order)
                prepared.append(q)
            except Exception as exc:  # noqa: BLE001
                log.error("Не удалось подписать ордер %s: %s", q, exc)

        if not signed:
            return 0

        await self._post_limiter.acquire()
        try:
            responses = await self.client.post_orders(signed)
        except Exception as exc:  # noqa: BLE001
            log.error("post_orders упал: %s", exc)
            return 0

        placed = 0
        for q, resp in zip(prepared, responses):
            if getattr(resp, "ok", False) and getattr(resp, "order_id", None):
                self._track(
                    LiveOrder(str(resp.order_id), q.token_id, q.side, q.price, q.size,
                              level=q.level)
                )
                placed += 1
                log_event(
                    "place", order_id=str(resp.order_id), token=q.token_id,
                    side=q.side, outcome=q.outcome, price=q.price, size=q.size,
                )
            else:
                code = getattr(resp, "code", "?")
                msg = getattr(resp, "message", "")
                log.warning(
                    "Ордер отклонён [%s] %s | %s %s @ %s", code, msg, q.side, q.outcome, q.price
                )
                log_event("reject", code=str(code), message=str(msg), price=q.price)
        return placed

    async def reconcile(self, quotes: list[Quote], tick: Decimal) -> tuple[int, int]:
        """
        Привести реальные ордера к желаемым. Возвращает (отменено, поставлено).
        """
        async with self._lock:
            to_cancel: list[str] = []
            to_place: list[Quote] = []

            wanted_keys = {q.key() for q in quotes}

            # 1. Снимаем всё, чего больше нет в желаемом наборе.
            for key, order in list(self._live.items()):
                if key not in wanted_keys:
                    to_cancel.append(order.order_id)

            # 2. Для оставшихся решаем: оставить или переставить.
            for q in quotes:
                if self.needs_replace(q, tick):
                    existing = self._live.get(q.key())
                    if existing is not None:
                        to_cancel.append(existing.order_id)
                    to_place.append(q)

            cancelled = await self.cancel(to_cancel)
            placed = await self.place_batch(to_place)
            return cancelled, placed

    # ----------------------------------------------------- user stream

    async def _handle_user_event(self, event) -> None:  # noqa: ANN001
        etype = getattr(event, "type", "")
        payload = getattr(event, "payload", None)
        if payload is None:
            return

        if etype == "order":
            oid = str(getattr(payload, "id", ""))
            status = str(getattr(payload, "status", "")).upper()
            evt = str(getattr(payload, "order_event_type", "")).upper()
            key = self._by_id.get(oid)
            if key and key in self._live:
                order = self._live[key]
                matched = getattr(payload, "size_matched", None)
                if matched is not None:
                    order.size_matched = Decimal(str(matched))
                if status in ("CANCELED", "CANCELLED", "MATCHED") or evt in (
                    "CANCELLATION", "CANCELED",
                ):
                    self._untrack(oid)
                elif order.remaining <= 0:
                    self._untrack(oid)

        elif etype == "trade":
            status = str(getattr(payload, "status", "")).upper()
            # Учитываем трейд один раз по первому валидному статусу. FAILED
            # после уже учтённого MATCHED здесь не откатывается — это ловит
            # периодическая сверка позиций с биржей.
            if status not in ("MATCHED", "CONFIRMED", "MINED", "SUCCESS", ""):
                return

            trade_id = str(getattr(payload, "id", "") or "")
            if self._remember_trade(trade_id):
                log.debug("Дубль trade-события %s (%s) — пропускаю", trade_id[:16], status)
                return

            fills = self._extract_own_fills(payload, trade_id)
            if not fills:
                # Ни одна нога трейда не наша по известным order id. Не
                # бронируем ничего: верхнеуровневые side/price — тейкерские,
                # и учесть их «как есть» значит рискнуть перевернуть знак
                # позиции. Расхождение подберёт сверка позиций.
                log.error(
                    "Трейд %s не атрибуцирован ни одному нашему ордеру — "
                    "НЕ учитываю (сверка позиций скорректирует)", trade_id[:16],
                )
                log_event("fill_unattributed", trade_id=trade_id, status=status)
                return

            for fill in fills:
                log.info(
                    "ФИЛЛ: %s %s @ %s (%s)",
                    fill.side, fill.size, fill.price, fill.token_id[:10],
                )
                log_event(
                    "fill", trade_id=fill.trade_id, token=fill.token_id,
                    side=fill.side, price=fill.price, size=fill.size,
                    condition_id=fill.condition_id, status=status,
                    fee_rate_bps=fill.fee_rate_bps,
                )
                if self.on_fill and fill.condition_id:
                    await self.on_fill(fill)

    def _extract_own_fills(self, payload, trade_id: str) -> list[Fill]:  # noqa: ANN001
        """
        Достать из trade-события НАШИ ноги.

        Формат user-канала описывает трейд с точки зрения тейкера:
        верхнеуровневые side/price/size — его. Мы котируем только
        post_only, то есть всегда мейкер, и наша нога лежит в maker_orders —
        со СВОИМИ side, price (цена нашего лимитника) и matched_amount.
        Атрибуцируем по order id; поле trader_side оставлено как страховка
        на случай, если мы каким-то образом оказались тейкером.
        """
        fills: list[Fill] = []

        for mo in getattr(payload, "maker_orders", None) or ():
            order_id = str(getattr(mo, "order_id", "") or "")
            if not order_id or order_id not in self._known_orders:
                continue
            token = str(getattr(mo, "token_id", "") or "")
            side = str(getattr(mo, "side", "")).upper()
            price = Decimal(str(getattr(mo, "price", "0")))
            size = Decimal(str(getattr(mo, "matched_amount", "0")))
            raw_fee = getattr(mo, "fee_rate_bps", None)
            if not token or side not in ("BUY", "SELL") or size <= 0:
                continue
            fills.append(
                Fill(
                    trade_id=trade_id,
                    condition_id=self._token_market.get(token, ""),
                    token_id=token,
                    side=side,  # type: ignore[arg-type]
                    price=price,
                    size=size,
                    fee_rate_bps=Decimal(str(raw_fee)) if raw_fee is not None else None,
                )
            )
        if fills:
            return fills

        # Тейкерская ветка: с post_only сюда попадать не должны, но если
        # taker_order_id наш — верхнеуровневые поля описывают именно нас.
        taker_order_id = str(getattr(payload, "taker_order_id", "") or "")
        trader_side = str(getattr(payload, "trader_side", "") or "").upper()
        if taker_order_id in self._known_orders or trader_side == "TAKER":
            token = str(getattr(payload, "token_id", "") or "")
            side = str(getattr(payload, "side", "")).upper()
            price = Decimal(str(getattr(payload, "price", "0")))
            size = Decimal(str(getattr(payload, "size", "0")))
            raw_fee = getattr(payload, "fee_rate_bps", None)
            if token and side in ("BUY", "SELL") and size > 0:
                log.warning(
                    "Мы оказались ТЕЙКЕРОМ в трейде %s — при post_only такого "
                    "быть не должно", trade_id[:16],
                )
                fills.append(
                    Fill(
                        trade_id=trade_id,
                        condition_id=self._token_market.get(token, ""),
                        token_id=token,
                        side=side,  # type: ignore[arg-type]
                        price=price,
                        size=size,
                        fee_rate_bps=(
                            Decimal(str(raw_fee)) if raw_fee is not None else None
                        ),
                    )
                )
        return fills

    async def run_user_stream(self) -> None:
        """Слушает свои ордера и трейды. Без этого бот слеп к своим филлам."""
        backoff = 1.0
        while not self._stop.is_set():
            try:
                log.info("Подключаюсь к user stream")
                # subscribe() — корутина: сначала await (получаем
                # SubscriptionHandle), и только он — async context manager.
                async with await self.client.subscribe(UserSpec()) as stream:
                    backoff = 1.0
                    async for event in stream:
                        if self._stop.is_set():
                            break
                        try:
                            await self._handle_user_event(event)
                        except Exception as exc:  # noqa: BLE001
                            log.error("Ошибка обработки user-события: %s", exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("User stream упал: %s. Retry через %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 20.0)

    async def sync_open_orders(self) -> None:
        """
        Сверка с биржей. Локальное состояние может разойтись с реальностью
        после разрыва соединения — тогда бот думает, что ордеров нет, и
        ставит дубликаты.
        """
        if self.dry_run:
            return
        try:
            found: dict[tuple[str, str, int], LiveOrder] = {}
            by_id: dict[str, tuple[str, str, int]] = {}
            grouped: dict[tuple[str, str], list[LiveOrder]] = {}
            # Пагинатор итерируется страницами — элементы через iter_items().
            async for o in self.client.list_open_orders().iter_items():
                order = LiveOrder(
                    order_id=str(o.id),
                    token_id=str(o.token_id),
                    side=str(o.side).upper(),
                    price=Decimal(str(o.price)),
                    original_size=Decimal(str(o.original_size)),
                    size_matched=Decimal(str(o.size_matched or 0)),
                )
                grouped.setdefault((order.token_id, order.side), []).append(order)
                self._remember_order(order.order_id)
            # Уровни лестницы биржа не знает: восстанавливаем их по цене —
            # лучший бид (или лучший аск) получает уровень 0, как у quoting.
            for (token_id, side), orders in grouped.items():
                orders.sort(key=lambda o: o.price, reverse=(side == "BUY"))
                for level, order in enumerate(orders):
                    order.level = level
                    key = (token_id, side, level)
                    found[key] = order
                    by_id[order.order_id] = key

            if len(found) != len(self._live):
                log.warning(
                    "Рассинхрон ордеров: локально %d, на бирже %d — синхронизирую",
                    len(self._live), len(found),
                )
            self._live = found
            self._by_id = by_id
        except Exception as exc:  # noqa: BLE001
            log.error("sync_open_orders не удался: %s", exc)

    def stop(self) -> None:
        self._stop.set()
