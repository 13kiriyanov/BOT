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
from collections import deque
from decimal import Decimal
from typing import Awaitable, Callable

from polymarket import AsyncSecureClient
from polymarket.streams import UserSpec

from .logging_setup import log_event
from .models import ZERO, LiveOrder, Quote

log = logging.getLogger("polybot.exec")

FillCallback = Callable[
    [str, str, str, Decimal, Decimal, Decimal | None], Awaitable[None]
]
"""(condition_id, token_id, side, price, size, fee_rate_bps)

fee_rate_bps — ставка, которую биржа реально применила к филлу, или None,
если она её не прислала. Нужна, чтобы поймать расхождение нашей модели
комиссий с действительностью.
"""


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
        self.order_ttl_s = order_ttl_s
        self.on_fill = on_fill

        # (token_id, side) -> LiveOrder
        self._live: dict[tuple[str, str], LiveOrder] = {}
        # order_id -> (token_id, side) для быстрого поиска при апдейтах.
        self._by_id: dict[str, tuple[str, str]] = {}
        # token_id -> condition_id, чтобы филлы попадали в нужную позицию.
        self._token_market: dict[str, str] = {}

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
        self._live[(order.token_id, order.side)] = order
        self._by_id[order.order_id] = (order.token_id, order.side)

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
                    LiveOrder(oid, q.token_id, q.side, q.price, q.size)
                )
                log.info(
                    "[DRY] %s %s %s @ %s (%s)",
                    q.side, q.outcome, q.size, q.price, q.token_id[:10],
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
                    LiveOrder(str(resp.order_id), q.token_id, q.side, q.price, q.size)
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
            token = str(getattr(payload, "token_id", ""))
            side = str(getattr(payload, "side", "")).upper()
            price = Decimal(str(getattr(payload, "price", "0")))
            size = Decimal(str(getattr(payload, "size", "0")))
            status = str(getattr(payload, "status", "")).upper()

            # Учитываем только подтверждённые трейды, иначе задвоим позицию.
            if status not in ("MATCHED", "CONFIRMED", "MINED", "SUCCESS", ""):
                return
            if size <= 0 or not token:
                return

            raw_fee_bps = getattr(payload, "fee_rate_bps", None)
            fee_rate_bps = Decimal(str(raw_fee_bps)) if raw_fee_bps is not None else None

            condition_id = self._token_market.get(token, "")
            log.info("ФИЛЛ: %s %s @ %s (%s)", side, size, price, token[:10])
            log_event(
                "fill", token=token, side=side, price=price,
                size=size, condition_id=condition_id, status=status,
                fee_rate_bps=fee_rate_bps,
            )
            if self.on_fill and condition_id:
                await self.on_fill(condition_id, token, side, price, size, fee_rate_bps)

    async def run_user_stream(self) -> None:
        """Слушает свои ордера и трейды. Без этого бот слеп к своим филлам."""
        backoff = 1.0
        while not self._stop.is_set():
            try:
                log.info("Подключаюсь к user stream")
                async with self.client.subscribe(UserSpec()) as stream:
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
            found: dict[tuple[str, str], LiveOrder] = {}
            by_id: dict[str, tuple[str, str]] = {}
            async for o in self.client.list_open_orders():
                order = LiveOrder(
                    order_id=str(o.id),
                    token_id=str(o.token_id),
                    side=str(o.side).upper(),
                    price=Decimal(str(o.price)),
                    original_size=Decimal(str(o.original_size)),
                    size_matched=Decimal(str(o.size_matched or 0)),
                )
                key = (order.token_id, order.side)
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
