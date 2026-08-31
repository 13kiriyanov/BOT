"""
Главный движок: связывает фиды, модель, котирование, исполнение и риск.

Архитектура — набор независимых asyncio-задач:
  price_feed      — спот BTC/ETH
  book_stream     — стаканы Polymarket
  user_stream     — свои ордера и филлы
  discovery_loop  — периодический поиск активных рынков
  quote_loop      — ГЛАВНЫЙ цикл: пересчёт и перевыставление котировок
  merge_loop      — merge полных пар обратно в USDC
  watchdog        — dead-man switch

Watchdog намеренно живёт в отдельной задаче и следит за heartbeat главного
цикла. Если quote_loop завис или упал — watchdog снимает все ордера.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from decimal import Decimal

from polymarket import AsyncSecureClient

from .config import Settings
from .discovery import MarketDiscovery
from .execution import OrderManager
from .fair_value import FairValueModel
from .logging_setup import log_event
from .models import (
    ONE,
    POSITION_DECIMALS,
    ZERO,
    Book,
    Fill,
    MarketPosition,
    Outcome,
    Quote,
    RecoveredPosition,
    TargetMarket,
    shares_to_base_units,
)
from .orderbook import OrderBookManager
from .price_feed import SpotFeed
from .quoting import QuoteGenerator
from .risk import HaltReason, RiskManager

log = logging.getLogger("polybot.engine")

# Сколько раз пробуем прочитать позиции при старте и с какой паузой.
# Читаем настойчиво: неудача здесь означает остановку торговли.
RECOVERY_ATTEMPTS = 3
RECOVERY_RETRY_DELAY_S = 2.0


class TradingEngine:
    def __init__(self, settings: Settings) -> None:
        self.cfg = settings
        self.client: AsyncSecureClient | None = None

        self.spot = SpotFeed(
            vol_halflife_s=settings.strategy.vol_halflife_s,
            momentum_halflife_s=settings.strategy.momentum_halflife_s,
            vol_floor_annual=settings.strategy.vol_floor_annual,
        )
        self.books = OrderBookManager()
        self.fv_model = FairValueModel(
            model_weight=settings.strategy.model_weight,
            momentum_drift_coef=settings.strategy.momentum_drift_coef,
            max_model_deviation=settings.strategy.max_model_deviation,
        )
        self.quoter = QuoteGenerator(settings.strategy, settings.risk)
        self.risk = RiskManager(settings.risk)

        self.discovery: MarketDiscovery | None = None
        self.orders: OrderManager | None = None

        self.markets: dict[str, TargetMarket] = {}
        self.positions: dict[str, MarketPosition] = {}
        # Позиции, найденные на бирже при старте, чью сторону (YES/NO) ещё
        # предстоит подтвердить по token_id найденного рынка.
        self._recovered_by_token: dict[str, RecoveredPosition] = {}
        # Ставки комиссии, выученные из реальных филлов: condition_id ->
        # (rate, exponent). discovery пересоздаёт объекты рынков каждый цикл,
        # и без этого словаря знание терялось бы через 20 секунд.
        self._fee_overrides: dict[str, tuple[Decimal, Decimal]] = {}
        # Момент последнего филла или merge по рынку: сверка позиций не
        # трогает рынки со свежей активностью (data-api отстаёт от CLOB).
        self._last_activity: dict[str, float] = {}
        self._last_merge = 0.0
        self._shutdown = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------ setup

    async def start(self) -> None:
        w = self.cfg.wallet
        log.info("=" * 70)
        log.info("ЗАПУСК. Режим: %s", "DRY RUN (без реальных ордеров)" if self.cfg.runtime.dry_run else ">>> РЕАЛЬНАЯ ТОРГОВЛЯ <<<")
        log.info("=" * 70)

        creds = None
        if w.api_key and w.api_secret and w.api_passphrase:
            from polymarket import ApiKeyCreds

            creds = ApiKeyCreds(
                api_key=w.api_key.get_secret_value(),
                secret=w.api_secret.get_secret_value(),
                passphrase=w.api_passphrase.get_secret_value(),
            )

        self.client = await AsyncSecureClient.create(
            private_key=w.private_key.get_secret_value(),
            wallet=w.wallet_address,
            credentials=creds,
        )
        log.info("Клиент готов. Кошелёк: %s", getattr(self.client, "wallet", "?"))

        if not self.cfg.runtime.dry_run:
            await self._check_balance()

        self.orders = OrderManager(
            self.client,
            dry_run=self.cfg.runtime.dry_run,
            requote_threshold_ticks=self.cfg.strategy.requote_threshold_ticks,
            order_ttl_s=self.cfg.strategy.order_ttl_s,
            on_fill=self._on_fill,
        )
        self.discovery = MarketDiscovery(
            self.client,
            series_slugs=self.cfg.strategy.series_slugs,
            title_keywords=self.cfg.strategy.title_keywords,
            min_seconds=self.cfg.strategy.min_seconds_to_expiry,
            max_seconds=self.cfg.strategy.max_seconds_to_expiry,
            max_markets=self.cfg.strategy.max_concurrent_markets,
            fallback_fee_rate=self.cfg.strategy.fallback_fee_rate,
        )

        # На старте снимаем всё, что могло остаться от прошлой сессии.
        if not self.cfg.runtime.dry_run:
            await self.orders.cancel_all()

        # Ордера сняты — но позиции прошлой сессии остались на кошельке.
        await self._recover_positions()

    async def _check_balance(self) -> None:
        try:
            bal = await self.client.get_balance_allowance(asset_type="COLLATERAL")  # type: ignore[union-attr]
            raw = getattr(bal, "balance", None)
            # BalanceAllowance.balance приходит в базовых единицах (6 знаков),
            # а не в USDC: без деления лог врёт в миллион раз.
            human = Decimal(raw) / POSITION_DECIMALS if raw is not None else "?"
            log.info("Баланс USDC: %s", human)
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось прочитать баланс: %s", exc)

    # --------------------------------------------- восстановление позиций

    async def _fetch_positions(self) -> list[RecoveredPosition]:
        """Прочитать открытые позиции кошелька и привести к нашим моделям."""
        found: list[RecoveredPosition] = []
        async for p in self.client.list_positions():  # type: ignore[union-attr]
            rec = RecoveredPosition.from_api(p)
            if rec is None:
                continue
            # Резолвленные рынки — это уже требование к USDC, а не рыночный
            # риск. Заводить их в лимиты значит навсегда занять нотионал.
            if rec.redeemable:
                log.warning(
                    "Позиция %s (%s shares) резолвлена и ждёт redeem — в учёт не беру",
                    rec.title or rec.condition_id[:12], rec.size,
                )
                continue
            found.append(rec)
        return found

    async def _recover_positions(self) -> None:
        """
        Завести в локальный учёт позиции, уже открытые на кошельке.

        Бот не единственный источник правды: после рестарта или падения
        shares прошлой сессии никуда не деваются (ордера мы снимаем, позиции
        остаются). Пока их нет в учёте, риск-лимиты считаются от нуля,
        inventory skew котирует так, будто мы нейтральны, а merge не видит
        уже собранных пар.
        """
        if not self.cfg.strategy.recover_positions:
            log.warning(
                "STRAT_RECOVER_POSITIONS=false — учёт стартует с нуля, "
                "лимиты не увидят позиций прошлой сессии"
            )
            return

        recovered: list[RecoveredPosition] = []
        error: Exception | None = None
        for attempt in range(1, RECOVERY_ATTEMPTS + 1):
            try:
                recovered = await self._fetch_positions()
                error = None
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                error = exc
                log.warning(
                    "Не удалось прочитать позиции (попытка %d/%d): %s",
                    attempt, RECOVERY_ATTEMPTS, exc,
                )
                await asyncio.sleep(RECOVERY_RETRY_DELAY_S * attempt)

        if error is not None:
            if self.cfg.runtime.dry_run:
                log.error("Позиции не прочитаны: %s. В dry run продолжаю.", error)
                return
            # Котировать поверх неизвестной позиции опаснее, чем не котировать.
            self.risk.halt(HaltReason.FATAL, f"позиции не прочитаны: {error}")
            return

        if not recovered:
            log.info("Открытых позиций на бирже нет — стартуем с чистого листа")
            return

        for rec in recovered:
            if rec.outcome is None:
                if not rec.token_id:
                    # Ни ярлыка, ни токена: сторону определить нечем.
                    log.error(
                        "Позиция %s (%s shares) без стороны и без token_id — "
                        "в учёт не попадёт, разберитесь вручную",
                        rec.title or rec.condition_id[:12], rec.size,
                    )
                    continue
                # Сторону подтвердим по token_id, когда discovery найдёт рынок.
                self._recovered_by_token[rec.token_id] = rec
                log.warning(
                    "Позиция %s (%s shares) без ярлыка стороны — жду рынок",
                    rec.title or rec.condition_id[:12], rec.size,
                )
                continue
            self._position(rec.condition_id).apply_recovered(
                rec.outcome, rec.size, rec.avg_price
            )
            if rec.token_id:
                self._recovered_by_token[rec.token_id] = rec
            log.warning(
                "ВОССТАНОВЛЕНО: %s | %s %s shares по %s | %s",
                rec.title or rec.condition_id[:12], rec.outcome, rec.size,
                rec.avg_price if rec.avg_price is not None else "цена неизвестна, считаю по 1.0",
                rec.condition_id[:12],
            )

        total_net = sum((p.net for p in self.positions.values()), ZERO)
        total_cost = sum((p.total_cost for p in self.positions.values()), ZERO)
        log.warning(
            "Итог восстановления: рынков=%d net=%s нотионал=%s. "
            "Лимиты риска считаются уже с учётом этого.",
            len(self.positions), total_net, total_cost,
        )
        log_event(
            "recover", markets=len(self.positions), net=total_net, notional=total_cost,
        )

    def _confirm_recovered(self, market: TargetMarket) -> None:
        """
        Подтвердить сторону восстановленных позиций по token_id рынка.

        token_id — единственный надёжный признак стороны. Ярлык `outcome`
        приходит текстом ('Yes' / 'Up'), и если он разойдётся с token_id,
        локальный net окажется зеркальным реальности: бот будет «разгружать»
        позицию, докупая её. Такое расхождение останавливает торговлю.
        """
        sides: tuple[tuple[str, Outcome], ...] = (
            (market.yes_token_id, "YES"),
            (market.no_token_id, "NO"),
        )
        for token, outcome in sides:
            rec = self._recovered_by_token.pop(token, None)
            if rec is None:
                continue
            if rec.outcome is None:
                self._position(rec.condition_id).apply_recovered(
                    outcome, rec.size, rec.avg_price
                )
                log.warning(
                    "[%s] ВОССТАНОВЛЕНО по token_id: %s %s shares",
                    market.slug, outcome, rec.size,
                )
            elif rec.outcome != outcome:
                log.critical(
                    "[%s] Ярлык позиции (%s) разошёлся с token_id (%s). "
                    "Локальный учёт зеркален реальности — останавливаюсь.",
                    market.slug, rec.outcome, outcome,
                )
                self.risk.halt(
                    HaltReason.FATAL,
                    f"сторона восстановленной позиции неоднозначна: {market.slug}",
                )

    # ------------------------------------------------------- позиции

    def _position(self, condition_id: str) -> MarketPosition:
        if condition_id not in self.positions:
            self.positions[condition_id] = MarketPosition(condition_id=condition_id)
        return self.positions[condition_id]

    async def _on_fill(self, fill: Fill) -> None:
        """Колбэк из user stream: обновить позицию и риск."""
        market = self.markets.get(fill.condition_id)
        if market is None:
            log.warning("Филл по неизвестному рынку %s", fill.condition_id[:12])
            return

        self._audit_reported_fee(market, fill.fee_rate_bps)

        outcome: Outcome = "YES" if fill.token_id == market.yes_token_id else "NO"
        fee = market.fee_for(fill.price, fill.size)
        pos = self._position(fill.condition_id)
        before = pos.realized_pnl
        pos.apply_fill(outcome, fill.side, fill.price, fill.size, fee)
        self._last_activity[fill.condition_id] = time.time()

        self.risk.record_fill(fill.price, fill.size)
        if pos.realized_pnl != before:
            self.risk.record_realized(pos.realized_pnl - before)

        pairs = pos.complete_pairs
        basis = pos.pair_cost_basis()
        log.info(
            "[%s] Позиция: YES=%s NO=%s пар=%s себест.пары=%s net=%s%s",
            market.slug, pos.yes_size, pos.no_size, pairs,
            f"{basis:.4f}" if basis else "-", pos.net,
            f" комиссия={fee:.4f}" if fee > 0 else "",
        )

    def _audit_reported_fee(
        self, market: TargetMarket, fee_rate_bps: Decimal | None
    ) -> None:
        """
        Сверить нашу модель комиссии с тем, что биржа списала по факту.

        Мы считаем, что taker-only комиссию мейкер не платит, а все наши
        ордера post_only. Если по нашему филлу пришла ненулевая ставка, а мы
        заложили ноль — предположение неверно, и каждая следующая пара будет
        собираться в минус. Поднимаем ставку по факту: спред раздвинется на
        следующем же цикле котирования.
        """
        if fee_rate_bps is None or fee_rate_bps <= 0 or market.fee_rate > 0:
            return
        rate = fee_rate_bps / Decimal("10000")
        log.critical(
            "[%s] Биржа списала комиссию %s bps, хотя рынок считался для нас "
            "бесплатным. Ставлю rate=%s и расширяю спред.",
            market.slug, fee_rate_bps, rate,
        )
        market.fee_rate = rate
        # Точную форму (rate * (p*(1-p))**exp) по одной ставке не восстановить.
        # Плоская ставка — верхняя граница, ошибка уйдёт в нашу пользу.
        market.fee_exponent = ZERO
        self._fee_overrides[market.condition_id] = (rate, ZERO)
        log_event(
            "fee_mismatch", condition_id=market.condition_id,
            slug=market.slug, rate_bps=fee_rate_bps, rate=rate,
        )

    def _apply_fee_override(self, market: TargetMarket) -> None:
        """
        Вернуть рынку ставку комиссии, выученную из реального филла.

        Берём максимум: если расписание рынка объявило ставку выше нашей
        выученной, верим расписанию — оно про будущее, а филл про прошлое.
        """
        override = self._fee_overrides.get(market.condition_id)
        if override is None:
            return
        rate, exponent = override
        if market.fee_rate < rate:
            market.fee_rate = rate
            market.fee_exponent = exponent

    # --------------------------------------------------------- циклы

    async def discovery_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                spots = {
                    a: p for a in ("BTC", "ETH") if (p := self.spot.price(a)) is not None
                }
                found = await self.discovery.find_markets(spots)  # type: ignore[union-attr]

                new_map = {m.condition_id: m for m in found}
                added = set(new_map) - set(self.markets)
                removed = set(self.markets) - set(new_map)

                for cid in removed:
                    old = self.markets[cid]
                    log.info("[%s] Рынок вне окна торговли — ухожу", old.slug)
                    await self._exit_market(old)

                for cid in added:
                    m = new_map[cid]
                    log.info(
                        "[%s] Новый рынок: %s | до экспирации %.0fs | страйк %s | "
                        "комиссия %s",
                        m.slug, m.asset, m.seconds_left, m.strike,
                        m.fee_rate if m.fee_rate > 0 else "нет",
                    )
                    self.orders.register_market(cid, m.yes_token_id, m.no_token_id)  # type: ignore[union-attr]
                    self._confirm_recovered(m)

                # Ставку, выученную из филла, возвращаем на место: объекты
                # рынков здесь пересоздаются, а знание о комиссии — нет.
                for m in new_map.values():
                    self._apply_fee_override(m)

                self.markets = new_map

                tokens = set()
                for m in self.markets.values():
                    tokens.add(m.yes_token_id)
                    tokens.add(m.no_token_id)
                self.books.set_tokens(tokens)

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("discovery_loop: %s", exc, exc_info=True)

            await asyncio.sleep(self.cfg.strategy.discovery_interval_s)

    async def _exit_market(self, market: TargetMarket) -> None:
        """Снять котировки по рынку, который больше не торгуем."""
        tokens = {market.yes_token_id, market.no_token_id}
        stale = [o.order_id for o in self.orders.orders_for_market(tokens)]  # type: ignore[union-attr]
        if stale:
            await self.orders.cancel(stale)  # type: ignore[union-attr]
        self.discovery.forget(market.condition_id)  # type: ignore[union-attr]

    def _book_for(self, market: TargetMarket, outcome: str) -> Book | None:
        """Стакан токена, при необходимости восстановленный из дополнения."""
        token = market.token_for(outcome)  # type: ignore[arg-type]
        other = market.token_for(market.other(outcome))  # type: ignore[arg-type]
        book = self.books.book(token)
        if book is None or (not book.bids and not book.asks):
            return self.books.implied_from_complement(token, other)
        return book

    async def quote_loop(self) -> None:
        """ГЛАВНЫЙ ЦИКЛ."""
        while not self._shutdown.is_set():
            cycle_start = time.perf_counter()
            try:
                self.risk.heartbeat()

                ok = self.risk.check_global(
                    self.positions, self.orders.open_count  # type: ignore[union-attr]
                )
                if not ok:
                    if self.risk.is_halted:
                        await self.orders.cancel_all()  # type: ignore[union-attr]
                        await asyncio.sleep(5.0)
                    else:
                        await asyncio.sleep(self.cfg.strategy.quote_interval_s)
                    continue

                desired: list[Quote] = []
                for market in list(self.markets.values()):
                    desired.extend(await self._quotes_for_market(market))

                if self.markets:
                    tick = next(iter(self.markets.values())).tick_size
                    await self.orders.reconcile(desired, tick)  # type: ignore[union-attr]

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("quote_loop: %s", exc, exc_info=True)
                self.risk.record_reject()

            elapsed = time.perf_counter() - cycle_start
            await asyncio.sleep(max(0.0, self.cfg.strategy.quote_interval_s - elapsed))

    async def _quotes_for_market(self, market: TargetMarket) -> list[Quote]:
        """Посчитать желаемые котировки по одному рынку."""
        pos = self._position(market.condition_id)
        yes_book = self._book_for(market, "YES")
        no_book = self._book_for(market, "NO")

        if yes_book is None:
            return []

        # --- проверки допуска -------------------------------------------
        price_stale = self.spot.is_stale(market.asset, self.cfg.risk.stale_price_timeout_s)
        book_stale = yes_book.is_stale(self.cfg.risk.stale_book_timeout_s)
        allowed, reason = self.risk.can_quote_market(
            market,
            book_stale=book_stale,
            price_stale=price_stale,
            spread=yes_book.spread,
            depth=yes_book.depth("bids") + yes_book.depth("asks"),
            max_spread=self.cfg.strategy.max_market_spread,
            min_depth=self.cfg.strategy.min_book_depth,
        )
        if not allowed:
            log.debug("[%s] Не котирую: %s", market.slug, reason)
            # Позицию всё равно надо разгружать, если она перекошена.
            return self._unwind_if_needed(market, pos, yes_book, no_book)

        # --- fair value --------------------------------------------------
        mid = yes_book.microprice() or yes_book.mid
        if mid is None:
            return []

        spot = self.spot.price(market.asset)
        if spot is None or market.strike is None:
            # Без спота или страйка модель не работает — чистый MM по рынку.
            fv = self.fv_model.compute(
                spot=1.0, strike=1.0, seconds_left=market.seconds_left,
                market_mid=mid, sigma_annual=0.5,
                drift_per_second=0.0, vol_ready=False,
            )
        else:
            fv = self.fv_model.compute(
                spot=spot,
                strike=float(market.strike),
                seconds_left=market.seconds_left,
                market_mid=mid,
                sigma_annual=self.spot.sigma(market.asset),
                drift_per_second=self.spot.drift(market.asset),
                vol_ready=self.spot.vol_ready(market.asset),
            )

        log.debug(
            "[%s] mid=%.4f model=%.4f fair=%.4f edge=%+.4f sigma=%.2f conf=%.2f net=%s",
            market.slug, float(mid), float(fv.model_prob), float(fv.fair),
            float(fv.edge), float(fv.sigma_annual), float(fv.confidence), pos.net,
        )

        # --- разгрузка, если инвентарь перекошен -------------------------
        unwind = self._unwind_if_needed(market, pos, yes_book, no_book)
        if unwind:
            return unwind

        # --- котировки ---------------------------------------------------
        quotes = self.quoter.build_quotes(market, fv, pos, yes_book, no_book)

        # --- финальный риск-клип размеров --------------------------------
        result: list[Quote] = []
        for q in quotes:
            side_size = pos.yes_size if q.outcome == "YES" else pos.no_size
            # Покупка YES при лонге YES (или NO при лонге NO) увеличивает |net|.
            increases = (q.outcome == "YES" and pos.net >= 0) or (
                q.outcome == "NO" and pos.net <= 0
            )
            size = self.risk.clamp_order_size(
                q.size, side_size, pos.net, increases
            )
            if size >= market.min_order_size:
                q.size = size
                result.append(q)
        return result

    def _unwind_if_needed(
        self,
        market: TargetMarket,
        pos: MarketPosition,
        yes_book: Book | None,
        no_book: Book | None,
    ) -> list[Quote]:
        """Аварийная разгрузка одностороннего перекоса."""
        limit = (
            self.cfg.strategy.directional_max_net
            if self.cfg.strategy.allow_directional
            else Decimal("0")
        )
        # Близко к экспирации не оставляем голого направления.
        if market.seconds_left < 60:
            limit = min(limit, self.cfg.strategy.order_size)

        if abs(pos.net) <= limit:
            return []

        book = yes_book if pos.net > 0 else no_book
        return self.quoter.build_unwind_quotes(market, pos, book)

    async def merge_loop(self) -> None:
        """
        Merge полных пар обратно в USDC.

        Это главный механизм оборачиваемости капитала. Без merge каждая
        собранная пара замораживает деньги до резолюции рынка, и бот
        успевает сделать 1-2 круга за окно вместо десятков.
        """
        if not self.cfg.strategy.auto_merge:
            return
        while not self._shutdown.is_set():
            await asyncio.sleep(self.cfg.strategy.merge_interval_s)
            if self.cfg.runtime.dry_run or self.risk.is_halted:
                continue
            try:
                gas = self.cfg.strategy.merge_gas_cost
                for cid, pos in list(self.positions.items()):
                    pairs = pos.complete_pairs
                    if pairs < self.cfg.strategy.min_merge_size:
                        continue

                    basis = pos.pair_cost_basis()
                    # Без известной себестоимости считаем по худшему
                    # допустимому случаю: так порог сработает строже.
                    worst = basis if basis is not None else self.cfg.strategy.max_pair_cost
                    expected = (ONE - worst) * pairs
                    if gas > 0 and expected < gas * self.cfg.strategy.merge_min_profit_ratio:
                        log.debug(
                            "[%s] Merge отложен: прибыль %.4f не оправдывает газ %.4f",
                            cid[:12], expected, gas,
                        )
                        continue

                    # merge_positions ждёт amount в базовых единицах ERC-1155
                    # (6 знаков), а не в shares. Передать сюда shares значит
                    # смержить миллионную долю пачки, заплатив полный газ.
                    amount = shares_to_base_units(pairs)
                    if amount <= 0:
                        continue
                    merged = Decimal(amount) / POSITION_DECIMALS

                    log.info(
                        "MERGE %s пар по %s (себестоимость %s, газ %s)",
                        merged, cid[:12], f"{basis:.4f}" if basis else "?", gas,
                    )
                    handle = await self.client.merge_positions(  # type: ignore[union-attr]
                        condition_id=cid, amount=amount
                    )
                    before = pos.realized_pnl
                    pos.apply_merge(merged, gas)
                    self._last_activity[cid] = time.time()
                    profit = pos.realized_pnl - before
                    self.risk.record_realized(profit)
                    log_event(
                        "merge", condition_id=cid, pairs=merged, amount=amount,
                        basis=basis, gas=gas, profit=profit,
                        tx=str(getattr(handle, "transaction_hash", "")),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("merge_loop: %s", exc)

    async def watchdog(self) -> None:
        """Dead-man switch: главный цикл завис -> снимаем всё."""
        while not self._shutdown.is_set():
            await asyncio.sleep(1.0)
            if self.risk.heartbeat_expired() and not self.risk.is_halted:
                log.critical(
                    "WATCHDOG: нет heartbeat %.1fs — аварийная отмена ордеров",
                    self.risk.seconds_since_heartbeat(),
                )
                self.risk.halt(HaltReason.HEARTBEAT)
                if self.orders:
                    await self.orders.cancel_all()

    async def status_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(30.0)
            snap = self.risk.snapshot()
            pairs = sum((p.complete_pairs for p in self.positions.values()), ZERO)
            merged = sum((p.merged_pairs for p in self.positions.values()), ZERO)
            net = sum((p.net for p in self.positions.values()), ZERO)
            fees = sum((p.fees_paid for p in self.positions.values()), ZERO)
            gas = sum((p.merge_costs for p in self.positions.values()), ZERO)
            log.info(
                "СТАТУС | рынков=%d ордеров=%d | пар=%s смержено=%s net=%s | "
                "PnL=%.4f филлов=%d | комиссии=%.4f газ=%.4f | %s",
                len(self.markets), self.orders.open_count if self.orders else 0,
                pairs, merged, net, snap["realized_pnl"], snap["fills"],
                fees, gas,
                "HALTED:" + snap["reason"] if snap["halted"] else "OK",
            )
            log_event(
                "status", **snap, pairs=pairs, merged=merged, net=net,
                fees=fees, merge_gas=gas,
            )

    async def sync_loop(self) -> None:
        """Периодическая сверка ордеров с биржей."""
        while not self._shutdown.is_set():
            await asyncio.sleep(45.0)
            if self.orders and not self.risk.is_halted:
                await self.orders.sync_open_orders()

    # ------------------------------------------------------------- run

    async def run(self) -> None:
        await self.start()
        assert self.client and self.orders

        price_task = (
            self.spot.run_polymarket(self.client)
            if self.cfg.runtime.price_source == "polymarket"
            else self.spot.run_binance(self.cfg.runtime.binance_ws_url)
        )

        self._tasks = [
            asyncio.create_task(price_task, name="price"),
            asyncio.create_task(self.books.run(self.client), name="books"),
            asyncio.create_task(self.orders.run_user_stream(), name="user"),
            asyncio.create_task(self.discovery_loop(), name="discovery"),
            asyncio.create_task(self.quote_loop(), name="quote"),
            asyncio.create_task(self.merge_loop(), name="merge"),
            asyncio.create_task(self.watchdog(), name="watchdog"),
            asyncio.create_task(self.status_loop(), name="status"),
            asyncio.create_task(self.sync_loop(), name="sync"),
        ]

        log.info("Ждём наполнения фидов (5с) перед первым котированием...")
        await asyncio.sleep(5.0)

        try:
            await self._shutdown.wait()
        finally:
            await self.shutdown()

    def request_shutdown(self) -> None:
        log.warning("Получен сигнал остановки")
        self._shutdown.set()

    async def shutdown(self) -> None:
        """Корректное завершение: СНАЧАЛА снять ордера, потом всё остальное."""
        log.warning("Останавливаюсь...")
        self.risk.halt(HaltReason.MANUAL, "shutdown")

        if self.orders:
            try:
                await asyncio.wait_for(self.orders.cancel_all(), timeout=10.0)
            except Exception as exc:  # noqa: BLE001
                log.error("Не удалось снять ордера при выходе: %s", exc)
            self.orders.stop()

        self.spot.stop()
        self.books.stop()

        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        if self.client:
            try:
                await self.client.close()
            except Exception:  # noqa: BLE001
                pass

        snap = self.risk.snapshot()
        log.info("=" * 70)
        log.info(
            "СЕССИЯ ЗАВЕРШЕНА | PnL=%.4f | филлов=%d | объём=%.2f | аптайм=%.0fs",
            snap["realized_pnl"], snap["fills"], snap["volume"], snap["uptime_s"],
        )
        log.info("=" * 70)


def install_signal_handlers(engine: TradingEngine) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, engine.request_shutdown)
        except NotImplementedError:
            pass  # Windows
