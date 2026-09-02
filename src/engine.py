"""
Главный движок: связывает фиды, модель, котирование, исполнение и риск.

Архитектура — набор независимых asyncio-задач:
  price_feed      — спот BTC/ETH
  book_stream     — стаканы Polymarket
  user_stream     — свои ордера и филлы
  discovery_loop  — периодический поиск активных рынков
  quote_loop      — ГЛАВНЫЙ цикл: пересчёт и перевыставление котировок
  merge_loop      — merge полных пар обратно в USDC
  position_sync   — сверка локального учёта позиций с биржей
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
from .fair_value import FairValueModel, RealizedTwap, implied_strike_twap
from .logging_setup import log_event
from .markout import MarkoutTracker
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
from .regime import Regime, RegimeDetector
from .risk import HaltReason, RiskManager

log = logging.getLogger("polybot.engine")

# Сколько раз пробуем прочитать позиции при старте и с какой паузой.
# Читаем настойчиво: неудача здесь означает остановку торговли.
RECOVERY_ATTEMPTS = 3
RECOVERY_RETRY_DELAY_S = 2.0

# Период сверки локального учёта позиций с биржей.
POSITION_SYNC_INTERVAL_S = 60.0
# Рынки со свежим филлом или merge сверка пропускает: data-api, из которого
# читается list_positions(), отстаёт от CLOB на секунды, и сразу после
# исполнения «расхождение» — это лаг индексатора, а не ошибка учёта.
SYNC_ACTIVITY_GUARD_S = 90.0

# --- Калибровка страйка по живым данным -------------------------------------
# Спот «в момент открытия окна» засчитывается, только если мы поймали
# пересечение start_ts не позже, чем через столько секунд (quote_loop
# крутится каждые ~0.35 с, так что обычно ловим в первом же цикле).
WINDOW_OPEN_TOLERANCE_S = 3.0
# Инверсию модели по mid не делаем ближе к экспирации: d ~ 1/sqrt(tau),
# и K становится гиперчувствителен к шуму mid.
STRIKE_CALIB_MIN_SECONDS = 45.0
# ...и при экстремальном mid: наклон inv_cdf на краях огромен, ошибка
# в полтика mid превращается в ошибку страйка в десятки долларов.
STRIKE_CALIB_MID_BAND = (0.10, 0.90)


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
        # Замер adverse selection по каждому филлу (см. markout.py).
        self.markout = MarkoutTracker(self._markout_mid)
        # Детектор режима на каждый актив (BTC/ETH). Кормится тиками спота
        # и нашими филлами; реакция котирования включается флагами конфига.
        self._regimes: dict[str, RegimeDetector] = {}
        self._last_regime: dict[str, Regime] = {}
        self.spot.add_listener(self._on_spot_tick)

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
        # Состояние сторожа страйка: condition_id -> {"diverged_since",
        # "invalidations", "blocked"}. blocked=True — модель по рынку
        # отключена до конца окна (страйк дважды признан невалидным).
        self._strike_meta: dict[str, dict] = {}
        # Реализованная часть TWAP по каждому рынку (окно [start_ts, end_ts]).
        # Кормится тиками фида резолюции; без покрытого начала окна модель
        # для рынка не работает (state() вернёт None).
        self._twap: dict[str, RealizedTwap] = {}
        # Сглаженный mid (EWMA) для fair value: condition_id -> (value, ts).
        self._mid_ewma: dict[str, tuple[float, float]] = {}
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
        # Пагинатор итерируется страницами — элементы через iter_items().
        async for p in self.client.list_positions().iter_items():  # type: ignore[union-attr]
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

    def _regime_for(self, asset: str) -> RegimeDetector:
        if asset not in self._regimes:
            self._regimes[asset] = RegimeDetector.from_settings(self.cfg.strategy)
            self._last_regime[asset] = Regime.CALM
        return self._regimes[asset]

    def _on_spot_tick(
        self, asset: str, price: float, ts: float, window: int | None
    ) -> None:
        # Детектор режима — по одному ряду актива (60s или общий Binance),
        # иначе два окна TWAP считались бы за двойной поток тиков.
        if window in (None, 60):
            detector = self._regime_for(asset)
            detector.on_spot(price, ts)
            self._note_regime_change(asset, detector)
        # Реализованная часть TWAP — только рынкам с ОКНОМ этого тика:
        # 5-минутный рынок резолвится по 30s-ряду, 60s-тик ему чужой.
        for market in self.markets.values():
            if market.asset != asset:
                continue
            if window is not None and window != market.twap_window_s:
                continue
            tracker = self._twap.get(market.condition_id)
            if tracker is not None:
                tracker.update(price, ts)

    def _note_regime_change(self, asset: str, detector: RegimeDetector) -> None:
        current = detector.regime
        if current == self._last_regime.get(asset):
            return
        self._last_regime[asset] = current
        snap = detector.snapshot()
        log.warning("[%s] РЕЖИМ РЫНКА: %s | %s", asset, current.value, snap)
        log_event("regime", asset=asset, **snap)

    def _markout_mid(self, token_id: str, complement_id: str) -> Decimal | None:
        """
        Mid токена для замера mark-out: прямая книга, иначе зеркало из
        дополнения. Протухшая книга — это не цена: рынок мог истечь, и
        замороженный mid дал бы фиктивный mark-out.
        """
        stale_after = self.cfg.risk.stale_book_timeout_s
        book = self.books.book(token_id)
        if book is not None and not book.is_stale(stale_after):
            mid = book.mid
            if mid is not None:
                return mid
        implied = self.books.implied_from_complement(token_id, complement_id)
        if implied is not None and not implied.is_stale(stale_after):
            return implied.mid
        return None

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

        self.markout.record_fill(
            fill, outcome, market.token_for(market.other(outcome))
        )

        detector = self._regime_for(market.asset)
        detector.on_fill(outcome, fill.side, fill.size, fill.ts)
        self._note_regime_change(market.asset, detector)

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
                # Воронка фильтров каждым проходом: «рынков=0 потому что
                # кандидатов не было» и «кандидаты были, но все отсеялись»
                # должны различаться прямо в логе, без дебага.
                log.info(
                    "discovery: %s",
                    self.discovery.last_funnel.describe(),  # type: ignore[union-attr]
                )

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
                    # Накопитель реализованной части TWAP. Если бот увидел
                    # рынок после начала окна, накопитель сам признает
                    # начало непокрытым и модель для рынка не включится.
                    if m.start_ts is not None:
                        self._twap.setdefault(
                            cid, RealizedTwap(m.start_ts, m.end_ts)
                        )

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
        self.markout.forget_market(market.condition_id)
        self._strike_meta.pop(market.condition_id, None)
        self._twap.pop(market.condition_id, None)
        self._mid_ewma.pop(market.condition_id, None)

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

        # --- страйк: блокировка сторожем / отложенная калибровка ---------
        meta = self._strike_meta.get(market.condition_id)
        if meta is not None and meta.get("blocked"):
            # Дважды невалиден: модель для рынка выключена до конца окна,
            # даже если discovery пересобрал рынок с тем же страйком.
            market.strike = None
        elif market.strike is None:
            self._try_calibrate_strike(market, yes_book)

        # --- проверки допуска -------------------------------------------
        price_stale = self.spot.is_stale(
            market.asset, self.cfg.risk.stale_price_timeout_s, market.twap_window_s
        )
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

        # Крайние рынки не котируем: почти решённый исход — отвратительное
        # отношение риска к награде для мейкера (наблюдалось вживую:
        # BUY YES @ 0.03 + BUY NO @ 0.94). Разгрузка позиции разрешена.
        if not (
            self.cfg.strategy.quote_mid_min
            <= mid
            <= self.cfg.strategy.quote_mid_max
        ):
            log.debug(
                "[%s] mid=%.3f вне диапазона котирования [%s, %s] — только "
                "разгрузка", market.slug, float(mid),
                self.cfg.strategy.quote_mid_min, self.cfg.strategy.quote_mid_max,
            )
            return self._unwind_if_needed(market, pos, yes_book, no_book)

        # Сглаженный mid — ВХОД модели и якорь fair: микропрайс дёргается
        # каждым тиком стакана, и без сглаживания центр котирования скачет
        # на 3-4 тика за секунды (бесконечный cancel/replace, потеря места
        # в очереди). Границы цен ниже считаются по живому стакану.
        smoothed_mid = self._smoothed_mid(market.condition_id, float(mid))

        spot = self.spot.price(market.asset, market.twap_window_s)
        tracker = self._twap.get(market.condition_id)
        twap_state = tracker.state(time.time()) if tracker is not None else None

        if spot is None or market.strike is None or twap_state is None:
            # Без цены фида, страйка или покрытого начала окна модель не
            # работает — чистый MM по рынку. (Начало окна не наблюдалось =>
            # реализованная часть TWAP неизвестна, и честной вероятности
            # среднего не существует.)
            fv = self.fv_model.compute(
                spot=1.0, strike=1.0, seconds_left=market.seconds_left,
                market_mid=smoothed_mid, sigma_annual=0.5,
                drift_per_second=0.0, vol_ready=False,
            )
        else:
            fv = self.fv_model.compute(
                spot=spot,
                strike=float(market.strike),
                seconds_left=market.seconds_left,
                market_mid=smoothed_mid,
                sigma_annual=self.spot.sigma(market.asset, market.twap_window_s),
                drift_per_second=self.spot.drift(market.asset, market.twap_window_s),
                vol_ready=self.spot.vol_ready(market.asset, market.twap_window_s),
                twap_alpha=twap_state[0],
                twap_realized=twap_state[1],
            )

        # Сторож расхождения: модель, устойчиво спорящая с рынком на порог,
        # означает ложный страйк. Клип max_model_deviation в этом случае НЕ
        # защита: он оставляет fair сдвинутым на всю величину клипа, и обе
        # ноги котируются там, где их никогда не исполнят.
        if market.strike is not None and float(fv.confidence) > 0:
            if self._check_model_divergence(market, fv, smoothed_mid, spot):
                # Страйк только что снят — пересчитываем fair чистым MM
                # по рынку, не дожидаясь следующего цикла.
                fv = self.fv_model.compute(
                    spot=1.0, strike=1.0, seconds_left=market.seconds_left,
                    market_mid=smoothed_mid, sigma_annual=0.5,
                    drift_per_second=0.0, vol_ready=False,
                )

        log.debug(
            "[%s] mid=%.4f model=%.4f fair=%.4f edge=%+.4f sigma=%.2f conf=%.2f "
            "net=%s spot=%s strike=%s",
            market.slug, float(mid), float(fv.model_prob), float(fv.fair),
            float(fv.edge), float(fv.sigma_annual), float(fv.confidence), pos.net,
            f"{spot:.2f}" if spot is not None else "-", market.strike or "-",
        )

        # --- разгрузка, если инвентарь перекошен -------------------------
        unwind = self._unwind_if_needed(market, pos, yes_book, no_book)
        if unwind:
            return unwind

        # --- котировки ---------------------------------------------------
        regime_state = (
            self._regimes[market.asset].state()
            if market.asset in self._regimes
            else None
        )
        quotes = self.quoter.build_quotes(
            market, fv, pos, yes_book, no_book, regime_state
        )

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

    def _smoothed_mid(self, condition_id: str, mid: float) -> Decimal:
        """EWMA рыночного mid; при выключенном сглаживании — сырое значение."""
        halflife = self.cfg.strategy.fair_mid_smoothing_halflife_s
        if halflife <= 0:
            return Decimal(str(round(mid, 6)))
        now = time.time()
        prev = self._mid_ewma.get(condition_id)
        if prev is None:
            value = mid
        else:
            prev_value, prev_ts = prev
            lam = 0.5 ** (max(now - prev_ts, 0.0) / halflife)
            value = lam * prev_value + (1.0 - lam) * mid
        self._mid_ewma[condition_id] = (value, now)
        return Decimal(str(round(value, 6)))

    # ------------------------------------------------- страйк: живые данные

    def _try_calibrate_strike(self, market: TargetMarket, yes_book: Book | None) -> None:
        """
        Определить страйк по живым данным — отложенно, а не один раз при
        обнаружении рынка. При старте бота спот-фид ещё пуст и вол не
        прогрета: калибровка в discovery либо не срабатывала вовсе (страйк
        None у всех рынков, модель зажата в 0.5), либо срабатывала по
        устаревшим REST-данным и кэшировала ошибку навсегда.

        Два источника, по убыванию надёжности:
        1. Спот ровно в момент start_ts (открытие окна) — это определение
           страйка Up/Down-рынка.
        2. Инверсия GBM по живому mid нашего стакана, свежему споту и живой
           вол-оценке — только после открытия окна, не у экспирации и не на
           экстремальном mid.
        """
        cid = market.condition_id
        meta = self._strike_meta.get(cid)
        if meta is not None and meta.get("blocked"):
            # Дважды невалиден — новые страйки для рынка не создаём вовсе.
            return
        now = time.time()
        window = market.twap_window_s
        spot = self.spot.price(market.asset, window)
        spot_fresh = spot is not None and not self.spot.is_stale(
            market.asset, self.cfg.risk.stale_price_timeout_s, window
        )

        # 1. Пересечение открытия окна: спот сейчас и есть страйк.
        if (
            market.start_ts is not None
            and spot_fresh
            and 0.0 <= now - market.start_ts <= WINDOW_OPEN_TOLERANCE_S
        ):
            strike = Decimal(str(round(spot, 2)))  # type: ignore[arg-type]
            market.strike = strike
            if self.discovery is not None:
                self.discovery.observe_window_open(cid, strike)
                self.discovery.set_strike(cid, strike)
            log.info(
                "[%s] Страйк из наблюдения открытия окна: %s (spot=%.2f)",
                market.slug, strike, spot,
            )
            return

        # 2. Инверсия TWAP-модели по живому рынку.
        if not spot_fresh or market.seconds_left < STRIKE_CALIB_MIN_SECONDS:
            return
        if market.start_ts is not None and now < market.start_ts:
            # До открытия окна mid не несёт информации о будущем страйке.
            return
        if not self.spot.vol_ready(market.asset, window) or yes_book is None:
            return
        # Рынок ценит TWAP — инвертировать надо TWAP-модель, а для неё
        # нужна реализованная часть окна. Начало окна не покрыто => модель
        # всё равно не включится, калибровать незачем.
        tracker = self._twap.get(cid)
        twap_state = tracker.state(now) if tracker is not None else None
        if twap_state is None:
            return
        mid = yes_book.microprice() or yes_book.mid
        if mid is None:
            return
        lo, hi = STRIKE_CALIB_MID_BAND
        if not (lo <= float(mid) <= hi):
            return
        sigma = self.spot.sigma(market.asset, window)
        strike = implied_strike_twap(
            spot, float(mid), market.seconds_left, sigma,  # type: ignore[arg-type]
            alpha=twap_state[0], realized_avg=twap_state[1],
        )
        if strike is None:
            return
        market.strike = strike
        if self.discovery is not None:
            self.discovery.set_strike(cid, strike)
        log.info(
            "[%s] Страйк калиброван по живому рынку (TWAP): K=%s (spot=%.2f "
            "mid=%.4f sigma=%.2f alpha=%.2f реализовано=%.2f)",
            market.slug, strike, spot, float(mid), sigma,
            twap_state[0], twap_state[1],
        )

    def _check_model_divergence(
        self, market: TargetMarket, fv, mid: Decimal, spot: float | None
    ) -> bool:  # noqa: ANN001 - fv: FairValue
        """
        Сторож валидности страйка. True — страйк только что признан ложным.

        Триггер — УСТОЙЧИВОЕ расхождение: |model - mid| выше порога дольше
        strike_divergence_hold_s подряд (разовый выброс сбрасывает таймер).
        Реакция — отключить модель для рынка (чистый MM по рынку), а не
        клипать её: это направление отказа безопасно. После второго срыва —
        блок до конца окна: страйк из тех же источников будет так же ложным.
        """
        threshold = float(self.cfg.strategy.strike_divergence_threshold)
        divergence = abs(float(fv.model_prob) - float(mid))
        meta = self._strike_meta.setdefault(
            market.condition_id,
            {"diverged_since": None, "invalidations": 0, "blocked": False},
        )
        now = time.time()

        if divergence < threshold:
            meta["diverged_since"] = None
            return False
        since = meta["diverged_since"]
        if since is None:
            meta["diverged_since"] = now
            return False
        if now - since < self.cfg.strategy.strike_divergence_hold_s:
            return False

        meta["diverged_since"] = None
        meta["invalidations"] += 1
        log.warning(
            "[%s] Модель устойчиво расходится с рынком %.1f с: model=%.3f "
            "mid=%.3f strike=%s spot=%s — страйк невалиден, модель отключена",
            market.slug, now - since, float(fv.model_prob), float(mid),
            market.strike, f"{spot:.2f}" if spot is not None else "-",
        )
        market.strike = None
        if self.discovery is not None:
            self.discovery.invalidate_strike(market.condition_id)
        if meta["invalidations"] >= 2:
            meta["blocked"] = True
            log.warning(
                "[%s] Страйк невалиден повторно — модель для рынка отключена "
                "до конца окна", market.slug,
            )
        return True

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
            for line in self.markout.summary_lines():
                log.info(line)
            for asset, detector in sorted(self._regimes.items()):
                log.info("РЕЖИМ %s | %s", asset, detector.snapshot())
            log_event(
                "status", **snap, pairs=pairs, merged=merged, net=net,
                fees=fees, merge_gas=gas, markout=self.markout.summary(),
            )

    async def sync_loop(self) -> None:
        """Периодическая сверка ордеров с биржей."""
        while not self._shutdown.is_set():
            await asyncio.sleep(45.0)
            if self.orders and not self.risk.is_halted:
                await self.orders.sync_open_orders()

    # ------------------------------------------------- сверка позиций

    async def position_sync_loop(self) -> None:
        """
        Периодическая сверка локального учёта позиций с биржей.

        Локальный учёт строится из событий user-stream, а события теряются:
        разрыв WebSocket, пропущенный статус, FAILED после учтённого
        MATCHED, merge, не прошедший on-chain. Каждая потеря — тихое
        расхождение с реальностью, на котором все лимиты риска считаются
        от вымышленной позиции. Сверка — единственный механизм, который
        возвращает учёт к фактам.
        """
        while not self._shutdown.is_set():
            await asyncio.sleep(POSITION_SYNC_INTERVAL_S)
            if self.cfg.runtime.dry_run or self.risk.is_halted:
                # В dry run филлы локальные, на бирже их нет — сверять нечего.
                continue
            try:
                await self._sync_positions_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("position_sync_loop: %s", exc)

    def _exchange_outcome(self, rec: RecoveredPosition) -> Outcome | None:
        """Сторона биржевой позиции: token_id рынка надёжнее ярлыка API."""
        market = self.markets.get(rec.condition_id)
        if market is not None and rec.token_id:
            if rec.token_id == market.yes_token_id:
                return "YES"
            if rec.token_id == market.no_token_id:
                return "NO"
        return rec.outcome

    async def _sync_positions_once(self) -> None:
        """Одна сверка: расхождение больше допуска — предупреждение и
        коррекция к данным биржи; больше order_size*3 — остановка."""
        exchange: dict[str, dict[Outcome, RecoveredPosition]] = {}
        # Пагинатор итерируется страницами — элементы через iter_items().
        async for p in self.client.list_positions().iter_items():  # type: ignore[union-attr]
            rec = RecoveredPosition.from_api(p)
            if rec is None or rec.redeemable:
                continue
            outcome = self._exchange_outcome(rec)
            if outcome is None:
                log.warning(
                    "Сверка: у позиции %s не определить сторону — пропускаю",
                    rec.title or rec.condition_id[:12],
                )
                continue
            exchange.setdefault(rec.condition_id, {})[outcome] = rec

        now = time.time()
        halt_threshold = self.cfg.strategy.order_size * 3

        for cid in sorted(set(self.positions) | set(exchange)):
            if now - self._last_activity.get(cid, 0.0) < SYNC_ACTIVITY_GUARD_S:
                log.debug("Сверка %s: свежая активность, пропускаю", cid[:12])
                continue

            market = self.markets.get(cid)
            tolerance = (
                market.min_order_size
                if market is not None
                else self.cfg.strategy.fallback_min_order_size
            )
            pos = self.positions.get(cid)

            for outcome in ("YES", "NO"):
                local = ZERO
                if pos is not None:
                    local = pos.yes_size if outcome == "YES" else pos.no_size
                rec = exchange.get(cid, {}).get(outcome)  # type: ignore[arg-type]
                exch = rec.size if rec is not None else ZERO
                diff = abs(local - exch)
                if diff <= tolerance:
                    continue

                if diff > halt_threshold:
                    log.critical(
                        "РАСХОЖДЕНИЕ УЧЁТА: %s %s локально=%s на бирже=%s "
                        "(разница %s > %s). Учёт недостоверен — останавливаюсь.",
                        cid[:12], outcome, local, exch, diff, halt_threshold,
                    )
                    log_event(
                        "position_desync", condition_id=cid, outcome=outcome,
                        local=local, exchange=exch, diff=diff, action="halt",
                    )
                    self.risk.halt(
                        HaltReason.DESYNC,
                        f"{cid[:12]} {outcome}: локально {local}, на бирже {exch}",
                    )
                    return

                log.warning(
                    "Расхождение учёта: %s %s локально=%s на бирже=%s — "
                    "корректирую к бирже",
                    cid[:12], outcome, local, exch,
                )
                log_event(
                    "position_desync", condition_id=cid, outcome=outcome,
                    local=local, exchange=exch, diff=diff, action="correct",
                )
                self._position(cid).correct_side(
                    outcome,  # type: ignore[arg-type]
                    exch,
                    rec.avg_price if rec is not None else None,
                )
                # Коррекция уже учла биржевое состояние — отложенное
                # восстановление по этим токенам стало бы двойным учётом.
                for token, pending in list(self._recovered_by_token.items()):
                    if pending.condition_id == cid:
                        self._recovered_by_token.pop(token, None)

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
            asyncio.create_task(self.position_sync_loop(), name="possync"),
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
        try:
            await asyncio.wait_for(self.markout.aclose(), timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            log.error("Не удалось остановить замеры mark-out: %s", exc)

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
