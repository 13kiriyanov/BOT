"""
Тесты движка: восстановление позиций при старте и экономика merge.

Сеть и ключи не нужны — клиент SDK подменён заглушкой. Нужен только сам
пакет polymarket, потому что engine импортирует его типы; без него тесты
пропускаются, а не падают.

Запуск:  pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import pytest

pytest.importorskip("polymarket", reason="engine импортирует типы SDK")

from polymarket.pagination import AsyncPaginator, Page  # noqa: E402

from src import engine as engine_mod  # noqa: E402
from src.engine import TradingEngine  # noqa: E402
from src.models import Book, BookLevel, Fill, MarketPosition, TargetMarket  # noqa: E402
from src.risk import HaltReason  # noqa: E402

D = Decimal


# ----------------------------------------------------------------- заглушки


class Cfg:
    """Минимальный конфиг: только то, что читает движок."""

    class strategy:
        vol_halflife_s = 45.0
        momentum_halflife_s = 8.0
        vol_floor_annual = D("0.30")
        model_weight = D("0.35")
        momentum_drift_coef = D("0.30")
        max_model_deviation = D("0.15")
        target_pair_cost = D("0.985")
        max_pair_cost = D("0.995")
        min_half_spread_ticks = 1
        order_size = D("20")
        inventory_skew_coef = D("0.010")
        allow_directional = True
        directional_min_edge = D("0.025")
        directional_max_net = D("60")
        auto_merge = True
        min_merge_size = D("25")
        merge_interval_s = 0.0
        merge_gas_cost = D("0.02")
        merge_min_profit_ratio = D("3")
        recover_positions = True
        fallback_fee_rate = D("0.02")
        fallback_min_order_size = D("5")
        regime_trending_response = False  # дефолт конфига: вердикт измерения
        regime_volatile_no_quote = True
        trending_crowded_extra_ticks = 3
        trending_remove_crowded = False
        trending_tighten_ticks = 1
        regime_window_s = 120.0
        regime_min_fills = 6
        regime_imbalance_enter = 0.70
        regime_imbalance_soft = 0.45
        regime_imbalance_exit = 0.40
        regime_autocorr_enter = 0.25
        regime_vol_ratio_enter = 1.8
        regime_vol_ratio_exit = 1.35
        regime_min_hold_s = 45.0
        regime_vol_min_elapsed_s = 300.0
        strike_divergence_threshold = D("0.25")
        strike_divergence_hold_s = 8.0
        pair_sum_warn_gap = D("0.05")

    class risk:
        max_position_per_side = D("250")
        max_net_exposure = D("120")
        max_notional = D("500")
        daily_loss_limit = D("50")
        max_open_orders = 16
        heartbeat_timeout_s = 6.0
        stale_price_timeout_s = 4.0
        stale_book_timeout_s = 15.0
        max_consecutive_rejects = 8

    class runtime:
        dry_run = False


class ApiPosition:
    def __init__(self, **fields) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


class FakeClient:
    """Клиент, отдающий заранее заданные позиции и считающий merge-вызовы."""

    def __init__(self, positions=(), fail_positions: Exception | None = None) -> None:
        self._positions = list(positions)
        self._fail = fail_positions
        self.merge_calls: list[dict] = []

    def list_positions(self) -> AsyncPaginator:
        """
        Форма как у SDK: НАСТОЯЩИЙ AsyncPaginator поверх Page. Итерация
        `async for p in paginator` отдаёт страницы, а не позиции, — код,
        забывший .iter_items(), молча увидит пустой кошелёк. Прежняя
        заглушка отдавала элементы напрямую и прятала ровно этот баг.
        По одному элементу на страницу: чтение только первой страницы —
        тоже провал.
        """
        outer = self

        async def fetch(cursor: str | None) -> Page:
            if outer._fail is not None:
                raise outer._fail
            idx = int(cursor or 0)
            has_more = idx + 1 < len(outer._positions)
            return Page(
                items=tuple(outer._positions[idx:idx + 1]),
                has_more=has_more,
                next_cursor=str(idx + 1) if has_more else None,
            )

        return AsyncPaginator(fetch)

    async def merge_positions(self, *, condition_id: str, amount: int):
        self.merge_calls.append({"condition_id": condition_id, "amount": amount})
        return ApiPosition(transaction_hash="0xdead")


def make_engine(client: FakeClient) -> TradingEngine:
    engine = TradingEngine(Cfg())  # type: ignore[arg-type]
    engine.client = client  # type: ignore[assignment]
    return engine


def make_fill(**overrides) -> Fill:
    fields = dict(
        trade_id="t1",
        condition_id="0xcond",
        token_id="tok_yes",
        side="BUY",
        price=D("0.49"),
        size=D("10"),
        fee_rate_bps=None,
    )
    fields.update(overrides)
    return Fill(**fields)  # type: ignore[arg-type]


def make_market(**overrides) -> TargetMarket:
    fields = dict(
        condition_id="0xcond",
        slug="bitcoin-up-or-down",
        question="Bitcoin Up or Down?",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        end_ts=time.time() + 300,
        tick_size=D("0.01"),
        min_order_size=D("5"),
        neg_risk=False,
        asset="BTC",
    )
    fields.update(overrides)
    return TargetMarket(**fields)  # type: ignore[arg-type]


# --------------------------------------------- восстановление после рестарта


def test_start_recovers_positions_into_risk_base():
    """Позиции прошлой сессии попадают в учёт до первого котирования."""
    client = FakeClient(
        positions=[
            ApiPosition(
                condition_id="0xcond", token_id="tok_yes", size="60",
                avg_price="0.49", outcome="Up", redeemable=False,
                title="BTC Up or Down",
            ),
            ApiPosition(
                condition_id="0xcond", token_id="tok_no", size="40",
                avg_price="0.48", outcome="Down", redeemable=False,
                title="BTC Up or Down",
            ),
        ]
    )
    engine = make_engine(client)
    asyncio.run(engine._recover_positions())

    pos = engine.positions["0xcond"]
    assert pos.yes_size == D("60")
    assert pos.no_size == D("40")
    assert pos.complete_pairs == D("40")     # merge увидит их сразу
    assert pos.net == D("20")
    assert not engine.risk.is_halted


def test_redeemable_position_is_not_booked():
    """Резолвленная позиция — требование к USDC, а не рыночный риск."""
    client = FakeClient(
        positions=[
            ApiPosition(
                condition_id="0xold", token_id="tok_yes", size="500",
                avg_price="0.50", outcome="Yes", redeemable=True, title="старое окно",
            )
        ]
    )
    engine = make_engine(client)
    asyncio.run(engine._recover_positions())
    assert engine.positions == {}


def test_unlabeled_position_is_resolved_by_token_id():
    """Без ярлыка стороны позиция ждёт рынок и учитывается по token_id."""
    client = FakeClient(
        positions=[
            ApiPosition(
                condition_id="0xcond", token_id="tok_no", size="30",
                avg_price="0.45", outcome=None, redeemable=False,
            )
        ]
    )
    engine = make_engine(client)
    asyncio.run(engine._recover_positions())

    assert engine.positions == {}                      # сторона ещё неизвестна
    engine._confirm_recovered(make_market())
    assert engine.positions["0xcond"].no_size == D("30")
    assert engine.positions["0xcond"].net == D("-30")


def test_mismatched_side_halts_trading():
    """
    Ярлык биржи против token_id: локальный net оказался бы зеркальным
    реальности. Это обязано останавливать торговлю, а не «выравниваться».
    """
    client = FakeClient(
        positions=[
            ApiPosition(
                condition_id="0xcond", token_id="tok_no", size="30",
                avg_price="0.45", outcome="Yes", redeemable=False,
            )
        ]
    )
    engine = make_engine(client)
    asyncio.run(engine._recover_positions())
    engine._confirm_recovered(make_market())

    assert engine.risk.is_halted
    assert engine.risk.state.reason == HaltReason.FATAL


@pytest.fixture
def instant_retries(monkeypatch):
    """Ретраи чтения позиций без реальных пауз."""
    monkeypatch.setattr(engine_mod, "RECOVERY_RETRY_DELAY_S", 0.0)


def test_unreadable_positions_halt_live_trading(instant_retries):
    """Не смогли прочитать позиции — торговать вслепую нельзя."""
    engine = make_engine(FakeClient(fail_positions=RuntimeError("502 Bad Gateway")))
    asyncio.run(engine._recover_positions())
    assert engine.risk.is_halted
    assert engine.risk.state.reason == HaltReason.FATAL


def test_dry_run_survives_unreadable_positions(instant_retries):
    """В dry run ордеров нет, поэтому halt только мешал бы прогону."""
    engine = make_engine(FakeClient(fail_positions=RuntimeError("502 Bad Gateway")))
    engine.cfg.runtime.dry_run = True  # type: ignore[attr-defined]
    try:
        asyncio.run(engine._recover_positions())
        assert not engine.risk.is_halted
    finally:
        engine.cfg.runtime.dry_run = False  # type: ignore[attr-defined]


# ----------------------------------------------------------- merge и его цена


def _run_one_merge_pass(engine: TradingEngine) -> None:
    """Прокрутить merge_loop ровно один проход."""
    async def drive() -> None:
        task = asyncio.create_task(engine.merge_loop())
        for _ in range(50):
            await asyncio.sleep(0)
            if engine._shutdown.is_set():
                break
        engine._shutdown.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())


def test_merge_sends_base_units_not_shares():
    """
    merge_positions() ждёт amount в базовых единицах ERC-1155. Если послать
    туда shares, биржа смержит миллионную долю пачки, газ спишется целиком,
    а локальный учёт будет считать пары закрытыми.
    """
    client = FakeClient()
    engine = make_engine(client)
    pos = engine._position("0xcond")
    pos.apply_fill("YES", "BUY", D("0.49"), D("100"))
    pos.apply_fill("NO", "BUY", D("0.48"), D("100"))

    # merge_positions завершает проход, чтобы цикл не крутился вечно.
    original = client.merge_positions

    async def stop_after(**kwargs):
        result = await original(**kwargs)
        engine._shutdown.set()
        return result

    client.merge_positions = stop_after  # type: ignore[method-assign]
    _run_one_merge_pass(engine)

    assert client.merge_calls == [{"condition_id": "0xcond", "amount": 100_000_000}]
    assert pos.complete_pairs == D("0")
    assert pos.merged_pairs == D("100")


def test_merge_is_skipped_when_gas_eats_the_profit():
    """Merge ради прибыли размером с газ — это сжигание газа."""
    client = FakeClient()
    engine = make_engine(client)
    pos = engine._position("0xcond")
    # 30 пар по 0.9985 => валовая прибыль 0.045 при газе 0.02 и пороге x3.
    pos.apply_fill("YES", "BUY", D("0.4990"), D("30"))
    pos.apply_fill("NO", "BUY", D("0.4995"), D("30"))

    _run_one_merge_pass(engine)

    assert client.merge_calls == []
    assert pos.complete_pairs == D("30")     # пары остались нетронутыми


def test_merge_books_gas_into_pnl():
    """Газ merge должен уменьшать и PnL позиции, и PnL риск-менеджера."""
    client = FakeClient()
    engine = make_engine(client)
    pos = engine._position("0xcond")
    pos.apply_fill("YES", "BUY", D("0.49"), D("100"))
    pos.apply_fill("NO", "BUY", D("0.48"), D("100"))

    original = client.merge_positions

    async def stop_after(**kwargs):
        result = await original(**kwargs)
        engine._shutdown.set()
        return result

    client.merge_positions = stop_after  # type: ignore[method-assign]
    _run_one_merge_pass(engine)

    # 100 пар по 0.97 => 3.00 валовых, минус газ 0.02.
    assert pos.realized_pnl == pytest.approx(D("2.98"))
    assert pos.merge_costs == D("0.02")
    assert engine.risk.state.realized_pnl == pytest.approx(D("2.98"))


# ------------------------------------------------------- сверка позиций


def sync_once(engine) -> None:
    asyncio.run(engine._sync_positions_once())


def test_position_sync_corrects_medium_divergence_to_exchange():
    """
    Учёт разошёлся с биржей больше, чем на min_order_size: предупреждение
    и коррекция К БИРЖЕ — она источник правды, а не наш локальный счётчик.
    """
    client = FakeClient(
        positions=[
            ApiPosition(
                condition_id="0xcond", token_id="tok_yes", size="35",
                avg_price="0.49", outcome="Up", redeemable=False,
            )
        ]
    )
    engine = make_engine(client)
    engine.markets["0xcond"] = make_market()
    pos = engine._position("0xcond")
    pos.apply_fill("YES", "BUY", D("0.50"), D("50"))  # локально 50, на бирже 35
    pnl_before = pos.realized_pnl

    sync_once(engine)

    assert pos.yes_size == D("35")
    # Средняя цена осталась наша: коррекция — не сделка.
    assert pos.yes_cost == D("0.50") * D("35")
    assert pos.realized_pnl == pnl_before
    assert not engine.risk.is_halted


def test_position_sync_tolerates_small_divergence():
    """Расхождение в пределах min_order_size — не сигнал, а шум округления."""
    client = FakeClient(
        positions=[
            ApiPosition(
                condition_id="0xcond", token_id="tok_yes", size="48",
                avg_price="0.49", outcome="Up", redeemable=False,
            )
        ]
    )
    engine = make_engine(client)
    engine.markets["0xcond"] = make_market()  # min_order_size = 5
    engine._position("0xcond").apply_fill("YES", "BUY", D("0.50"), D("50"))

    sync_once(engine)
    assert engine.positions["0xcond"].yes_size == D("50")  # не тронуто


def test_position_sync_halts_on_large_divergence():
    """Расхождение больше order_size*3 — учёт недостоверен, торговать нельзя."""
    client = FakeClient(positions=[])  # на бирже пусто
    engine = make_engine(client)
    engine.markets["0xcond"] = make_market()
    engine._position("0xcond").apply_fill("YES", "BUY", D("0.50"), D("100"))
    # 100 - 0 = 100 > order_size(20) * 3

    sync_once(engine)

    assert engine.risk.is_halted
    assert engine.risk.state.reason == HaltReason.DESYNC


def test_position_sync_skips_markets_with_recent_activity():
    """
    data-api отстаёт от CLOB: сразу после филла «расхождение» — это лаг
    индексатора. Рынки со свежей активностью сверка не трогает.
    """
    client = FakeClient(positions=[])
    engine = make_engine(client)
    engine.markets["0xcond"] = make_market()
    engine._position("0xcond").apply_fill("YES", "BUY", D("0.50"), D("100"))
    engine._last_activity["0xcond"] = time.time()  # только что был филл

    sync_once(engine)

    assert not engine.risk.is_halted
    assert engine.positions["0xcond"].yes_size == D("100")


def test_position_sync_books_exchange_only_position():
    """Позиция есть на бирже, но не в учёте — бронируется по данным биржи."""
    client = FakeClient(
        positions=[
            ApiPosition(
                condition_id="0xother", token_id="tok_x", size="30",
                avg_price="0.40", outcome="Yes", redeemable=False,
            )
        ]
    )
    engine = make_engine(client)  # локально по 0xother ничего нет

    sync_once(engine)

    pos = engine.positions["0xother"]
    assert pos.yes_size == D("30")
    assert pos.yes_cost == D("0.40") * D("30")   # средняя взята с биржи
    assert not engine.risk.is_halted


# ------------------------------------------------------------- mark-out


def test_fill_schedules_markout_measurement():
    """Каждый филл попадает в трекер mark-out с верным дополнением."""
    from src.markout import MarkoutTracker

    events = []

    def sink(event, **fields):
        events.append(fields)

    engine = make_engine(FakeClient())
    engine.markets["0xcond"] = make_market()
    engine.markout = MarkoutTracker(
        engine._markout_mid, horizons_s=(0.01,), sink=sink
    )

    async def drive():
        await engine._on_fill(make_fill(token_id="tok_no", price=D("0.48")))
        while engine.markout.pending:
            await asyncio.sleep(0.005)

    asyncio.run(drive())

    assert len(events) == 1
    ev = events[0]
    assert ev["outcome"] == "NO"
    assert ev["token"] == "tok_no"
    # Книг в тесте нет — mid недоступен, и это честно отражено, а не выдумано.
    assert ev["markout_0.01s"] is None


# ------------------------------------------------------------ режим рынка


def test_fill_feeds_regime_detector():
    """Каждый филл попадает в детектор режима своего актива."""
    engine = make_engine(FakeClient())
    engine.markets["0xcond"] = make_market()

    asyncio.run(engine._on_fill(make_fill()))

    detector = engine._regimes["BTC"]
    assert detector.state().fills_in_window == 1

    # Тики спота через слушателя фида тоже доходят до детектора.
    engine.spot.ingest("BTC", 100_000.0)
    assert detector._last_price == 100_000.0


# ---------------------------------------------------------- сверка комиссии


def test_reported_fee_overrides_free_market_assumption():
    """
    Мы считаем, что мейкер не платит taker-only комиссию. Если биржа всё же
    списала её с нашего филла — предположение неверно, и спред обязан
    раздвинуться, а не остаться прежним.
    """
    engine = make_engine(FakeClient())
    market = make_market()
    engine.markets["0xcond"] = market
    assert market.fee_rate == D("0")

    asyncio.run(engine._on_fill(make_fill(fee_rate_bps=Decimal("20"))))

    assert market.fee_rate == D("0.002")          # 20 bps
    # Форму комиссии по одной ставке не восстановить, поэтому ставка плоская:
    # это верхняя граница, ошибка уходит в нашу пользу.
    assert market.fee_per_share(D("0.49")) == D("0.002")
    # Сверка идёт до учёта филла, поэтому комиссия попадает в себестоимость
    # уже этого филла — биржа списала её именно с него.
    assert engine.positions["0xcond"].fees_paid == D("0.020")


def test_learned_fee_survives_market_rediscovery():
    """
    discovery пересоздаёт объекты рынков каждый цикл. Если выученная из
    филла ставка не переживёт пересоздание, бот будет заново собирать пару
    в минус каждые 20 секунд — и каждый раз узнавать об этом постфактум.
    """
    engine = make_engine(FakeClient())
    engine.markets["0xcond"] = make_market()
    asyncio.run(engine._on_fill(make_fill(fee_rate_bps=Decimal("20"))))

    # Тот же рынок, заново собранный discovery: ставка снова нулевая.
    fresh = make_market()
    assert fresh.fee_rate == D("0")
    engine._apply_fee_override(fresh)
    assert fresh.fee_rate == D("0.002")

    # Расписание рынка объявило ставку выше выученной — верим расписанию.
    declared = make_market(fees_enabled=True, fee_rate=D("0.02"), fee_exponent=D("1"))
    engine._apply_fee_override(declared)
    assert declared.fee_rate == D("0.02")
    assert declared.fee_exponent == D("1")


def test_reported_fee_does_not_downgrade_known_rate():
    """Известную из расписания ставку сверка не трогает."""
    engine = make_engine(FakeClient())
    market = make_market(fees_enabled=True, fee_rate=D("0.02"), fee_exponent=D("1"))
    engine.markets["0xcond"] = market

    asyncio.run(engine._on_fill(make_fill(fee_rate_bps=Decimal("1"))))

    assert market.fee_rate == D("0.02")
    assert market.fee_exponent == D("1")


# --------------------------------------------- страйк: живая калибровка


def warm_spot(engine: TradingEngine, asset: str = "BTC",
              price: float = 109_500.0, n: int = 40) -> float:
    """Свежий прогретый спот: n тиков, последний — прямо сейчас."""
    now = time.time()
    for i in range(n):
        engine.spot.ingest(asset, price * (1 + 1e-5 * (i % 3)), now - 4 + i * 0.1)
    return engine.spot.price(asset)  # type: ignore[return-value]


def live_book(bid: str, ask: str) -> Book:
    return Book(
        token_id="tok_yes",
        bids=[BookLevel(D(bid), D("100"))],
        asks=[BookLevel(D(ask), D("80"))],
        updated_at=time.time(),
    )


def test_strike_observed_at_window_open_crossing():
    """Пересечение start_ts со свежим спотом: спот сейчас и есть страйк."""
    engine = make_engine(FakeClient())
    spot = warm_spot(engine)
    market = make_market(start_ts=time.time() - 1.0)

    engine._try_calibrate_strike(market, None)

    assert market.strike is not None
    assert abs(float(market.strike) - spot) < 0.011  # round до цента

    # Открытие, пропущенное больше допуска назад, задним числом не считается.
    late = make_market(condition_id="0xlate", start_ts=time.time() - 30.0)
    engine._try_calibrate_strike(late, None)   # и книги нет — implied не сможет
    assert late.strike is None


def test_strike_deferred_calibration_uses_live_book_and_sigma():
    """
    Отложенная инверсия GBM: живой mid нашего стакана + свежий спот + живая
    сигма. При mid около 0.5 страйк обязан лечь рядом со спотом.
    """
    engine = make_engine(FakeClient())
    spot = warm_spot(engine)
    market = make_market(start_ts=time.time() - 30.0)

    engine._try_calibrate_strike(market, live_book("0.50", "0.52"))

    assert market.strike is not None
    assert abs(float(market.strike) / spot - 1.0) < 0.01

    # До открытия окна mid ничего не знает о будущем страйке — не калибруем.
    pre = make_market(condition_id="0xpre", start_ts=time.time() + 60.0)
    engine._try_calibrate_strike(pre, live_book("0.50", "0.52"))
    assert pre.strike is None

    # Экстремальный mid: наклон inv_cdf на краю огромен — не калибруем.
    edge = make_market(condition_id="0xedge", start_ts=time.time() - 30.0)
    engine._try_calibrate_strike(edge, live_book("0.95", "0.97"))
    assert edge.strike is None


def test_strike_calibration_requires_fresh_spot():
    """Протухший или пустой спот-фид не даёт калибровать вовсе."""
    engine = make_engine(FakeClient())
    market = make_market(start_ts=time.time() - 30.0)

    engine._try_calibrate_strike(market, live_book("0.50", "0.52"))
    assert market.strike is None            # спота нет вовсе

    # Спот есть, но старше stale_price_timeout_s (4.0 в конфиге теста).
    stale_ts = time.time() - 30
    for i in range(40):
        engine.spot.ingest("BTC", 109_500.0, stale_ts - 4 + i * 0.1)
    engine._try_calibrate_strike(market, live_book("0.50", "0.52"))
    assert market.strike is None


# --------------------------------------------- страйк: сторож расхождения


class WatchdogCfg(Cfg):
    class strategy(Cfg.strategy):
        strike_divergence_hold_s = 0.05


def diverged_fv() -> "engine_mod.FairValueModel":
    from src.models import FairValue
    return FairValue(
        fair=D("0.43"), model_prob=D("0.96"), market_mid=D("0.28"),
        edge=D("0.68"), sigma_annual=D("0.45"), confidence=D("1"),
    )


def agreeing_fv() -> "engine_mod.FairValueModel":
    from src.models import FairValue
    return FairValue(
        fair=D("0.29"), model_prob=D("0.30"), market_mid=D("0.28"),
        edge=D("0.02"), sigma_annual=D("0.45"), confidence=D("1"),
    )


def test_divergence_watchdog_needs_persistence():
    """Разовый выброс не снимает страйк: таймер сбрасывается возвратом."""
    engine = TradingEngine(WatchdogCfg())  # type: ignore[arg-type]
    engine.client = FakeClient()  # type: ignore[assignment]
    market = make_market(strike=D("100000"))
    mid = D("0.28")

    assert engine._check_model_divergence(market, diverged_fv(), mid, 109500.0) is False
    assert market.strike is not None       # только взвод таймера
    assert engine._check_model_divergence(market, agreeing_fv(), mid, 109500.0) is False
    assert engine._strike_meta["0xcond"]["diverged_since"] is None  # сброшен


def test_divergence_watchdog_invalidates_then_blocks(caplog):
    """Устойчивое расхождение: инвалидация; повторное — блок до конца окна."""
    import logging as _logging

    engine = TradingEngine(WatchdogCfg())  # type: ignore[arg-type]
    engine.client = FakeClient()  # type: ignore[assignment]
    market = make_market(strike=D("100000"))
    mid = D("0.28")

    with caplog.at_level(_logging.WARNING, logger="polybot.engine"):
        assert engine._check_model_divergence(market, diverged_fv(), mid, 109500.0) is False
        time.sleep(0.06)
        assert engine._check_model_divergence(market, diverged_fv(), mid, 109500.0) is True

    assert market.strike is None
    meta = engine._strike_meta["0xcond"]
    assert meta["invalidations"] == 1 and not meta["blocked"]
    assert any("страйк невалиден" in r.message for r in caplog.records)

    # Повторный срыв (например, после рекалибровки) — блок до конца окна.
    market.strike = D("100000")
    engine._check_model_divergence(market, diverged_fv(), mid, 109500.0)
    time.sleep(0.06)
    assert engine._check_model_divergence(market, diverged_fv(), mid, 109500.0) is True
    assert engine._strike_meta["0xcond"]["blocked"] is True

    # Блок закрывает и рекалибровку: даже идеальные входы не вернут модель.
    warm_spot(engine)
    market.start_ts = time.time() - 30.0
    engine._try_calibrate_strike(market, live_book("0.50", "0.52"))
    assert market.strike is None
