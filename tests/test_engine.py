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

from src import engine as engine_mod  # noqa: E402
from src.engine import TradingEngine  # noqa: E402
from src.models import MarketPosition, TargetMarket  # noqa: E402
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

    def list_positions(self):
        outer = self

        class _Iter:
            def __aiter__(self):
                if outer._fail is not None:
                    raise outer._fail
                return self._gen()

            async def _gen(self):
                for p in outer._positions:
                    yield p

        return _Iter()

    async def merge_positions(self, *, condition_id: str, amount: int):
        self.merge_calls.append({"condition_id": condition_id, "amount": amount})
        return ApiPosition(transaction_hash="0xdead")


def make_engine(client: FakeClient) -> TradingEngine:
    engine = TradingEngine(Cfg())  # type: ignore[arg-type]
    engine.client = client  # type: ignore[assignment]
    return engine


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

    asyncio.run(
        engine._on_fill(
            "0xcond", "tok_yes", "BUY", D("0.49"), D("10"), Decimal("20")
        )
    )

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
    asyncio.run(
        engine._on_fill(
            "0xcond", "tok_yes", "BUY", D("0.49"), D("10"), Decimal("20")
        )
    )

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

    asyncio.run(
        engine._on_fill(
            "0xcond", "tok_yes", "BUY", D("0.49"), D("10"), Decimal("1")
        )
    )

    assert market.fee_rate == D("0.02")
    assert market.fee_exponent == D("1")
