"""
Офлайн-тесты стратегии. Сеть и ключи не нужны.

Запуск:  pytest tests/ -v
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from src.fair_value import FairValueModel, VolatilityEstimator, _norm_cdf
from src.models import (
    Book,
    BookLevel,
    FairValue,
    MarketPosition,
    RecoveredPosition,
    TargetMarket,
    outcome_label,
    shares_to_base_units,
)
from src.quoting import QuoteGenerator, round_to_tick
from src.regime import Regime, RegimeState
from src.risk import HaltReason, RiskManager

D = Decimal


# ----------------------------------------------------------------- фикстуры


class StratCfg:
    target_pair_cost = D("0.985")
    max_pair_cost = D("0.995")
    min_half_spread_ticks = 1
    order_size = D("20")
    inventory_skew_coef = D("0.010")
    allow_directional = True
    directional_min_edge = D("0.025")
    directional_max_net = D("60")
    ladder_levels = 1
    ladder_step_ticks = 2
    ladder_level_size = D("0")
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


class RiskCfg:
    max_position_per_side = D("250")
    max_net_exposure = D("120")
    max_notional = D("500")
    daily_loss_limit = D("50")
    max_open_orders = 16
    heartbeat_timeout_s = 6.0
    stale_price_timeout_s = 4.0
    stale_book_timeout_s = 15.0
    max_consecutive_rejects = 8


@pytest.fixture
def market() -> TargetMarket:
    return TargetMarket(
        condition_id="0xcond",
        slug="bitcoin-up-or-down-test",
        question="Bitcoin Up or Down?",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        end_ts=time.time() + 300,
        tick_size=D("0.01"),
        min_order_size=D("5"),
        neg_risk=False,
        asset="BTC",
        strike=D("100000"),
    )


def fee_market(rate: str, exponent: str = "1") -> TargetMarket:
    """Тот же рынок, но с включённой комиссией."""
    return TargetMarket(
        condition_id="0xcond",
        slug="bitcoin-up-or-down-fee",
        question="Bitcoin Up or Down?",
        yes_token_id="tok_yes",
        no_token_id="tok_no",
        end_ts=time.time() + 300,
        tick_size=D("0.01"),
        min_order_size=D("5"),
        neg_risk=False,
        asset="BTC",
        strike=D("100000"),
        fees_enabled=True,
        fee_rate=D(rate),
        fee_exponent=D(exponent),
    )


class FakeApiPosition:
    """Позиция в том виде, в каком её отдаёт SDK (нам важны только поля)."""

    def __init__(self, **fields) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


@pytest.fixture
def books() -> tuple[Book, Book]:
    """Зеркальные стаканы: bid_YES=0.50/ask_YES=0.52 <=> bid_NO=0.48/ask_NO=0.50."""
    yes = Book(
        token_id="tok_yes",
        bids=[BookLevel(D("0.50"), D("100")), BookLevel(D("0.49"), D("200"))],
        asks=[BookLevel(D("0.52"), D("100")), BookLevel(D("0.53"), D("200"))],
        updated_at=time.time(),
    )
    no = Book(
        token_id="tok_no",
        bids=[BookLevel(D("0.48"), D("100")), BookLevel(D("0.47"), D("200"))],
        asks=[BookLevel(D("0.50"), D("100")), BookLevel(D("0.51"), D("200"))],
        updated_at=time.time(),
    )
    return yes, no


# ------------------------------------------------------------ fair value


def test_norm_cdf_sanity():
    assert _norm_cdf(0.0) == pytest.approx(0.5)
    assert _norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert _norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_at_the_money_is_coinflip():
    """Спот == страйк => вероятность близка к 50%."""
    m = FairValueModel(D("1.0"), D("0"), D("1.0"))
    p = m.model_probability(100_000, 100_000, 300, 0.5, 0.0)
    assert p == pytest.approx(0.5, abs=0.01)


def test_deep_in_the_money():
    """Спот сильно выше страйка на коротком горизонте => вероятность -> 1."""
    m = FairValueModel(D("1.0"), D("0"), D("1.0"))
    p = m.model_probability(105_000, 100_000, 60, 0.5, 0.0)
    assert p > 0.95


def test_more_time_pulls_toward_half():
    """Чем больше времени, тем ближе к 0.5 при том же отклонении."""
    m = FairValueModel(D("1.0"), D("0"), D("1.0"))
    near = m.model_probability(101_000, 100_000, 60, 0.6, 0.0)
    far = m.model_probability(101_000, 100_000, 900, 0.6, 0.0)
    assert abs(far - 0.5) < abs(near - 0.5)


def test_model_never_explodes_at_expiry():
    """tau -> 0 не должно давать NaN/inf."""
    m = FairValueModel(D("1.0"), D("0"), D("1.0"))
    for secs in (0.0, 0.001, 1.0, 2.0):
        p = m.model_probability(100_050, 100_000, secs, 0.5, 0.0)
        assert 0.0 <= p <= 1.0


def test_deviation_from_market_is_clipped():
    """Даже при абсурдной модели fair не уходит дальше max_deviation."""
    m = FairValueModel(model_weight=D("1.0"), momentum_drift_coef=D("0"),
                       max_model_deviation=D("0.05"))
    fv = m.compute(
        spot=150_000, strike=100_000, seconds_left=300,
        market_mid=D("0.50"), sigma_annual=0.5,
        drift_per_second=0.0, vol_ready=True,
    )
    assert abs(fv.fair - D("0.50")) <= D("0.05")


def test_confidence_drops_near_expiry():
    m = FairValueModel(D("0.35"), D("0.3"), D("0.15"))
    assert m.confidence(10, True) < m.confidence(300, True)
    assert m.confidence(300, False) == 0.0


def test_vol_estimator_converges():
    """Оценщик должен реагировать на реальные движения цены."""
    est = VolatilityEstimator(45.0, 8.0, 0.30)
    price, ts = 100_000.0, time.time()
    for i in range(400):
        price *= 1.0 + (0.0004 if i % 2 == 0 else -0.0004)
        est.update(price, ts + i)
    assert est.ready
    assert est.sigma_annual > 0.05


# ---------------------------------------------------------------- quoting


def test_pair_cost_invariant(market, books):
    """ГЛАВНЫЙ ИНВАРИАНТ: сумма двух бидов всегда < max_pair_cost."""
    q = QuoteGenerator(StratCfg(), RiskCfg())
    m = FairValueModel(D("0.35"), D("0.3"), D("0.15"))
    yes_book, no_book = books
    pos = MarketPosition(condition_id="0xcond")

    for mid in ("0.20", "0.35", "0.50", "0.65", "0.85"):
        fv = m.compute(
            spot=100_000, strike=100_000, seconds_left=300,
            market_mid=D(mid), sigma_annual=0.5,
            drift_per_second=0.0, vol_ready=True,
        )
        quotes = q.build_quotes(market, fv, pos, yes_book, no_book)
        if not quotes:
            continue
        total = sum(x.price for x in quotes)
        assert total < StratCfg.max_pair_cost, f"mid={mid} дал пару {total}"


def test_quotes_never_cross_the_ask(market, books):
    """Post-only ордер, пересекающий ask, был бы отклонён биржей."""
    q = QuoteGenerator(StratCfg(), RiskCfg())
    m = FairValueModel(D("0.35"), D("0.3"), D("0.15"))
    yes_book, no_book = books
    fv = m.compute(
        spot=100_000, strike=100_000, seconds_left=300, market_mid=D("0.51"),
        sigma_annual=0.5, drift_per_second=0.0, vol_ready=True,
    )
    quotes = q.build_quotes(market, fv, MarketPosition("0xcond"), yes_book, no_book)
    for quote in quotes:
        book = yes_book if quote.outcome == "YES" else no_book
        assert quote.price < book.best_ask


def test_inventory_skew_direction():
    """Лонг YES => резервная цена ниже, чтобы разгружать позицию."""
    q = QuoteGenerator(StratCfg(), RiskCfg())
    m = FairValueModel(D("0.35"), D("0"), D("0.15"))
    fv = m.compute(
        spot=100_000, strike=100_000, seconds_left=300, market_mid=D("0.50"),
        sigma_annual=0.5, drift_per_second=0.0, vol_ready=True,
    )
    flat = MarketPosition("c", yes_size=D("0"), no_size=D("0"))
    long_yes = MarketPosition("c", yes_size=D("100"), no_size=D("0"))

    r_flat = q.reservation_price(fv.fair, flat, fv)
    r_long = q.reservation_price(fv.fair, long_yes, fv)
    assert r_long < r_flat


def test_round_to_tick():
    assert round_to_tick(D("0.4967"), D("0.01")) == D("0.49")
    assert round_to_tick(D("0.5"), D("0.01")) == D("0.50")


# ------------------------------------------------------------- разгрузка


def test_unwind_sell_does_not_cross_the_bid(market, books):
    """
    Разгрузка — post_only SELL. Цена, равная лучшему биду, marketable:
    биржа отклонит ордер, и разгрузка не исполнится НИКОГДА. Продаём на
    тик выше бида — самый агрессивный некроссящий аск.
    """
    q = QuoteGenerator(StratCfg(), RiskCfg())
    yes_book, _ = books
    pos = MarketPosition("0xcond")
    pos.apply_fill("YES", "BUY", D("0.49"), D("100"))   # net +100

    quotes = q.build_unwind_quotes(market, pos, yes_book)
    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.side == "SELL" and quote.outcome == "YES"
    assert quote.price > yes_book.best_bid              # не кроссит
    assert quote.price == yes_book.best_bid + market.tick_size
    assert quote.size == D("40")                        # min(|net|, 2*order)


def test_unwind_at_price_ceiling_returns_nothing(market):
    """Бид у потолка: некроссящего мейкер-аска не существует — молчим."""
    book = Book(
        token_id="tok_yes",
        bids=[BookLevel(D("0.99"), D("100"))],
        asks=[],
        updated_at=time.time(),
    )
    pos = MarketPosition("0xcond")
    pos.apply_fill("YES", "BUY", D("0.90"), D("100"))
    q = QuoteGenerator(StratCfg(), RiskCfg())
    assert q.build_unwind_quotes(market, pos, book) == []


# --------------------------------------------------------- реакция на режим


def regime_state(regime: Regime, crowded=None) -> RegimeState:
    return RegimeState(
        regime=regime, crowded_side=crowded, imbalance=0.9,
        fills_in_window=8, vol_ratio=None, autocorr=None,
    )


def fv_at_half():
    m = FairValueModel(D("0.35"), D("0"), D("0.15"))
    return m.compute(
        spot=100_000, strike=100_000, seconds_left=300, market_mid=D("0.50"),
        sigma_annual=0.5, drift_per_second=0.0, vol_ready=True,
    )


class TrendingOnCfg(StratCfg):
    regime_trending_response = True   # тесты реакции включают её явно


def test_trending_widens_crowded_and_tightens_starving(market):
    """
    TRENDING, нас засыпает NO: NO-бид отодвигается (не подставляемся под
    вынос), YES-бид подтягивается к рынку (достраивает пары к накопленному).
    Книги не передаём, чтобы видеть чистый эффект асимметрии.
    """
    q = QuoteGenerator(TrendingOnCfg(), RiskCfg())
    fv = fv_at_half()
    pos = MarketPosition("0xcond")

    base = {x.outcome: x.price for x in q.build_quotes(market, fv, pos, None, None)}
    trend = {
        x.outcome: x.price
        for x in q.build_quotes(
            market, fv, pos, None, None,
            regime_state(Regime.TRENDING, crowded="NO"),
        )
    }

    assert trend["NO"] < base["NO"]        # заваленная сторона шире
    assert trend["YES"] > base["YES"]      # голодная — ближе к рынку
    # extra >= tighten: сумма пары от асимметрии не растёт.
    assert trend["YES"] + trend["NO"] <= base["YES"] + base["NO"]


def test_trending_asymmetry_keeps_fee_invariant(books):
    """Асимметрия не имеет права ломать главный инвариант, и с комиссией."""
    q = QuoteGenerator(TrendingOnCfg(), RiskCfg())
    fv = fv_at_half()
    yes_book, no_book = books
    market = fee_market("0.02")

    for crowded in ("YES", "NO"):
        quotes = q.build_quotes(
            market, fv, MarketPosition("0xcond"), yes_book, no_book,
            regime_state(Regime.TRENDING, crowded=crowded),
        )
        if len(quotes) < 2:
            continue
        bids = sum(x.price for x in quotes)
        fee = market.fee_per_pair(quotes[0].price, quotes[1].price)
        assert bids + fee < StratCfg.max_pair_cost


def test_trending_response_flag_off_keeps_symmetry(market):
    """Реакция отключаема флагом: дефолт StratCfg — выключено."""
    q = QuoteGenerator(StratCfg(), RiskCfg())   # regime_trending_response=False
    fv = fv_at_half()
    pos = MarketPosition("0xcond")

    base = {x.outcome: x.price for x in q.build_quotes(market, fv, pos, None, None)}
    trend = {
        x.outcome: x.price
        for x in q.build_quotes(
            market, fv, pos, None, None,
            regime_state(Regime.TRENDING, crowded="NO"),
        )
    }
    assert base == trend


def test_trending_remove_crowded_quotes_single_side(market):
    """Строгий вариант: заваленная сторона снимается, остаётся одна нога."""
    class RemoveCfg(TrendingOnCfg):
        trending_remove_crowded = True

    q = QuoteGenerator(RemoveCfg(), RiskCfg())
    quotes = q.build_quotes(
        market, fv_at_half(), MarketPosition("0xcond"), None, None,
        regime_state(Regime.TRENDING, crowded="NO"),
    )
    assert len(quotes) == 1
    assert quotes[0].outcome == "YES" and quotes[0].side == "BUY"


def test_volatile_regime_stops_quoting(market):
    """В VOLATILE не котируем; механизм отключаем своим флагом."""
    q = QuoteGenerator(StratCfg(), RiskCfg())
    fv = fv_at_half()
    pos = MarketPosition("0xcond")

    assert q.build_quotes(
        market, fv, pos, None, None, regime_state(Regime.VOLATILE)
    ) == []

    class NoGateCfg(StratCfg):
        regime_volatile_no_quote = False

    q2 = QuoteGenerator(NoGateCfg(), RiskCfg())
    assert q2.build_quotes(
        market, fv, pos, None, None, regime_state(Regime.VOLATILE)
    ) != []


def test_trending_without_crowded_side_is_symmetric(market):
    """Сторона неизвестна (поток обнулился) — асимметрию не применяем."""
    q = QuoteGenerator(TrendingOnCfg(), RiskCfg())
    fv = fv_at_half()
    pos = MarketPosition("0xcond")
    base = {x.outcome: x.price for x in q.build_quotes(market, fv, pos, None, None)}
    trend = {
        x.outcome: x.price
        for x in q.build_quotes(
            market, fv, pos, None, None,
            regime_state(Regime.TRENDING, crowded=None),
        )
    }
    assert base == trend


# -------------------------------------------------------------- комиссии


def test_fee_per_share_peaks_at_the_money():
    """Комиссия rate*(p*(1-p))**exp максимальна у 0.50 и падает к краям."""
    m = fee_market("0.02")
    assert m.fee_per_share(D("0.50")) > m.fee_per_share(D("0.80"))
    assert m.fee_per_share(D("0.50")) > m.fee_per_share(D("0.20"))
    # Рынок без комиссии не берёт ничего ни на одной цене.
    free = fee_market("0")
    assert free.fee_per_share(D("0.50")) == D("0")


def test_pair_cost_invariant_with_fees(books):
    """
    ГЛАВНЫЙ ИНВАРИАНТ НА FEE-РЫНКЕ: биды ПЛЮС комиссия обеих ног всегда
    ниже max_pair_cost. Без этого пара собирается с отрицательной чистой
    маржой, и по логам это не видно — там сумма бидов выглядит нормально.
    """
    q = QuoteGenerator(StratCfg(), RiskCfg())
    m = FairValueModel(D("0.35"), D("0.3"), D("0.15"))
    yes_book, no_book = books
    pos = MarketPosition(condition_id="0xcond")

    for rate in ("0.001", "0.005", "0.01", "0.02"):
        market = fee_market(rate)
        for mid in ("0.20", "0.35", "0.50", "0.65", "0.85"):
            fv = m.compute(
                spot=100_000, strike=100_000, seconds_left=300,
                market_mid=D(mid), sigma_annual=0.5,
                drift_per_second=0.0, vol_ready=True,
            )
            quotes = q.build_quotes(market, fv, pos, yes_book, no_book)
            if not quotes:
                continue
            bids = sum(x.price for x in quotes)
            fee = market.fee_per_pair(quotes[0].price, quotes[1].price)
            assert bids + fee < StratCfg.max_pair_cost, (
                f"rate={rate} mid={mid}: биды {bids} + комиссия {fee} >= планки"
            )


def test_fee_market_is_quoted_wider(books):
    """Комиссия должна раздвигать спред, а не съедать маржу молча."""
    q = QuoteGenerator(StratCfg(), RiskCfg())
    m = FairValueModel(D("0.35"), D("0"), D("0.15"))
    yes_book, no_book = books
    fv = m.compute(
        spot=100_000, strike=100_000, seconds_left=300, market_mid=D("0.50"),
        sigma_annual=0.5, drift_per_second=0.0, vol_ready=True,
    )
    pos = MarketPosition("0xcond")
    free = q.build_quotes(fee_market("0"), fv, pos, yes_book, no_book)
    paid = q.build_quotes(fee_market("0.02"), fv, pos, yes_book, no_book)

    assert free and paid
    assert sum(x.price for x in paid) < sum(x.price for x in free)


def test_prohibitive_fee_stops_quoting(books):
    """
    Если комиссия съедает всю планку — не котируем вовсе.
    Проверка обязана проваливаться в сторону остановки торговли.
    """
    q = QuoteGenerator(StratCfg(), RiskCfg())
    m = FairValueModel(D("0.35"), D("0"), D("0.15"))
    yes_book, no_book = books
    fv = m.compute(
        spot=100_000, strike=100_000, seconds_left=300, market_mid=D("0.50"),
        sigma_annual=0.5, drift_per_second=0.0, vol_ready=True,
    )
    # Плоская ставка 0.60 на ногу — дороже любой возможной маржи пары.
    ruinous = fee_market("0.60", exponent="0")
    assert q.build_quotes(ruinous, fv, MarketPosition("0xcond"), yes_book, no_book) == []


def test_fee_is_capitalized_into_pair_basis():
    """Комиссия покупки входит в себестоимость пары, а не теряется."""
    market = fee_market("0.02")
    pos = MarketPosition("c")
    for outcome, price in (("YES", D("0.49")), ("NO", D("0.48"))):
        pos.apply_fill(
            outcome, "BUY", price, D("100"), market.fee_for(price, D("100"))
        )

    basis = pos.pair_cost_basis()
    assert basis is not None
    # Без комиссии себестоимость была бы ровно 0.97.
    assert basis > D("0.97")
    assert basis == pytest.approx(
        D("0.97") + market.fee_per_pair(D("0.49"), D("0.48")), abs=D("1e-9")
    )
    assert pos.fees_paid > 0


# --------------------------------------------------------------- позиции


def test_complete_pairs_and_merge():
    pos = MarketPosition("c")
    pos.apply_fill("YES", "BUY", D("0.49"), D("100"))
    pos.apply_fill("NO", "BUY", D("0.48"), D("100"))

    assert pos.complete_pairs == D("100")
    assert pos.net == D("0")
    assert pos.pair_cost_basis() == D("0.97")

    pos.apply_merge(D("100"))
    # 100 пар по 0.97 -> $100. Прибыль 3 USDC.
    assert pos.realized_pnl == pytest.approx(D("3"))
    assert pos.yes_size == D("0")
    assert pos.merged_pairs == D("100")


def test_partial_pair_leaves_directional_exposure():
    pos = MarketPosition("c")
    pos.apply_fill("YES", "BUY", D("0.49"), D("100"))
    pos.apply_fill("NO", "BUY", D("0.48"), D("40"))
    assert pos.complete_pairs == D("40")
    assert pos.net == D("60")  # голый лонг YES


# ---------------------------------------------------------- merge и газ


def test_merge_gas_reduces_realized_pnl():
    """Газ merge — издержка, а не округление: он обязан попасть в PnL."""
    pos = MarketPosition("c")
    pos.apply_fill("YES", "BUY", D("0.49"), D("100"))
    pos.apply_fill("NO", "BUY", D("0.48"), D("100"))
    pos.apply_merge(D("100"), gas_cost=D("0.35"))

    # 100 пар по 0.97 -> $100 валовых 3.00, минус газ 0.35.
    assert pos.realized_pnl == pytest.approx(D("2.65"))
    assert pos.merge_costs == D("0.35")


def test_merge_amount_is_in_base_units():
    """
    merge_positions() ждёт amount в базовых единицах ERC-1155 (6 знаков),
    а не в shares. Передать shares значит смержить миллионную долю пачки,
    заплатив при этом полный газ.
    """
    assert shares_to_base_units(D("25")) == 25_000_000
    assert shares_to_base_units(D("0.5")) == 500_000
    assert shares_to_base_units(D("0")) == 0
    assert shares_to_base_units(D("-5")) == 0
    # Дробь тоньше базовой единицы округляется вниз, а не вверх:
    # мержить больше, чем есть на балансе, биржа не даст.
    assert shares_to_base_units(D("1.0000005")) == 1_000_000


# ------------------------------------------- восстановление после рестарта


def test_recovered_position_enters_risk_base():
    """
    Позиция прошлой сессии должна попасть в учёт: иначе лимиты риска
    считаются от нуля, а бот докупает поверх уже открытой позиции.
    """
    pos = MarketPosition("c")
    pos.apply_recovered("YES", D("100"), D("0.40"))

    assert pos.yes_size == D("100")
    assert pos.net == D("100")
    assert pos.total_cost == D("40")
    # PnL прошлой сессии — не наш результат: дневной лимит убытка
    # должен отсчитываться от старта.
    assert pos.realized_pnl == D("0")

    r = RiskManager(RiskCfg())
    r.heartbeat()
    assert r.check_global({"c": pos}, 0)          # net 100 < лимита 120

    # Ещё 50 shares с прошлой сессии — и лимит net exposure пробит сразу,
    # ДО первой собственной сделки. Ровно этого раньше не происходило.
    pos.apply_recovered("YES", D("50"), D("0.40"))
    assert not r.check_global({"c": pos}, 0)      # net 150 > лимита 120
    assert r.state.reason == HaltReason.NET_EXPOSURE


def test_recovered_position_without_price_is_conservative():
    """Нет средней цены — считаем по 1.0: нотионал завышен, лимиты строже."""
    pos = MarketPosition("c")
    pos.apply_recovered("NO", D("80"), None)
    assert pos.no_cost == D("80")

    zero_priced = MarketPosition("c2")
    zero_priced.apply_recovered("NO", D("80"), D("0"))
    assert zero_priced.no_cost == D("80")


def test_recovered_pairs_are_mergeable():
    """Восстановленные YES+NO — это готовая пара, её можно сразу мержить."""
    pos = MarketPosition("c")
    pos.apply_recovered("YES", D("50"), D("0.49"))
    pos.apply_recovered("NO", D("50"), D("0.48"))
    assert pos.complete_pairs == D("50")
    assert pos.pair_cost_basis() == D("0.97")


def test_outcome_label_reads_up_and_down():
    """Up/Down-серии подписывают исходы не 'Yes'/'No', а 'Up'/'Down'."""
    assert outcome_label(FakeApiPosition(outcome="Up")) == "YES"
    assert outcome_label(FakeApiPosition(outcome="Down")) == "NO"
    assert outcome_label(FakeApiPosition(outcome="Yes")) == "YES"
    assert outcome_label(FakeApiPosition(outcome="No")) == "NO"
    # Незнакомый ярлык -> падаем на индекс исхода.
    assert outcome_label(FakeApiPosition(outcome="???", outcome_index=1)) == "NO"
    # Нет ни ярлыка, ни индекса -> сторона неизвестна, решим по token_id.
    assert outcome_label(FakeApiPosition(outcome=None)) is None


def test_recovered_position_parsing():
    """Разбор ответа биржи не должен падать на отсутствующих полях."""
    rec = RecoveredPosition.from_api(
        FakeApiPosition(
            condition_id="0xabc", token_id="tok_yes", size="12.5",
            avg_price="0.42", outcome="Up", title="BTC Up or Down",
            redeemable=False,
        )
    )
    assert rec is not None
    assert (rec.size, rec.avg_price, rec.outcome) == (D("12.5"), D("0.42"), "YES")

    # Пустая позиция и позиция без condition_id в учёт не идут.
    assert RecoveredPosition.from_api(FakeApiPosition(condition_id="0xabc", size="0")) is None
    assert RecoveredPosition.from_api(FakeApiPosition(size="10")) is None

    # Резолвленную позицию движок пометит и не станет заводить в лимиты.
    resolved = RecoveredPosition.from_api(
        FakeApiPosition(condition_id="0xabc", size="10", redeemable=True)
    )
    assert resolved is not None and resolved.redeemable


# ------------------------------------------------------------------ риск


def test_daily_loss_halts_trading():
    r = RiskManager(RiskCfg())
    assert not r.is_halted
    r.record_realized(D("-51"))
    assert r.is_halted
    assert r.state.reason == HaltReason.DAILY_LOSS


def test_net_exposure_halts_trading():
    r = RiskManager(RiskCfg())
    r.heartbeat()
    pos = MarketPosition("c", yes_size=D("200"), no_size=D("0"))
    assert not r.check_global({"c": pos}, 0)
    assert r.state.reason == HaltReason.NET_EXPOSURE


def test_consecutive_rejects_halt():
    r = RiskManager(RiskCfg())
    for _ in range(8):
        r.record_reject()
    assert r.is_halted
    assert r.state.reason == HaltReason.REJECTS


def test_heartbeat_dead_man_switch():
    r = RiskManager(RiskCfg())
    r._last_heartbeat = time.time() - 100
    assert r.heartbeat_expired()
    assert not r.check_global({}, 0)
    assert r.state.reason == HaltReason.HEARTBEAT


def test_clamp_order_size_respects_limits():
    r = RiskManager(RiskCfg())
    # Свободного места по стороне 50 -> ордер урезается.
    assert r.clamp_order_size(D("100"), D("200"), D("0"), False) == D("50")
    # Сторона забита -> ордер запрещён.
    assert r.clamp_order_size(D("100"), D("250"), D("0"), False) == D("0")
    # Увеличение net при net=120 запрещено.
    assert r.clamp_order_size(D("50"), D("0"), D("120"), True) == D("0")


def test_market_gate_rejects_wide_spread(market):
    r = RiskManager(RiskCfg())
    ok, reason = r.can_quote_market(
        market, book_stale=False, price_stale=False,
        spread=D("0.20"), depth=D("100"),
        max_spread=D("0.06"), min_depth=D("50"),
    )
    assert not ok and "спред" in reason


def test_market_gate_rejects_stale_feed(market):
    r = RiskManager(RiskCfg())
    ok, _ = r.can_quote_market(
        market, book_stale=False, price_stale=True,
        spread=D("0.02"), depth=D("100"),
        max_spread=D("0.06"), min_depth=D("50"),
    )
    assert not ok


# ---------------------------------------------------------------- стакан


def test_books_are_complementary(books):
    """Проверка тождества: ask_NO == 1 - bid_YES."""
    yes, no = books
    assert no.best_ask == D("1") - yes.best_bid
    assert no.best_bid == D("1") - yes.best_ask


def test_microprice_leans_toward_thin_side():
    book = Book(
        token_id="t",
        bids=[BookLevel(D("0.50"), D("10"))],
        asks=[BookLevel(D("0.52"), D("500"))],
        updated_at=time.time(),
    )
    # Огромный ask давит цену вниз относительно mid 0.51.
    assert book.microprice() < book.mid


# ------------------------------------- диагностика сломанного fair value


def test_pair_sum_far_below_target_warns_but_still_quotes(market, caplog):
    """
    Fair съехал от рынка (ложный страйк + клип): каждую ногу зажимает её
    стакан далеко от резервной цены, сумма пары падает сильно ниже цели.
    Инвариант сверху цел, но экономически котировки мертвы — обязателен
    WARNING (без блокировки котирования и без спама: не чаще раза в 30 с).
    """
    import logging as _logging

    q = QuoteGenerator(StratCfg(), RiskCfg())
    fv = FairValue(
        fair=D("0.44"), model_prob=D("0.96"), market_mid=D("0.28"),
        edge=D("0.68"), sigma_annual=D("0.45"), confidence=D("1"),
    )
    yes = Book(token_id="tok_yes",
               bids=[BookLevel(D("0.27"), D("100"))],
               asks=[BookLevel(D("0.29"), D("100"))], updated_at=time.time())
    no = Book(token_id="tok_no",
              bids=[BookLevel(D("0.71"), D("100"))],
              asks=[BookLevel(D("0.73"), D("100"))], updated_at=time.time())
    pos = MarketPosition("0xcond")

    with caplog.at_level(_logging.WARNING, logger="polybot.quote"):
        quotes = q.build_quotes(market, fv, pos, yes, no)
        quotes_again = q.build_quotes(market, fv, pos, yes, no)

    assert len(quotes) == 2                     # диагностика не блокирует
    pair_sum = sum(quote.price for quote in quotes)
    assert pair_sum < StratCfg.target_pair_cost - StratCfg.pair_sum_warn_gap
    warned = [r for r in caplog.records if "ниже цели" in r.message]
    assert len(warned) == 1                     # антиспам: второй вызов молчит
    assert quotes_again                          # но котирует по-прежнему


def test_healthy_pair_sum_does_not_warn(market, books, caplog):
    """Нормальная пара у цели — никаких предупреждений."""
    import logging as _logging

    q = QuoteGenerator(StratCfg(), RiskCfg())
    yes, no = books
    fv = FairValue(
        fair=D("0.51"), model_prob=D("0.51"), market_mid=D("0.51"),
        edge=D("0"), sigma_annual=D("0.45"), confidence=D("1"),
    )
    with caplog.at_level(_logging.WARNING, logger="polybot.quote"):
        quotes = q.build_quotes(market, fv, MarketPosition("0xcond"), yes, no)
    assert len(quotes) == 2
    assert not [r for r in caplog.records if "ниже цели" in r.message]


# ------------------------------------------------------- TWAP-модель


def test_twap_atm_at_window_open_is_half():
    """В начале окна (alpha=0) при S == K вероятность ~0.5."""
    from src.fair_value import twap_probability

    p = twap_probability(100_000.0, 100_000.0, 300.0, 0.5, alpha=0.0)
    assert p == pytest.approx(0.5, abs=0.01)


def test_twap_variance_is_tighter_than_endpoint_from_start():
    """
    Даже при alpha=0 дисперсия среднего втрое меньше дисперсии конца:
    при том же отклонении S от K TWAP-модель увереннее точечной.
    """
    from src.fair_value import twap_probability

    m = FairValueModel(D("1.0"), D("0"), D("1.0"))
    s, k = 100_100.0, 100_000.0
    p_end = m.model_probability(s, k, 300, 0.5, 0.0)
    p_twap = twap_probability(s, k, 300.0, 0.5, alpha=0.0)
    assert p_twap > p_end > 0.5


def test_twap_confidence_grows_faster_toward_window_end():
    """
    Цена держится чуть выше страйка всё окно. По мере накопления
    реализованной части уверенность TWAP-модели растёт быстрее точечной
    при том же оставшемся времени, и растёт монотонно с alpha.
    """
    from src.fair_value import twap_probability

    m = FairValueModel(D("1.0"), D("0"), D("1.0"))
    k = 100_000.0
    s = avg = k * 1.001
    window = 300.0

    probs = []
    for alpha in (0.25, 0.5, 0.75, 0.9):
        left = window * (1.0 - alpha)
        p_end = m.model_probability(s, k, left, 0.5, 0.0)
        p_twap = twap_probability(s, k, left, 0.5, alpha=alpha, realized_avg=avg)
        assert p_twap > p_end, f"alpha={alpha}: {p_twap} <= {p_end}"
        probs.append(p_twap)
    assert probs == sorted(probs)          # монотонно растёт с alpha


def test_twap_locked_average_dominates_current_spot():
    """
    Набранный TWAP далеко выше страйка => вероятность ~1 независимо от
    текущего спота (реализованную часть уже не отменить).
    """
    from src.fair_value import twap_probability

    k = 100_000.0
    # Спот обвалился ниже страйка, но 90% окна среднее было на 2% выше.
    p = twap_probability(k * 0.99, k, 30.0, 0.5, alpha=0.9,
                         realized_avg=k * 1.02)
    assert p > 0.99
    # K_eff <= 0: даже нулевые будущие цены не спасут NO — ровно 1.
    p_locked = twap_probability(k * 0.5, k, 30.0, 0.5, alpha=0.9,
                                realized_avg=k * 1.15)
    assert p_locked == 1.0


def test_realized_twap_step_integration_and_coverage():
    from src.fair_value import RealizedTwap

    acc = RealizedTwap(start_ts=100.0, end_ts=400.0)
    acc.update(100.0, 98.0)      # тик ДО старта окна: держится до старта
    acc.update(100.0, 100.0)
    acc.update(200.0, 110.0)     # 10 секунд по 100
    alpha, avg = acc.state(120.0)  # ещё 10 секунд по 200
    assert alpha == pytest.approx(20.0 / 300.0)
    assert avg == pytest.approx(150.0)

    # За концом окна интеграл не растёт, alpha зажат единицей.
    acc.update(500.0, 390.0)
    alpha_end, avg_end = acc.state(1000.0)
    assert alpha_end == 1.0
    assert 100.0 < avg_end < 500.0

    # Начало окна не покрыто — реализованная часть неизвестна: None.
    late = RealizedTwap(start_ts=100.0, end_ts=400.0)
    late.update(100.0, 130.0)    # первый тик через 30 секунд после старта
    assert late.state(200.0) is None


def test_realized_twap_before_window_start_is_alpha_zero():
    from src.fair_value import RealizedTwap

    acc = RealizedTwap(start_ts=100.0, end_ts=400.0)
    acc.update(123.0, 50.0)
    state = acc.state(90.0)
    assert state is not None
    alpha, last = state
    assert alpha == 0.0 and last == 123.0


def test_implied_strike_twap_roundtrip():
    """Инверсия и прямая модель согласованы: K(p) -> p с точностью округления."""
    from src.fair_value import implied_strike_twap, twap_probability

    s, sigma, left = 109_500.0, 0.55, 180.0
    alpha, avg = 0.4, 109_480.0
    for p0 in (0.25, 0.5, 0.62, 0.8):
        k = implied_strike_twap(s, p0, left, sigma, alpha=alpha, realized_avg=avg)
        assert k is not None
        p_back = twap_probability(s, float(k), left, sigma,
                                  alpha=alpha, realized_avg=avg)
        assert p_back == pytest.approx(p0, abs=1e-3)


def test_vol_estimator_recovers_sigma_of_smoothed_series():
    """
    Наивная EWMA по секундным тикам SMA-60 занижает sigma в разы: модель
    насыщается в 0/1, fair прилипает к клипу max_model_deviation (вживую:
    fair=0.3132 при mid 0.46 — ровно mid − 0.15). Лаг-выборка 60с с
    поправкой скользящего среднего восстанавливает масштаб.
    """
    import math
    import random

    rng = random.Random(5)
    sigma_true = 0.55
    year = 365 * 24 * 3600
    spot = 109_500.0
    window: list[float] = []

    naive = VolatilityEstimator(45.0, 8.0, 0.30)
    fixed = VolatilityEstimator(600.0, 8.0, 0.30, sample_interval_s=60.0,
                                ma_window_s=60.0, ready_samples=12)
    for t in range(4000):
        spot *= math.exp(sigma_true * math.sqrt(1 / year) * rng.gauss(0, 1))
        window.append(spot)
        if len(window) > 60:
            window.pop(0)
        sma = sum(window) / len(window)
        if t >= 60:
            naive.update(sma, float(t))
            fixed.update(sma, float(t))

    assert naive.sigma_annual < 0.2, "наивная оценка должна занижать в разы"
    assert 0.40 < fixed.sigma_annual < 0.75, fixed.sigma_annual
    assert fixed.ready                      # ~65 сэмплов при пороге 12

    # Симптом вживую: с наивной sigma модель насыщена, с исправленной — нет.
    from src.fair_value import twap_probability

    k = 109_500.0
    saturated = twap_probability(k * 0.999, k, 200.0, naive.sigma_annual,
                                 alpha=1 / 3, realized_avg=k)
    healthy = twap_probability(k * 0.999, k, 200.0, fixed.sigma_annual,
                               alpha=1 / 3, realized_avg=k)
    assert saturated < 0.001
    assert 0.03 < healthy < 0.45


def test_vol_estimator_rejects_interval_below_ma_window():
    with pytest.raises(ValueError):
        VolatilityEstimator(600.0, 8.0, 0.30, sample_interval_s=30.0,
                            ma_window_s=60.0)


# ---------------------------------------------------------------- лестница


class LadderCfg(StratCfg):
    target_pair_cost = D("0.97")
    ladder_levels = 5
    ladder_step_ticks = 2
    ladder_level_size = D("15")


def _fv_at(mid: str, seconds_left: int = 300) -> FairValue:
    m = FairValueModel(D("0.35"), D("0.3"), D("0.15"))
    return m.compute(
        spot=100_000, strike=100_000, seconds_left=seconds_left,
        market_mid=D(mid), sigma_annual=0.5, drift_per_second=0.0, vol_ready=True,
    )


def test_ladder_levels_descend_by_step_and_keep_pair_invariant(market, books):
    """
    Лестница: уровень 0 — цена одиночной котировки, каждый следующий на
    ladder_step_ticks тиков дальше от mid, размер уровня — ladder_level_size.
    ГЛАВНЫЙ ИНВАРИАНТ распространяется на ЛЮБУЮ пару уровней: YES_i + NO_j +
    комиссия < max_pair_cost, потому что глубже — только дешевле.
    """
    q = QuoteGenerator(LadderCfg(), RiskCfg())
    yes_book, no_book = books
    pos = MarketPosition("0xcond")
    for mid in ("0.20", "0.35", "0.50", "0.65", "0.85"):
        quotes = q.build_quotes(market, _fv_at(mid), pos, yes_book, no_book)
        if not quotes:
            continue
        yes = sorted((x for x in quotes if x.outcome == "YES"), key=lambda x: x.level)
        no = sorted((x for x in quotes if x.outcome == "NO"), key=lambda x: x.level)
        assert len(yes) == 5 and len(no) == 5, mid
        for legs in (yes, no):
            assert [x.level for x in legs] == [0, 1, 2, 3, 4]
            for a, b in zip(legs, legs[1:]):
                assert a.price - b.price == D("0.02")   # шаг 2 тика по 0.01
            # Размер уровня — ladder_level_size с обычными поправками
            # (directional x1.25/x0.8), одинаковый на всех уровнях стороны.
            assert len({x.size for x in legs}) == 1
            assert legs[0].size in {D("15.00"), D("18.75"), D("12.00")}
            assert all(x.side == "BUY" for x in legs)
        for y in yes:
            for n in no:
                total = y.price + n.price + market.fee_per_pair(y.price, n.price)
                assert total < LadderCfg.max_pair_cost, (mid, y.level, n.level, total)
        # Ключи дедупликации различают уровни: N уровней = N разных ордеров.
        assert len({x.key() for x in quotes}) == len(quotes)


def test_ladder_of_one_level_is_the_single_quote(market, books):
    """ladder_levels=1 — ровно прежние две котировки, размер order_size."""
    single = QuoteGenerator(StratCfg(), RiskCfg())

    class OneLevel(LadderCfg):
        ladder_levels = 1
        target_pair_cost = StratCfg.target_pair_cost

    ladder = QuoteGenerator(OneLevel(), RiskCfg())
    yes_book, no_book = books
    pos = MarketPosition("0xcond")
    fv = _fv_at("0.50")
    a = single.build_quotes(market, fv, pos, yes_book, no_book)
    b = ladder.build_quotes(market, fv, pos, yes_book, no_book)
    assert [(x.outcome, x.price, x.size, x.level) for x in a] == \
        [(x.outcome, x.price, x.size, x.level) for x in b]
    assert len(a) == 2 and all(x.level == 0 for x in a)


def test_ladder_level_size_zero_means_order_size(market, books):
    class Cfg(LadderCfg):
        ladder_level_size = D("0")

    q = QuoteGenerator(Cfg(), RiskCfg())
    yes_book, no_book = books
    quotes = q.build_quotes(market, _fv_at("0.50"), MarketPosition("0xcond"), yes_book, no_book)
    assert quotes and all(x.size == StratCfg.order_size for x in quotes)


def test_ladder_drops_levels_below_one_tick(market):
    """Уровень, ушедший ниже одного тика, не ставится — глубже цены нет."""
    class Deep(LadderCfg):
        ladder_levels = 8
        ladder_step_ticks = 3

    q = QuoteGenerator(Deep(), RiskCfg())
    # Книга у нижнего края: YES ~0.12 — лестница из 8 уровней по 3¢ уходит
    # ниже нуля после четвёртого уровня.
    yes = Book(
        token_id="tok_yes",
        bids=[BookLevel(D("0.12"), D("100"))],
        asks=[BookLevel(D("0.14"), D("100"))],
        updated_at=time.time(),
    )
    no = Book(
        token_id="tok_no",
        bids=[BookLevel(D("0.86"), D("100"))],
        asks=[BookLevel(D("0.88"), D("100"))],
        updated_at=time.time(),
    )
    quotes = q.build_quotes(market, _fv_at("0.13"), MarketPosition("0xcond"), yes, no)
    yes_levels = [x for x in quotes if x.outcome == "YES"]
    no_levels = [x for x in quotes if x.outcome == "NO"]
    assert 0 < len(yes_levels) < 8
    assert all(x.price >= market.tick_size for x in yes_levels)
    assert len(no_levels) == 8   # у верхней стороны места хватает


def test_ladder_config_cross_validation():
    """Кросс-валидация: лестница обязана помещаться в лимиты риска."""
    from src.config import RiskSettings, Settings, StrategySettings

    class Bundle:
        def __init__(self, strategy, risk) -> None:
            self.strategy, self.risk = strategy, risk

    ok = Bundle(StrategySettings(ladder_levels=5, ladder_level_size=D("15"),
                                 max_concurrent_markets=2),
                RiskSettings(max_open_orders=20))
    Settings._validate_cross(ok)  # type: ignore[arg-type]

    too_deep = Bundle(StrategySettings(ladder_levels=20, ladder_level_size=D("15")),
                      RiskSettings(max_position_per_side=D("250"), max_open_orders=200))
    with pytest.raises(ValueError, match="RISK_MAX_POSITION_PER_SIDE"):
        Settings._validate_cross(too_deep)  # type: ignore[arg-type]

    too_many = Bundle(StrategySettings(ladder_levels=6, max_concurrent_markets=4),
                      RiskSettings(max_open_orders=16, max_position_per_side=D("250")))
    with pytest.raises(ValueError, match="RISK_MAX_OPEN_ORDERS"):
        Settings._validate_cross(too_many)  # type: ignore[arg-type]

    bad = Bundle(StrategySettings(ladder_levels=0), RiskSettings())
    with pytest.raises(ValueError, match="STRAT_LADDER"):
        Settings._validate_cross(bad)  # type: ignore[arg-type]
