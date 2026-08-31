"""
Офлайн-тесты стратегии. Сеть и ключи не нужны.

Запуск:  pytest tests/ -v
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from src.fair_value import FairValueModel, VolatilityEstimator, _norm_cdf
from src.models import Book, BookLevel, MarketPosition, TargetMarket
from src.quoting import QuoteGenerator, round_to_tick
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
