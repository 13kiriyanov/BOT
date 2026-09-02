"""
Тесты замера окна опережения: знак задержки, порог, тайм-аут, сводка.

Сеть и SDK не нужны: тики и стаканы подаются напрямую.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.leadlag import LeadLagTracker

D = Decimal


class Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, event: str, **fields) -> None:
        assert event == "lead_lag"
        self.events.append(fields)


def make_tracker(threshold=0.0005, lookback=5.0, timeout=10.0):
    sink = Sink()
    tracker = LeadLagTracker(threshold, lookback, timeout, sink=sink)
    tracker.register_market("c5", "BTC", 30, "btc-updown-5m-1", "tok_yes")
    return tracker, sink


def test_feed_leads_book_positive_delay():
    """Фид двинулся в t0, стакан ответил через 700 мс — задержка +700."""
    tracker, sink = make_tracker()
    tracker.on_book("tok_yes", D("0.49"), D("0.51"), 100.0)
    tracker.on_spot_tick("BTC", 100_000.0, 100.0, 30)
    tracker.on_spot_tick("BTC", 100_060.0, 101.0, 30)     # +6 bp > порога
    assert sink.events == []                               # ждём стакан
    tracker.on_book("tok_yes", D("0.49"), D("0.51"), 101.3)  # mid не сдвинулся
    tracker.on_book("tok_yes", D("0.50"), D("0.52"), 101.7)  # mid вверх — ответ
    assert len(sink.events) == 1
    ev = sink.events[0]
    assert ev["asset"] == "BTC" and ev["market"] == "btc-updown-5m-1"
    assert ev["direction"] == 1 and ev["window_s"] == 30
    assert ev["move_bp"] == pytest.approx(6.0)
    assert ev["delay_ms"] == pytest.approx(700.0) and ev["timeout"] is False


def test_book_leads_feed_negative_delay():
    """Стакан сдвинулся вниз за 400 мс ДО события фида — задержка −400."""
    tracker, sink = make_tracker()
    tracker.on_spot_tick("BTC", 100_000.0, 100.0, 30)
    tracker.on_book("tok_yes", D("0.49"), D("0.51"), 100.0)
    tracker.on_book("tok_yes", D("0.48"), D("0.50"), 100.6)   # стакан вниз
    tracker.on_spot_tick("BTC", 99_930.0, 101.0, 30)          # фид вниз, −7 bp
    assert len(sink.events) == 1
    assert sink.events[0]["delay_ms"] == pytest.approx(-400.0)
    assert sink.events[0]["direction"] == -1
    # Тот же сдвиг стакана второму событию фида не засчитывается.
    tracker.on_spot_tick("BTC", 99_860.0, 101.5, 30)
    assert len(sink.events) == 1


def test_opposite_book_move_does_not_resolve_and_timeout_is_counted():
    """Сдвиг стакана против направления — не ответ; без ответа — тайм-аут."""
    tracker, sink = make_tracker(timeout=2.0)
    tracker.on_book("tok_yes", D("0.49"), D("0.51"), 100.0)
    tracker.on_spot_tick("BTC", 100_000.0, 100.0, 30)
    tracker.on_spot_tick("BTC", 100_100.0, 101.0, 30)         # вверх
    tracker.on_book("tok_yes", D("0.48"), D("0.50"), 101.5)   # стакан ВНИЗ
    assert sink.events == []
    tracker.on_book("tok_yes", D("0.48"), D("0.50"), 104.0)   # дедлайн прошёл
    assert len(sink.events) == 1
    assert sink.events[0]["delay_ms"] is None and sink.events[0]["timeout"] is True
    assert tracker.summary()["BTC"]["n_timeout"] == 1
    assert tracker.summary()["BTC"]["n"] == 0


def test_threshold_accumulates_small_ticks_and_ignores_other_windows():
    """Ход копится от опорной цены; тики чужого окна рынок не трогают."""
    tracker, sink = make_tracker(threshold=0.001)
    tracker.on_book("tok_yes", D("0.49"), D("0.51"), 100.0)
    tracker.on_spot_tick("BTC", 100_000.0, 100.0, 30)
    for i in range(1, 5):                       # по 3 bp — порог 10 bp
        tracker.on_spot_tick("BTC", 100_000.0 * (1 + 0.0003 * i), 100.0 + i, 30)
    tracker.on_book("tok_yes", D("0.50"), D("0.52"), 105.0)
    assert len(sink.events) == 1                # событие на 4-м тике (12 bp)
    assert sink.events[0]["move_bp"] == pytest.approx(12.0)
    assert sink.events[0]["delay_ms"] == pytest.approx(1000.0)

    # 60-секундный ряд для 5-минутного рынка (окно 30) — чужой.
    tracker.on_spot_tick("BTC", 100_000.0, 200.0, 60)
    tracker.on_spot_tick("BTC", 101_000.0, 201.0, 60)
    tracker.on_book("tok_yes", D("0.51"), D("0.53"), 202.0)
    assert len(sink.events) == 1


def test_summary_median_and_p10_per_asset():
    tracker, sink = make_tracker()
    tracker.register_market("e15", "ETH", 60, "eth-updown-15m-1", "tok_eth")
    tracker.on_book("tok_yes", D("0.49"), D("0.51"), 0.0)
    price = 100_000.0
    for i in range(12):
        t0 = 10.0 * i
        tracker.on_spot_tick("BTC", price, t0, 30)
        price *= 1.001
        tracker.on_spot_tick("BTC", price, t0 + 1.0, 30)
        # Ответ стакана через (100 + 50*i) мс.
        tracker.on_book("tok_yes", D("0.50") + D(i) / 1000, D("0.52") + D(i) / 1000,
                        t0 + 1.0 + (0.1 + 0.05 * i))
    summary = tracker.summary()
    assert summary["BTC"]["n"] == 12
    assert summary["BTC"]["median_ms"] == pytest.approx(375.0, abs=1.0)
    assert summary["BTC"]["p10_ms"] <= summary["BTC"]["median_ms"]
    assert "ETH" not in summary                 # событий не было
    lines = tracker.summary_lines()
    assert len(lines) == 1 and lines[0].startswith("LEAD-LAG BTC | ")
    assert "медиана=+375" in lines[0] and "порог 5.0 bp" in lines[0]
    assert LeadLagTracker(0.0005, 5.0, 10.0, sink=sink).summary_lines() == []


def test_forget_market_stops_tracking():
    tracker, sink = make_tracker()
    tracker.forget_market("c5")
    tracker.on_spot_tick("BTC", 100_000.0, 100.0, 30)
    tracker.on_spot_tick("BTC", 101_000.0, 101.0, 30)
    tracker.on_book("tok_yes", D("0.50"), D("0.52"), 102.0)
    assert sink.events == []


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        LeadLagTracker(0.0, 5.0, 10.0)
    with pytest.raises(ValueError):
        LeadLagTracker(0.001, 5.0, 0.0)
