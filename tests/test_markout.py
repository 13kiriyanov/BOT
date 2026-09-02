"""
Тесты mark-out: знак, разбиение на корзины, FIFO-матчинг пар, устойчивость.

Сеть, ключи и SDK не нужны: mid подаётся заглушкой, горизонты — миллисекунды.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from src.markout import BUCKET_PAIRED, BUCKET_SOLO, BUCKET_UNWIND, MarkoutTracker
from src.models import Fill

D = Decimal

HORIZONS = (0.01, 0.02, 0.03)


def make_fill(**overrides) -> Fill:
    fields = dict(
        trade_id="t-1",
        condition_id="0xcond",
        token_id="tok_yes",
        side="BUY",
        price=D("0.49"),
        size=D("20"),
    )
    fields.update(overrides)
    return Fill(**fields)  # type: ignore[arg-type]


class Sink:
    """Собирает события markout вместо trades.jsonl."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, event: str, **fields) -> None:
        assert event == "markout"
        self.events.append(fields)


def run_tracker(mid_source, fills, horizons=HORIZONS, **kwargs):
    """Прогнать филлы через трекер и дождаться всех замеров."""
    sink = Sink()
    tracker = MarkoutTracker(mid_source, horizons_s=horizons, sink=sink, **kwargs)

    async def drive() -> None:
        for fill, outcome, complement in fills:
            tracker.record_fill(fill, outcome, complement)
        while tracker.pending:
            await asyncio.sleep(0.005)

    asyncio.run(drive())
    return tracker, sink


def test_buy_markout_sign_and_values():
    """Покупка: рынок вверх — плюс, рынок вниз — минус, ровно (mid-p)*size."""
    mids = iter([D("0.50"), D("0.47"), D("0.49")])

    def mid_source(token: str, complement: str) -> Decimal:
        return next(mids)

    _, sink = run_tracker(mid_source, [(make_fill(), "YES", "tok_no")])

    assert len(sink.events) == 1
    ev = sink.events[0]
    assert ev["trade_id"] == "t-1"
    assert ev["token"] == "tok_yes"
    assert ev["outcome"] == "YES"
    assert ev["sign"] == 1
    assert ev["price"] == D("0.49")
    assert ev["size"] == D("20")
    # (0.50 - 0.49) * 20 = +0.20; (0.47 - 0.49) * 20 = -0.40; ноль на 0.49.
    assert ev["mid_0.01s"] == D("0.50") and ev["markout_0.01s"] == D("0.20")
    assert ev["mid_0.02s"] == D("0.47") and ev["markout_0.02s"] == D("-0.40")
    assert ev["markout_0.03s"] == D("0.00")


def test_sell_markout_sign_is_inverted_and_bucketed_as_unwind():
    """Продажа: рынок упал после нас — это ХОРОШО, mark-out положителен."""
    def mid_source(token: str, complement: str) -> Decimal:
        return D("0.45")

    tracker, sink = run_tracker(
        mid_source, [(make_fill(side="SELL", price=D("0.50"), size=D("10")), "YES", "tok_no")]
    )

    ev = sink.events[0]
    assert ev["sign"] == -1
    assert ev["bucket"] == BUCKET_UNWIND
    # -1 * (0.45 - 0.50) * 10 = +0.50 на каждом горизонте.
    assert ev["markout_0.01s"] == D("0.50")

    summary = tracker.summary()["bucket"]
    assert summary["0.01s"][BUCKET_UNWIND]["n"] == 1
    assert summary["0.01s"][BUCKET_UNWIND]["per_share"] == pytest.approx(0.05)


def test_fifo_pairing_splits_fills_into_paired_and_solo():
    """
    BUY YES 20, затем BUY NO 15: NO-нога целиком в паре, YES-нога — 15 в
    паре и 5 solo. Смешивать эти корзины нельзя — у них разная экономика.
    """
    def mid_source(token: str, complement: str) -> Decimal:
        return D("0.50")

    fills = [
        (make_fill(trade_id="t-yes", token_id="tok_yes", size=D("20")), "YES", "tok_no"),
        (
            make_fill(trade_id="t-no", token_id="tok_no", price=D("0.48"), size=D("15")),
            "NO",
            "tok_yes",
        ),
    ]
    tracker, sink = run_tracker(mid_source, fills)

    by_id = {ev["trade_id"]: ev for ev in sink.events}
    assert by_id["t-yes"]["paired_size"] == D("15")
    assert by_id["t-yes"]["solo_size"] == D("5")
    assert by_id["t-no"]["paired_size"] == D("15")
    assert by_id["t-no"]["solo_size"] == D("0")

    bucket = tracker.summary()["bucket"]
    # В паре 15 shares YES + 15 shares NO, solo — 5 shares YES.
    assert bucket["0.01s"][BUCKET_PAIRED]["size"] == pytest.approx(30.0)
    assert bucket["0.01s"][BUCKET_SOLO]["size"] == pytest.approx(5.0)

    outcome = tracker.summary()["outcome"]
    assert outcome["0.01s"]["YES"]["size"] == pytest.approx(20.0)
    assert outcome["0.01s"]["NO"]["size"] == pytest.approx(15.0)


def test_sell_consumes_unpaired_inventory():
    """
    Проданный односторонний инвентарь парой уже не станет: покупка другой
    стороны после продажи матчится только с тем, что реально осталось.
    """
    def mid_source(token: str, complement: str) -> Decimal:
        return D("0.50")

    fills = [
        (make_fill(trade_id="t-yes", size=D("20")), "YES", "tok_no"),
        (make_fill(trade_id="t-sell", side="SELL", size=D("20")), "YES", "tok_no"),
        (
            make_fill(trade_id="t-no", token_id="tok_no", price=D("0.48"), size=D("20")),
            "NO",
            "tok_yes",
        ),
    ]
    _, sink = run_tracker(mid_source, fills)

    by_id = {ev["trade_id"]: ev for ev in sink.events}
    assert by_id["t-no"]["paired_size"] == D("0")     # матчиться уже не с чем
    assert by_id["t-no"]["solo_size"] == D("20")


def test_missing_mid_is_counted_not_faked():
    """
    Рынок истёк или книги нет: mid недоступен. Точка не выдумывается —
    в событии null, в сводке отдельный счётчик n_miss.
    """
    mids = iter([D("0.50"), None, None])

    def mid_source(token: str, complement: str):
        return next(mids)

    tracker, sink = run_tracker(mid_source, [(make_fill(), "YES", "tok_no")])

    ev = sink.events[0]
    assert ev["markout_0.01s"] == D("0.20")
    assert ev["mid_0.02s"] is None and ev["markout_0.02s"] is None

    bucket = tracker.summary()["bucket"]
    assert bucket["0.02s"][BUCKET_SOLO]["n_miss"] == 1
    assert bucket["0.02s"][BUCKET_SOLO]["n"] == 0


def test_mid_source_exception_does_not_kill_measurement():
    """Метрика не имеет права ронять движок: исключение = пропуск точки."""
    calls = {"n": 0}

    def mid_source(token: str, complement: str) -> Decimal:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("книга исчезла")
        return D("0.50")

    tracker, sink = run_tracker(mid_source, [(make_fill(), "YES", "tok_no")])
    ev = sink.events[0]
    assert ev["markout_0.02s"] is None
    assert ev["markout_0.03s"] is not None


def test_pending_measurements_are_bounded():
    """Переполнение очереди замеров теряет точки, а не память."""
    def mid_source(token: str, complement: str) -> Decimal:
        return D("0.50")

    tracker, sink = run_tracker(
        mid_source,
        [
            (make_fill(trade_id=f"t-{i}"), "YES", "tok_no")
            for i in range(5)
        ],
        max_pending=2,
    )
    assert len(sink.events) == 2
    assert tracker._dropped == 3


def test_summary_lines_render_both_views():
    """status_loop получает строки и по корзинам, и по сторонам."""
    def mid_source(token: str, complement: str) -> Decimal:
        return D("0.51")

    tracker, _ = run_tracker(mid_source, [(make_fill(), "YES", "tok_no")])
    lines = tracker.summary_lines()
    assert len(lines) == 3
    assert lines[0].startswith("MARKOUT | ")
    assert "solo=" in lines[0] and "paired=" in lines[0]
    assert lines[1].startswith("MARKOUT по стороне | ")
    assert "YES=" in lines[1] and "NO=" in lines[1]
    # Третья строка — разрез по горизонту рынка; без слага горизонт "?".
    assert lines[2].startswith("MARKOUT по горизонту рынка | ")
    assert "?/solo=" in lines[2]
    # Пустой трекер не шумит в статус.
    assert MarkoutTracker(mid_source, horizons_s=HORIZONS).summary_lines() == []


def test_market_horizon_label_from_slug():
    from src.markout import horizon_label_from_slug

    assert horizon_label_from_slug("btc-updown-5m-1788276000") == "5m"
    assert horizon_label_from_slug("eth-updown-15m-1788276000") == "15m"
    assert horizon_label_from_slug("btc-updown-4h-1788276000") == "4h"
    assert horizon_label_from_slug("bitcoin-up-or-down-test") == "?"
    assert horizon_label_from_slug("") == "?"


def test_summary_splits_by_market_horizon():
    """
    Разрез по горизонту рынка: филлы 5-минутных и 15-минутных рынков
    копятся в разные ячейки с той же экономикой корзин (paired/solo), чтобы
    на живых филлах было видно, где mark-out лучше.
    """
    mids = {"tok_yes": D("0.50"), "tok_no": D("0.45")}

    def mid_source(token, _complement):
        return mids[token]

    fills = [
        # 5m: одиночная покупка YES по 0.49 -> solo, mark-out +0.01/share.
        (make_fill(trade_id="a", condition_id="c5", token_id="tok_yes", price=D("0.49")),
         "YES", "tok_no"),
        # 15m: покупка NO по 0.48 -> solo, mark-out -0.03/share.
        (make_fill(trade_id="b", condition_id="c15", token_id="tok_no", price=D("0.48")),
         "NO", "tok_yes"),
    ]
    sink = Sink()
    tracker = MarkoutTracker(mid_source, horizons_s=(0.01,), sink=sink)

    async def drive() -> None:
        tracker.record_fill(*fills[0], market_horizon="5m")
        tracker.record_fill(*fills[1], market_horizon="15m")
        while tracker.pending:
            await asyncio.sleep(0.005)

    asyncio.run(drive())

    by_h = tracker.summary()["market_horizon"]["0.01s"]
    assert by_h["5m/solo"]["per_share"] == pytest.approx(0.01)
    assert by_h["15m/solo"]["per_share"] == pytest.approx(-0.03)
    assert by_h["5m/solo"]["n"] == 1 and by_h["15m/solo"]["n"] == 1
    assert {e["market_horizon"] for e in sink.events} == {"5m", "15m"}
    lines = tracker.summary_lines()
    assert any(line.startswith("MARKOUT по горизонту рынка") and "5m/solo=" in line
               and "15m/solo=" in line for line in lines)
    # Без разреза (горизонт неизвестен) строка всё равно есть — ячейка "?/...".
    tracker2, _ = run_tracker(mid_source, [fills[0]], horizons=(0.01,))
    assert "?/solo" in tracker2.summary()["market_horizon"]["0.01s"]
