"""
Тесты детектора режима: вход/выход с гистерезисом, приоритет VOLATILE,
сторона, которую засыпает поток. Сеть и SDK не нужны, время виртуальное.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal

from src.regime import Regime, RegimeDetector

D = Decimal


def feed_calm_spot(det: RegimeDetector, t0: float = 0.0, seconds: int = 120) -> float:
    """Ровный спот: наполняет вольные EWMA и бары автокорреляции."""
    rng = random.Random(7)
    price = 100_000.0
    for i in range(seconds):
        price *= 1.0 + rng.gauss(0, 1e-4)
        det.on_spot(price, t0 + float(i))
    return t0 + float(seconds)


def one_sided_fills(det: RegimeDetector, ts: float, n: int = 8) -> None:
    """Поток, который засыпает нас NO (аптренд: тейкеры покупают YES)."""
    for i in range(n):
        det.on_fill("NO", "BUY", D("20"), ts + i * 0.5)


def test_vol_signal_is_silent_during_warmup():
    """
    Пока медленная EWMA не прогрета, отношение вол шумит — сигнал обязан
    молчать, а не срабатывать ложно. Дефолтный порог прогрева равен
    полупериоду медленной EWMA в тиках.
    """
    det = RegimeDetector()
    feed_calm_spot(det, seconds=200)   # меньше дефолтных 300 сэмплов
    assert det.state().vol_ratio is None
    assert det.regime == Regime.CALM


def test_fresh_detector_is_calm():
    det = RegimeDetector()
    assert det.regime == Regime.CALM
    assert det.state().crowded_side is None


def test_one_sided_flow_enters_trending_with_crowded_side():
    det = RegimeDetector(min_fills=6)
    ts = feed_calm_spot(det)
    one_sided_fills(det, ts)

    assert det.regime == Regime.TRENDING
    state = det.state()
    assert state.crowded_side == "NO"      # нас засыпает NO
    assert state.imbalance == 1.0


def test_balanced_flow_stays_calm():
    det = RegimeDetector(min_fills=6)
    ts = feed_calm_spot(det)
    for i in range(6):
        det.on_fill("YES", "BUY", D("20"), ts + i)
        det.on_fill("NO", "BUY", D("20"), ts + i + 0.4)
    assert det.regime == Regime.CALM


def test_trending_hysteresis_holds_between_exit_and_enter():
    """Между exit и enter состояние держится; ниже exit — отпускает."""
    det = RegimeDetector(
        min_fills=4, imbalance_enter=0.7, imbalance_soft=0.55,
        imbalance_exit=0.4, min_hold_s=0.0,
    )
    ts = feed_calm_spot(det)
    one_sided_fills(det, ts, n=6)          # imbalance = 1.0 -> TRENDING
    assert det.regime == Regime.TRENDING

    # Противоположные филлы опускают односторонность до ~0.5:
    # выше exit (0.4), ниже enter (0.7) -> состояние ДЕРЖИТСЯ.
    det.on_fill("YES", "BUY", D("40"), ts + 10)
    assert 0.4 < det.state().imbalance < 0.7
    assert det.regime == Regime.TRENDING

    # Ещё встречный поток — односторонность ниже exit -> CALM.
    det.on_fill("YES", "BUY", D("60"), ts + 11)
    assert det.state().imbalance <= 0.4
    assert det.regime == Regime.CALM


def test_stale_fills_fall_out_of_window():
    """Односторонность стареет: без свежих филлов тренд отпускается."""
    det = RegimeDetector(window_s=30.0, min_fills=4)
    ts = feed_calm_spot(det)
    one_sided_fills(det, ts, n=6)
    assert det.regime == Regime.TRENDING

    # Через 40 секунд тиков спота старые филлы выпали из окна.
    for i in range(45):
        det.on_spot(100_000.0 * (1 + 1e-5 * (i % 2)), ts + 10 + i)
    assert det.regime == Regime.CALM


def test_vol_spike_enters_volatile_and_dominates_trending():
    det = RegimeDetector(
        min_fills=4, vol_ratio_enter=1.6, vol_ratio_exit=1.2,
        vol_fast_halflife_s=5.0, vol_slow_halflife_s=60.0, vol_min_samples=90,
    )
    ts = feed_calm_spot(det, seconds=180)
    one_sided_fills(det, ts, n=6)
    assert det.regime == Regime.TRENDING

    # Всплеск: амплитуда тиков на порядок выше фоновой.
    price = 100_000.0
    t = ts + 5
    for i in range(40):
        price *= 1.0 + (2e-3 if i % 2 == 0 else -2e-3)
        det.on_spot(price, t + i)
    assert det.regime == Regime.VOLATILE   # доминирует над трендом

    # Затухание: быстрая вола возвращается к фоновой, выходим по exit.
    rng = random.Random(9)
    for i in range(600):
        price *= 1.0 + rng.gauss(0, 1e-4)
        det.on_spot(price, t + 40 + i)
    assert det.regime != Regime.VOLATILE


def test_autocorr_accelerates_entry_on_soft_imbalance():
    """
    Мягкая односторонность сама по себе не триггерит тренд, но вместе с
    положительной автокорреляцией доходностей — да. Дрейф постоянной силы
    автокорреляции не создаёт, поэтому строим каскадный путь: длинные
    серии шагов в одну сторону.
    """
    det = RegimeDetector(
        min_fills=4, imbalance_enter=0.9, imbalance_soft=0.5,
        autocorr_enter=0.2, autocorr_min_bars=20,
    )
    # Каскады: 10 секунд вверх, 10 вниз — соседние бары почти всегда
    # одного знака, автокорреляция сильно положительная.
    price, ts = 100_000.0, 0.0
    for i in range(120):
        direction = 1.0 if (i // 10) % 2 == 0 else -1.0
        price *= 1.0 + direction * 3e-4
        det.on_spot(price, ts + i)
    assert det.state().autocorr is not None and det.state().autocorr > 0.2

    # Односторонность 0.6: между soft (0.5) и enter (0.9).
    det.on_fill("NO", "BUY", D("80"), ts + 120)
    det.on_fill("NO", "BUY", D("80"), ts + 120.5)
    det.on_fill("YES", "BUY", D("40"), ts + 121)
    det.on_fill("NO", "BUY", D("40"), ts + 121.5)
    state = det.state()
    assert 0.5 <= state.imbalance < 0.9
    assert det.regime == Regime.TRENDING


def test_min_hold_survives_evidence_starvation():
    """
    Реакция котирования душит поток филлов, по которому тренд обнаружен.
    Минимальное удержание не даёт состоянию осциллировать, а память о
    заваленной стороне сохраняет направление реакции, даже когда живой
    знак потока в окне обнулился.
    """
    det = RegimeDetector(window_s=30.0, min_fills=4, min_hold_s=60.0)
    ts = feed_calm_spot(det)
    one_sided_fills(det, ts, n=5)          # вход, нас засыпает NO
    assert det.regime == Regime.TRENDING

    # 40 секунд тишины: окно филлов полностью опустело, но удержание
    # ещё действует — и состояние, и сторона реакции на месте.
    for i in range(41):
        det.on_spot(100_000.0, ts + 3 + i)
    assert det.state().fills_in_window == 0
    assert det.regime == Regime.TRENDING
    assert det.state().crowded_side == "NO"

    # После истечения удержания пустое окно отпускает состояние.
    for i in range(25):
        det.on_spot(100_000.0, ts + 44 + i)
    assert det.regime == Regime.CALM
    assert det.state().crowded_side is None


def test_lone_stale_fill_does_not_pin_trending():
    """
    Удержание TRENDING тоже требует свидетельств: один доживающий в окне
    филл (imbalance == 1.0) не должен пиннить состояние после того, как
    поток кончился.
    """
    det = RegimeDetector(window_s=60.0, min_fills=4, min_hold_s=0.0)
    ts = feed_calm_spot(det)
    # Пять филлов с интервалом 12 с: входим в TRENDING.
    for i in range(5):
        det.on_fill("NO", "BUY", D("20"), ts + i * 12)
    assert det.regime == Regime.TRENDING

    # Ещё 55 секунд тиков: в окне остаётся один-единственный филл.
    for i in range(56):
        det.on_spot(100_000.0, ts + 48 + 1 + i)
    assert det.state().fills_in_window <= 1
    assert det.regime == Regime.CALM


def test_snapshot_exposes_metrics():
    det = RegimeDetector()
    ts = feed_calm_spot(det)
    one_sided_fills(det, ts)
    snap = det.snapshot()
    assert snap["regime"] == "trending"
    assert snap["crowded"] == "NO"
    assert snap["fills"] == 8
    assert 0 <= snap["imbalance"] <= 1


def test_sell_fills_contribute_with_inverted_sign():
    """Продажа YES двигает net вниз — как покупка NO."""
    det = RegimeDetector(min_fills=4)
    ts = feed_calm_spot(det)
    for i in range(6):
        det.on_fill("YES", "SELL", D("20"), ts + i * 0.5)
    assert det.regime == Regime.TRENDING
    assert det.state().crowded_side == "NO"
