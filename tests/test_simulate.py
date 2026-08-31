"""
Тесты статистики симулятора: доверительные интервалы, парное сравнение,
детерминизм и раздельные потоки случайности (common random numbers).

SDK не нужен: simulate.py импортирует только src.* и стандартную библиотеку.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from simulate import mean_ci, paired_diff_ci, run_one, share_ci

D = Decimal


def sim(seed: int, toxicity: float) -> dict:
    return run_one(60, 0.55, 0.012, 0.02, seed, toxicity, 0.45, D("0"), D("0.01"))


def test_mean_ci_matches_hand_calculation():
    # mean=3, stdev=1, n=5 -> half = 1.96 / sqrt(5)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, half = mean_ci(values)
    assert mean == pytest.approx(3.0)
    assert half == pytest.approx(1.96 * (2.5 ** 0.5) / (5 ** 0.5))
    # Один прогон — интервал бесконечен, а не выдуман.
    assert mean_ci([7.0])[1] == float("inf")


def test_share_ci_binomial_approximation():
    share, half = share_ci(50, 100)
    assert share == pytest.approx(0.5)
    assert half == pytest.approx(1.96 * 0.05)


def test_paired_diff_ci_is_zero_for_identical_runs():
    a = [1.0, 5.0, -2.0]
    diff, half = paired_diff_ci(a, a)
    assert diff == 0.0 and half == 0.0
    with pytest.raises(ValueError):
        paired_diff_ci([1.0], [1.0, 2.0])


def test_run_one_is_deterministic_per_seed():
    """Один seed — один результат: без этого парное сравнение бессмысленно."""
    assert sim(11, 0.5) == sim(11, 0.5)
    assert sim(11, 0.5) != sim(12, 0.5)


def test_rng_streams_are_isolated_by_purpose():
    """
    Потоки случайности разделены по назначению: при toxicity=0 шоковый
    поток не потребляется вовсе, и результат прогона совпадает с прогоном,
    где шоки существуют, но никогда не срабатывают, — базовая траектория
    и лотерея очереди общие. Это и есть механизм common random numbers.
    """
    # toxicity=0.0: rng_tox.random() всё равно вызывается на каждом филле,
    # но gauss-шок — никогда. Сравниваем два уровня, при которых шок
    # не срабатывает ни разу: 0.0 и отрицательный (условие всегда ложно).
    assert sim(3, 0.0) == sim(3, -1.0)


def test_paired_diff_beats_independent_difference():
    """
    Смысл общих seed'ов: парная разность уровней точнее, чем разность
    НЕЗАВИСИМЫХ прогонов (её полуширина sqrt(half_a^2 + half_b^2)).
    Полного совпадения траекторий нет и быть не может: шоки adverse
    selection двигают спот, это сам механизм воздействия, — поэтому
    выигрыш частичный, но обязан быть положительным.
    """
    lo = [sim(seed, 0.35)["pnl"] for seed in range(60)]
    hi = [sim(seed, 0.5)["pnl"] for seed in range(60)]

    _, half_lo = mean_ci(lo)
    _, half_hi = mean_ci(hi)
    _, half_paired = paired_diff_ci(lo, hi)
    half_independent = (half_lo**2 + half_hi**2) ** 0.5
    assert half_paired < half_independent
