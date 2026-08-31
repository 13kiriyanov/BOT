#!/usr/bin/env python3
"""
Офлайн-симулятор стратегии. Ни сети, ни ключей, ни денег.

У Polymarket НЕТ тестнета и paper-trading режима. Это единственный способ
прогнать логику котирования до того, как рисковать реальными средствами.

Что симулируется:
  * траектория спота как GBM с заданной волатильностью;
  * рынок, чья цена = «истинная» вероятность + шум + спред;
  * модель исполнения: наш пассивный ордер исполняется, когда рынок
    доходит до его цены, с вероятностью, зависящей от глубины прохода.

Что моделируется грубо (параметры --toxicity и --queue):
  * очередь в стакане: перед вами стоят другие мейкеры, вы исполняетесь
    не на каждом касании цены;
  * adverse selection: часть потока информированная, вас исполняют ровно
    перед движением против вас. ЭТО ГЛАВНЫЙ ФАКТОР УБЫТКОВ.

Комиссии рынка (--fee-rate) и газ merge (--merge-gas) моделируются: первые
списываются с каждого филла и капитализируются в себестоимость пары, второй
списывается с каждой транзакции merge.

Чего симулятор НЕ учитывает вовсе:
  * задержки сети, отказы API, частичные исполнения, реджекты;
  * то, что реальная toxicity против розничного бота почти наверняка
    выше 0.5 — вы медленнее всех остальных участников.

Вывод симулятора — ВЕРХНЯЯ ГРАНИЦА, а не прогноз. Используйте его для
поиска багов в логике и для проверки чувствительности к toxicity,
а НЕ для оценки будущей прибыли.

Запуск:  python simulate.py --runs 200
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import random
import statistics
from decimal import Decimal
from statistics import NormalDist

from src.fair_value import FairValueModel, VolatilityEstimator
from src.models import Book, BookLevel, MarketPosition, TargetMarket
from src.quoting import QuoteGenerator

D = Decimal
ND = NormalDist()
SECONDS_PER_YEAR = 365 * 24 * 3600


class SimStrat:
    target_pair_cost = D("0.985")
    max_pair_cost = D("0.995")
    min_half_spread_ticks = 1
    order_size = D("20")
    inventory_skew_coef = D("0.010")
    allow_directional = True
    directional_min_edge = D("0.025")
    directional_max_net = D("60")


class SimRisk:
    max_position_per_side = D("250")
    max_net_exposure = D("120")


Z95 = 1.96


def mean_ci(values: list[float], z: float = Z95) -> tuple[float, float]:
    """Среднее и полуширина его доверительного интервала (CLT)."""
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, float("inf")
    half = z * statistics.stdev(values) / math.sqrt(n)
    return mean, half


def share_ci(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Доля и полуширина её интервала (нормальное приближение биномиального)."""
    if n == 0:
        return 0.0, float("inf")
    p = hits / n
    return p, z * math.sqrt(p * (1.0 - p) / n)


def paired_diff_ci(
    baseline: list[float], other: list[float], z: float = Z95
) -> tuple[float, float]:
    """
    Разность средних по ПАРНЫМ прогонам (общие seed'ы) и её интервал.

    Дисперсия разности на общих сценариях намного ниже дисперсии самих
    уровней — поэтому соседние уровни toxicity сравниваются именно так,
    а не по перекрытию их индивидуальных интервалов.
    """
    if len(baseline) != len(other):
        raise ValueError("парное сравнение требует одинакового числа прогонов")
    diffs = [b - a for a, b in zip(baseline, other)]
    return mean_ci(diffs, z)


def true_probability(spot: float, strike: float, secs: float, sigma: float) -> float:
    tau = max(secs, 1.0) / SECONDS_PER_YEAR
    vt = sigma * math.sqrt(tau)
    if vt < 1e-9:
        return 1.0 if spot > strike else 0.0
    d = (math.log(spot / strike) - 0.5 * sigma**2 * tau) / vt
    return ND.cdf(d)


def make_book(mid: float, spread: float, tick: float = 0.01) -> Book:
    bid = max(tick, math.floor((mid - spread / 2) / tick) * tick)
    ask = min(1 - tick, math.ceil((mid + spread / 2) / tick) * tick)
    if ask <= bid:
        ask = bid + tick
    return Book(
        token_id="t",
        bids=[BookLevel(D(str(round(bid, 2))), D("300")),
              BookLevel(D(str(round(bid - tick, 2))), D("500"))],
        asks=[BookLevel(D(str(round(ask, 2))), D("300")),
              BookLevel(D(str(round(ask + tick, 2))), D("500"))],
        updated_at=9e18,  # никогда не протухает
    )


def run_one(window_s: int, sigma: float, market_noise: float,
            market_spread: float, seed: int,
            toxicity: float = 0.35, queue_factor: float = 0.45,
            fee_rate: Decimal = D("0"), merge_gas: Decimal = D("0.01")) -> dict:
    """Одно торговое окно."""
    # Отдельный поток случайности на каждое НАЗНАЧЕНИЕ. Прогоны с одним
    # seed и разной toxicity получают одинаковые базовую траекторию спота,
    # рыночный шум и лотерею очереди — различия между уровнями создаёт
    # только сама toxicity. Это common random numbers: парная разность
    # уровней получается точнее, чем разность независимых прогонов.
    # Оговорка: шоки adverse selection двигают спот — это сам механизм
    # воздействия, — поэтому после первого филла траектории расходятся.
    # Выравнивание частичное (корреляция уровней ~0.3-0.6), и выигрыш
    # тем меньше, чем дальше уровни друг от друга.
    rng_path = random.Random(seed * 4 + 0)     # GBM-приращения спота
    rng_market = random.Random(seed * 4 + 1)   # шум цены рынка
    rng_queue = random.Random(seed * 4 + 2)    # лотерея очереди исполнения
    rng_tox = random.Random(seed * 4 + 3)      # триггер и величина шока
    quoter = QuoteGenerator(SimStrat(), SimRisk())
    fvm = FairValueModel(D("0.35"), D("0.30"), D("0.15"))
    vol = VolatilityEstimator(45.0, 8.0, 0.30)

    spot = strike = 100_000.0
    pos = MarketPosition("sim")
    tick_dt = 1.0

    market = TargetMarket(
        condition_id="sim", slug="sim", question="sim",
        yes_token_id="t_yes", no_token_id="t_no",
        end_ts=0, tick_size=D("0.01"), min_order_size=D("5"),
        neg_risk=False, asset="BTC", strike=D(str(strike)),
        fees_enabled=fee_rate > 0, fee_rate=fee_rate,
        fee_exponent=D("1") if fee_rate > 0 else D("0"),
    )

    fills = 0
    for step in range(window_s):
        left = window_s - step
        market.end_ts = 9e18  # seconds_left переопределим вручную ниже
        object.__setattr__(market, "end_ts", __import__("time").time() + left)

        # Эволюция спота (GBM).
        dt_y = tick_dt / SECONDS_PER_YEAR
        spot *= math.exp(-0.5 * sigma**2 * dt_y + sigma * math.sqrt(dt_y) * rng_path.gauss(0, 1))
        vol.update(spot, step * tick_dt)

        # Рынок = истинная вероятность + шум.
        p_true = true_probability(spot, strike, left, sigma)
        p_mkt = min(0.97, max(0.03, p_true + rng_market.gauss(0, market_noise)))

        yes_book = make_book(p_mkt, market_spread)
        no_book = make_book(1 - p_mkt, market_spread)

        fv = fvm.compute(
            spot=spot, strike=strike, seconds_left=left,
            market_mid=D(str(round(p_mkt, 4))),
            sigma_annual=vol.sigma_annual,
            drift_per_second=vol.drift_per_second,
            vol_ready=vol.ready,
        )
        quotes = quoter.build_quotes(market, fv, pos, yes_book, no_book)

        # --- Модель исполнения ------------------------------------------
        # Три эффекта, без которых симуляция врёт в вашу пользу:
        #  1) ЛИМИТЫ ПОЗИЦИИ — иначе бот копит бесконечный инвентарь;
        #  2) ОЧЕРЕДЬ — перед вами стоят другие мейкеры, вы исполняетесь
        #     далеко не на каждом касании цены;
        #  3) ADVERSE SELECTION — часть потока информированная: вас
        #     исполняют ровно перед движением против вас.
        for q in quotes:
            side_size = pos.yes_size if q.outcome == "YES" else pos.no_size
            if side_size + q.size > float(SimRisk.max_position_per_side):
                continue
            if abs(pos.net) >= float(SimRisk.max_net_exposure):
                # Разрешаем только сокращающую сторону.
                reduces = (q.outcome == "NO" and pos.net > 0) or (
                    q.outcome == "YES" and pos.net < 0
                )
                if not reduces:
                    continue

            ref = p_mkt if q.outcome == "YES" else 1 - p_mkt
            through = ref - float(q.price)
            if through <= 0:
                continue
            # Очередь: базовая вероятность низкая, растёт с глубиной прохода.
            prob = min(0.12, through * 2.5) * queue_factor
            if rng_queue.random() >= prob:
                continue

            pos.apply_fill(
                q.outcome, "BUY", q.price, q.size, market.fee_for(q.price, q.size)
            )
            fills += 1

            # Adverse selection: с вероятностью `toxicity` этот филл был от
            # информированного участника, и цена немедленно уходит против нас.
            if rng_tox.random() < toxicity:
                shock = abs(rng_tox.gauss(0, 1)) * sigma * math.sqrt(1.0 / SECONDS_PER_YEAR)
                # Купили YES -> спот идёт вниз, и наоборот.
                spot *= math.exp(-shock * 8 if q.outcome == "YES" else shock * 8)

        # Периодический merge.
        if step % 20 == 0 and pos.complete_pairs >= 25:
            pos.apply_merge(pos.complete_pairs, merge_gas)

    # Резолюция остатка.
    won_yes = spot > strike
    settle = pos.yes_size if won_yes else pos.no_size
    cost = pos.total_cost
    pos.realized_pnl += settle - cost

    return {
        "pnl": float(pos.realized_pnl),
        "fills": fills,
        "merged": float(pos.merged_pairs),
        "residual_net": float(pos.net),
        "fees": float(pos.fees_paid),
        "gas": float(pos.merge_costs),
    }


def run_level(args, toxicity: float) -> list[dict]:
    """Прогнать один уровень toxicity на seed'ах 0..runs-1."""
    tasks = [
        (args.window, args.sigma, args.noise, args.spread, seed,
         toxicity, args.queue, Decimal(args.fee_rate), Decimal(args.merge_gas))
        for seed in range(args.runs)
    ]
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            return pool.starmap(run_one, tasks, chunksize=25)
    return [run_one(*t) for t in tasks]


def print_level_report(results: list[dict]) -> None:
    pnls = [r["pnl"] for r in results]
    fills = [r["fills"] for r in results]
    merged = [r["merged"] for r in results]
    fees = [r["fees"] for r in results]
    gas = [r["gas"] for r in results]

    mean, half = mean_ci(pnls)
    stdev = statistics.pstdev(pnls) or 1e-9
    win_share, win_half = share_ci(sum(1 for p in pnls if p > 0), len(pnls))

    print(f"Средний PnL за окно : {mean:+.3f} ± {half:.3f} USDC (95% CI)")
    print(f"Медиана             : {statistics.median(pnls):+.3f}")
    print(f"Ст. отклонение      : {stdev:.3f}")
    print(f"Sharpe (за окно)    : {mean / stdev:.3f}")
    print(f"Доля прибыльных окон: {win_share:.1%} ± {win_half:.1%}")
    print(f"Худшее окно         : {min(pnls):+.3f}")
    print(f"Лучшее окно         : {max(pnls):+.3f}")
    print(f"Филлов за окно      : {statistics.mean(fills):.1f}")
    print(f"Смержено пар        : {statistics.mean(merged):.1f}")
    print(f"Комиссии за окно    : {statistics.mean(fees):.3f} USDC")
    print(f"Газ merge за окно   : {statistics.mean(gas):.3f} USDC")


def run_sweep(args, levels: list[float]) -> None:
    """
    Прогнать несколько уровней toxicity на ОБЩИХ seed'ах и сравнить их
    парно. Именно параметр --sweep порождает таблицу для README.
    """
    print(f"Sweep по toxicity {levels}: {args.runs} прогонов на уровень, "
          f"общие seed'ы, jobs={args.jobs}")
    print("-" * 62)

    per_level: dict[float, list[float]] = {}
    for level in levels:
        results = run_level(args, level)
        per_level[level] = [r["pnl"] for r in results]
        mean, half = mean_ci(per_level[level])
        wins, win_half = share_ci(
            sum(1 for p in per_level[level] if p > 0), len(per_level[level])
        )
        print(f"toxicity {level:4.2f} | PnL {mean:+7.2f} ± {half:.2f} | "
              f"прибыльных {wins:.1%} ± {win_half:.1%}")

    print("-" * 62)
    print("Парные разности соседних уровней (общие seed'ы):")
    for lo, hi in zip(levels, levels[1:]):
        diff, half = paired_diff_ci(per_level[lo], per_level[hi])
        verdict = (
            "СТАТИСТИЧЕСКИ НЕРАЗЛИЧИМО" if abs(diff) <= half else "различимо"
        )
        print(f"  {lo:.2f} -> {hi:.2f}: ΔPnL {diff:+.2f} ± {half:.2f}  ({verdict})")
    print("-" * 62)
    print("Неразличимая разность означает: на этом числе прогонов эффект")
    print("уровня не отделяется от шума. Это свойство данных, а не ошибка.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--window", type=int, default=600, help="длина окна, сек")
    ap.add_argument("--sigma", type=float, default=0.55, help="годовая вол BTC")
    ap.add_argument("--noise", type=float, default=0.012, help="шум цены рынка")
    ap.add_argument("--spread", type=float, default=0.02, help="рыночный спред")
    ap.add_argument("--toxicity", type=float, default=0.35,
                    help="доля информированного потока (adverse selection)")
    ap.add_argument("--queue", type=float, default=0.45,
                    help="доля касаний цены, где вы дошли до головы очереди")
    ap.add_argument("--fee-rate", type=str, default="0",
                    help="ставка комиссии рынка (0 = рынок без комиссий)")
    ap.add_argument("--merge-gas", type=str, default="0.01",
                    help="стоимость одной транзакции merge, USDC")
    ap.add_argument("--sweep", type=str, default=None,
                    help="уровни toxicity через запятую (например 0.2,0.35,0.5,0.65): "
                         "прогнать все на общих seed'ах и сравнить парно")
    ap.add_argument("--jobs", type=int, default=1,
                    help="процессов для параллельного прогона")
    args = ap.parse_args()

    print(__doc__)

    if args.sweep:
        levels = [float(x) for x in args.sweep.split(",") if x.strip()]
        if len(levels) < 2:
            raise SystemExit("--sweep требует минимум два уровня")
        run_sweep(args, levels)
        return

    print(f"Прогонов: {args.runs} | окно {args.window}s | sigma {args.sigma} | "
          f"toxicity {args.toxicity} | queue {args.queue} | "
          f"комиссия {args.fee_rate} | газ merge {args.merge_gas}")
    print("-" * 62)
    print_level_report(run_level(args, args.toxicity))
    print("-" * 62)
    print("НАПОМИНАНИЕ: это верхняя граница. Прогоните --sweep 0.2,0.35,0.5,0.65,")
    print("чтобы увидеть чувствительность к adverse selection с парным")
    print("сравнением уровней. Реальная toxicity против розничного бота")
    print("обычно ВЫШЕ, чем вы думаете.")


if __name__ == "__main__":
    main()
