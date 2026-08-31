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
    rng = random.Random(seed)
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
        spot *= math.exp(-0.5 * sigma**2 * dt_y + sigma * math.sqrt(dt_y) * rng.gauss(0, 1))
        vol.update(spot, step * tick_dt)

        # Рынок = истинная вероятность + шум.
        p_true = true_probability(spot, strike, left, sigma)
        p_mkt = min(0.97, max(0.03, p_true + rng.gauss(0, market_noise)))

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
            if rng.random() >= prob:
                continue

            pos.apply_fill(
                q.outcome, "BUY", q.price, q.size, market.fee_for(q.price, q.size)
            )
            fills += 1

            # Adverse selection: с вероятностью `toxicity` этот филл был от
            # информированного участника, и цена немедленно уходит против нас.
            if rng.random() < toxicity:
                shock = abs(rng.gauss(0, 1)) * sigma * math.sqrt(1.0 / SECONDS_PER_YEAR)
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
    args = ap.parse_args()

    print(__doc__)
    print(f"Прогонов: {args.runs} | окно {args.window}s | sigma {args.sigma} | "
          f"toxicity {args.toxicity} | queue {args.queue} | "
          f"комиссия {args.fee_rate} | газ merge {args.merge_gas}")
    print("-" * 62)

    results = [
        run_one(args.window, args.sigma, args.noise, args.spread, seed,
                args.toxicity, args.queue, D(args.fee_rate), D(args.merge_gas))
        for seed in range(args.runs)
    ]
    pnls = [r["pnl"] for r in results]
    fills = [r["fills"] for r in results]
    merged = [r["merged"] for r in results]
    fees = [r["fees"] for r in results]
    gas = [r["gas"] for r in results]

    mean = statistics.mean(pnls)
    stdev = statistics.pstdev(pnls) or 1e-9
    wins = sum(1 for p in pnls if p > 0)

    print(f"Средний PnL за окно : {mean:+.3f} USDC")
    print(f"Медиана             : {statistics.median(pnls):+.3f}")
    print(f"Ст. отклонение      : {stdev:.3f}")
    print(f"Sharpe (за окно)    : {mean / stdev:.3f}")
    print(f"Доля прибыльных окон: {wins / len(pnls):.1%}")
    print(f"Худшее окно         : {min(pnls):+.3f}")
    print(f"Лучшее окно         : {max(pnls):+.3f}")
    print(f"Филлов за окно      : {statistics.mean(fills):.1f}")
    print(f"Смержено пар        : {statistics.mean(merged):.1f}")
    print(f"Комиссии за окно    : {statistics.mean(fees):.3f} USDC")
    print(f"Газ merge за окно   : {statistics.mean(gas):.3f} USDC")
    print("-" * 62)
    print("НАПОМИНАНИЕ: это верхняя граница. Прогоните --toxicity 0.5 и 0.65,")
    print("чтобы увидеть, как быстро стратегия уходит в минус. Реальная")
    print("toxicity против розничного бота обычно ВЫШЕ, чем вы думаете.")


if __name__ == "__main__":
    main()
