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

from src.fair_value import (
    FairValueModel,
    RealizedTwap,
    VolatilityEstimator,
    twap_probability,
)
from src.models import Book, BookLevel, MarketPosition, TargetMarket
from src.quoting import QuoteGenerator
from src.regime import Regime, RegimeDetector

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
    regime_trending_response = True
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
    # В симуляторе fair всегда рядом с рынком, диагностика суммы пары не
    # нужна: 0 отключает WARNING, чтобы не шуметь на тысячах окон.
    pair_sum_warn_gap = D("0")


class SimRisk:
    max_position_per_side = D("250")
    max_net_exposure = D("120")


# Во сколько раз трендовый поток учащает исполнение стороны, которую он
# выносит, и морит противоположную. Величины условны, как и вся модель
# исполнения; важен знак эффекта: пары в тренде складываться перестают.
TREND_FLOW_CROWD = 2.5
TREND_FLOW_STARVE = 0.3

# Паритет с фильтром краёв движка (quote_mid_min/max): рынки с mid вне
# диапазона бот не котирует — почти решённый исход.
QUOTE_MID_MIN = 0.10
QUOTE_MID_MAX = 0.90

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
            fee_rate: Decimal = D("0"), merge_gas: Decimal = D("0.01"),
            trend_prob: float = 0.0, trend_strength: float = 0.0,
            use_regime: bool = False, use_unwind: bool = True,
            max_net: float | None = None,
            resolution: str = "twap") -> dict:
    """
    Одно торговое окно.

    resolution: 'twap' (дефолт, как живые рынки: исход по среднему цены за
    окно, рынок и модель ценят среднее) или 'endpoint' (историческая
    модель конечной точки — для сравнительных прогонов).
    """
    # Отдельный поток случайности на каждое НАЗНАЧЕНИЕ. Прогоны с одним
    # seed и разной toxicity получают одинаковые базовую траекторию спота,
    # рыночный шум и лотерею очереди — различия между уровнями создаёт
    # только сама toxicity. Это common random numbers: парная разность
    # уровней получается точнее, чем разность независимых прогонов.
    # Оговорка: шоки adverse selection двигают спот — это сам механизм
    # воздействия, — поэтому после первого филла траектории расходятся.
    # Выравнивание частичное (корреляция уровней ~0.3-0.6), и выигрыш
    # тем меньше, чем дальше уровни друг от друга.
    rng_path = random.Random(seed * 8 + 0)     # GBM-приращения спота
    rng_market = random.Random(seed * 8 + 1)   # шум цены рынка
    rng_queue = random.Random(seed * 8 + 2)    # лотерея очереди исполнения
    rng_tox = random.Random(seed * 8 + 3)      # триггер и величина шока
    rng_trend = random.Random(seed * 8 + 4)    # назначение и направление тренда
    quoter = QuoteGenerator(SimStrat(), SimRisk())
    fvm = FairValueModel(D("0.35"), D("0.30"), D("0.15"))
    vol = VolatilityEstimator(45.0, 8.0, 0.30)
    # Детектор режима работает на виртуальном времени симуляции (секунды
    # шагов). use_regime=False — прогон «до части 3»: реакция выключена.
    detector = RegimeDetector.from_settings(SimStrat) if use_regime else None
    regime_seconds = {Regime.TRENDING: 0, Regime.VOLATILE: 0}

    spot = strike = 100_000.0
    pos = MarketPosition("sim")
    tick_dt = 1.0
    # Реализованная часть TWAP окна — тем же накопителем, что у движка:
    # резолюция и модель считают среднее одним и тем же интегратором.
    twap_acc = RealizedTwap(0.0, float(window_s)) if resolution == "twap" else None

    # --- Трендовый режим -------------------------------------------------
    # В доле trend_prob окон спот получает устойчивый снос вместо чистого
    # GBM: годовой дрейф подобран так, чтобы за окно ожидаемый лог-ход
    # составил trend_strength сигм самого окна. Назначение и направление
    # тренда сидят в СВОЁМ потоке случайности: группировка окон на
    # трендовые/спокойные одинакова при любых toxicity и включённой или
    # выключенной реакции — парное сравнение по группам корректно.
    window_years = max(window_s, 1) / SECONDS_PER_YEAR
    trending = rng_trend.random() < trend_prob
    trend_dir = 1.0 if rng_trend.random() < 0.5 else -1.0
    mu_annual = (
        trend_dir * trend_strength * sigma / math.sqrt(window_years)
        if trending else 0.0
    )

    # Тренд — это не только снос цены, но и ОДНОСТОРОННИЙ поток тейкеров.
    # В аптренде агрессоры покупают YES, то есть снимают YES-аски — а это
    # наши BUY-NO биды: их выносит, мы копим проигрышную сторону. Наши
    # YES-биды при этом голодают: продавцов мало. Без этой асимметрии
    # исполнения тренд в симуляции не мешал бы складывать пары.
    crowded_outcome = None
    if trending:
        crowded_outcome = "NO" if trend_dir > 0 else "YES"

    market = TargetMarket(
        condition_id="sim", slug="sim", question="sim",
        yes_token_id="t_yes", no_token_id="t_no",
        end_ts=0, tick_size=D("0.01"), min_order_size=D("5"),
        neg_risk=False, asset="BTC", strike=D(str(strike)),
        fees_enabled=fee_rate > 0, fee_rate=fee_rate,
        fee_exponent=D("1") if fee_rate > 0 else D("0"),
    )

    # Лимиты как в связке конфигов реального бота: кросс-валидация требует
    # directional_max_net <= max_net_exposure, поэтому при свипе лимита
    # порог разгрузки едет вместе с ним.
    net_cap = float(max_net) if max_net is not None else float(SimRisk.max_net_exposure)
    unwind_limit_base = min(float(SimStrat.directional_max_net), net_cap)

    fills = 0
    unwound = 0.0
    bought = {"YES": 0.0, "NO": 0.0}
    for step in range(window_s):
        left = window_s - step
        market.end_ts = 9e18  # seconds_left переопределим вручную ниже
        object.__setattr__(market, "end_ts", __import__("time").time() + left)

        now = float(step)
        # Эволюция спота (GBM).
        dt_y = tick_dt / SECONDS_PER_YEAR
        spot *= math.exp(
            (mu_annual - 0.5 * sigma**2) * dt_y
            + sigma * math.sqrt(dt_y) * rng_path.gauss(0, 1)
        )
        vol.update(spot, step * tick_dt)
        if detector is not None:
            detector.on_spot(spot, now)
            if detector.regime in regime_seconds:
                regime_seconds[detector.regime] += 1

        # Рынок = истинная вероятность + шум. В TWAP-режиме «истина» — это
        # вероятность для СРЕДНЕГО за окно с учётом уже реализованной части:
        # эффективный рынок ценит ту же величину, что и резолюция.
        twap_alpha: float | None = None
        twap_avg: float | None = None
        if twap_acc is not None:
            twap_acc.update(spot, now)
            state = twap_acc.state(now)
            assert state is not None  # начало окна покрыто по построению
            twap_alpha, twap_avg = state
            p_true = twap_probability(
                spot, strike, left, sigma, alpha=twap_alpha, realized_avg=twap_avg
            )
        else:
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
            twap_alpha=twap_alpha,
            twap_realized=twap_avg,
        )
        # --- Разгрузка перекоса: точная копия _unwind_if_needed движка ---
        # Если |net| превысил directional-лимит (у экспирации лимит ужат до
        # order_size), разгрузочный SELL ЗАМЕНЯЕТ обычные котировки.
        quotes = None
        if use_unwind:
            unwind_limit = unwind_limit_base
            if left < 60:
                unwind_limit = min(unwind_limit, float(SimStrat.order_size))
            if abs(float(pos.net)) > unwind_limit:
                book = yes_book if pos.net > 0 else no_book
                quotes = quoter.build_unwind_quotes(market, pos, book) or None

        if quotes is None:
            if not (QUOTE_MID_MIN <= p_mkt <= QUOTE_MID_MAX):
                # Край диапазона: движок такие рынки не котирует.
                quotes = []
            else:
                regime_state = detector.state() if detector is not None else None
                quotes = quoter.build_quotes(
                    market, fv, pos, yes_book, no_book, regime_state
                )

        # --- Модель исполнения ------------------------------------------
        # Три эффекта, без которых симуляция врёт в вашу пользу:
        #  1) ЛИМИТЫ ПОЗИЦИИ — иначе бот копит бесконечный инвентарь;
        #  2) ОЧЕРЕДЬ — перед вами стоят другие мейкеры, вы исполняетесь
        #     далеко не на каждом касании цены;
        #  3) ADVERSE SELECTION — часть потока информированная: вас
        #     исполняют ровно перед движением против вас.
        for q in quotes:
            side_size = pos.yes_size if q.outcome == "YES" else pos.no_size

            if q.side == "SELL":
                # Разгрузочный аск на тик выше бида. Покупателю он выгоден,
                # когда ниже ценности; в тренде продать заваленную сторону
                # труднее всего — её никто не покупает (голодание x0.3),
                # и это главная причина, почему разгрузка не бесплатна.
                ref = p_mkt if q.outcome == "YES" else 1 - p_mkt
                through = ref - float(q.price)
                if through <= 0:
                    continue
                prob = min(0.12, through * 2.5) * queue_factor
                if crowded_outcome is not None:
                    if q.outcome == crowded_outcome:
                        prob *= TREND_FLOW_STARVE
                    else:
                        prob *= TREND_FLOW_CROWD
                if rng_queue.random() >= prob:
                    continue
                sell_size = min(q.size, side_size)
                if sell_size <= 0:
                    continue
                pos.apply_fill(
                    q.outcome, "SELL", q.price, sell_size,
                    market.fee_for(q.price, sell_size),
                )
                unwound += float(sell_size)
                fills += 1
                if detector is not None:
                    detector.on_fill(q.outcome, "SELL", sell_size, now)
                # Adverse selection зеркален: продали YES информированному
                # покупателю — спот идёт вверх, против нас.
                if rng_tox.random() < toxicity:
                    shock = abs(rng_tox.gauss(0, 1)) * sigma * math.sqrt(1.0 / SECONDS_PER_YEAR)
                    spot *= math.exp(shock * 8 if q.outcome == "YES" else -shock * 8)
                continue

            if side_size + q.size > float(SimRisk.max_position_per_side):
                continue
            if abs(pos.net) >= net_cap:
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
            if crowded_outcome is not None:
                if q.outcome == crowded_outcome:
                    prob *= TREND_FLOW_CROWD     # эту сторону выносит поток
                else:
                    prob *= TREND_FLOW_STARVE    # эта сторона голодает
            if rng_queue.random() >= prob:
                continue

            pos.apply_fill(
                q.outcome, "BUY", q.price, q.size, market.fee_for(q.price, q.size)
            )
            bought[q.outcome] += float(q.size)
            fills += 1
            if detector is not None:
                detector.on_fill(q.outcome, "BUY", q.size, now)

            # Adverse selection: с вероятностью `toxicity` этот филл был от
            # информированного участника, и цена немедленно уходит против нас.
            if rng_tox.random() < toxicity:
                shock = abs(rng_tox.gauss(0, 1)) * sigma * math.sqrt(1.0 / SECONDS_PER_YEAR)
                # Купили YES -> спот идёт вниз, и наоборот.
                spot *= math.exp(-shock * 8 if q.outcome == "YES" else shock * 8)

        # Периодический merge.
        if step % 20 == 0 and pos.complete_pairs >= 25:
            pos.apply_merge(pos.complete_pairs, merge_gas)

    # Резолюция остатка. Живые рынки: TWAP за окно >= страйка (правила
    # говорят «greater than or equal»). endpoint — историческое сравнение
    # конечной цены, оставлено для сравнительных прогонов.
    if twap_acc is not None:
        final_state = twap_acc.state(float(window_s))
        assert final_state is not None
        won_yes = final_state[1] >= strike
    else:
        won_yes = spot > strike
    settle = pos.yes_size if won_yes else pos.no_size
    cost = pos.total_cost
    pairs_done = float(pos.merged_pairs + pos.complete_pairs)
    pos.realized_pnl += settle - cost

    total_bought = bought["YES"] + bought["NO"]
    return {
        "pnl": float(pos.realized_pnl),
        "fills": fills,
        "merged": float(pos.merged_pairs),
        "residual_net": float(pos.net),
        "abs_residual": abs(float(pos.net)),
        # Доля купленных shares, закончивших в полной паре (обе ноги пары).
        "pair_rate": (2.0 * pairs_done / total_bought) if total_bought > 0 else 0.0,
        "trending": trending,
        "unwound": unwound,
        # Сколько секунд окна детектор провёл в каждом состоянии — чтобы
        # видеть, что реакция вообще включалась, а не победила вхолостую.
        "regime_trending_s": regime_seconds[Regime.TRENDING],
        "regime_volatile_s": regime_seconds[Regime.VOLATILE],
        "fees": float(pos.fees_paid),
        "gas": float(pos.merge_costs),
    }


def run_level(
    args,
    toxicity: float,
    use_regime: bool | None = None,
    use_unwind: bool | None = None,
    max_net: float | None = None,
) -> list[dict]:
    """Прогнать один уровень toxicity на seed'ах 0..runs-1."""
    if use_regime is None:
        use_regime = args.regime
    if use_unwind is None:
        use_unwind = not args.no_unwind
    if max_net is None:
        max_net = args.max_net
    tasks = [
        (args.window, args.sigma, args.noise, args.spread, seed,
         toxicity, args.queue, Decimal(args.fee_rate), Decimal(args.merge_gas),
         args.trend_prob, args.trend_strength, use_regime, use_unwind, max_net,
         args.resolution)
        for seed in range(args.runs)
    ]
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            return pool.starmap(run_one, tasks, chunksize=25)
    return [run_one(*t) for t in tasks]


def group_stats(results: list[dict]) -> str:
    """Однострочная сводка группы окон: PnL, доля пар, непарный остаток."""
    if not results:
        return "нет окон"
    pnls = [r["pnl"] for r in results]
    mean, half = mean_ci(pnls)
    pair_rate = statistics.mean(r["pair_rate"] for r in results)
    residual = statistics.mean(r["abs_residual"] for r in results)
    unwound = statistics.mean(r.get("unwound", 0.0) for r in results)
    return (f"PnL {mean:+7.2f} ± {half:.2f} | пар {pair_rate:5.1%} | "
            f"|остаток| {residual:5.1f} | разгружено {unwound:5.1f} | "
            f"окон {len(results)}")


def print_trend_split(results: list[dict]) -> None:
    """Разрез по трендовым и спокойным окнам."""
    trend = [r for r in results if r["trending"]]
    calm = [r for r in results if not r["trending"]]
    if not trend:
        return
    print(f"  спокойные : {group_stats(calm)}")
    print(f"  трендовые : {group_stats(trend)}")


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
    print(f"Доля shares в парах : {statistics.mean([r['pair_rate'] for r in results]):.1%}")
    print(f"Средний |остаток|   : {statistics.mean([r['abs_residual'] for r in results]):.1f} shares")
    print(f"Комиссии за окно    : {statistics.mean(fees):.3f} USDC")
    print(f"Газ merge за окно   : {statistics.mean(gas):.3f} USDC")
    print_trend_split(results)


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
        print_trend_split(results)

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


def print_paired_by_group(base: list[dict], variant: list[dict]) -> dict:
    """Парные разности вариант-база по группам окон. Возвращает Δ по группам."""
    diffs: dict[str, tuple[float, float]] = {}
    for name, selector in (("спокойные", False), ("трендовые", True), ("все", None)):
        idx = [
            i for i, r in enumerate(base)
            if selector is None or r["trending"] == selector
        ]
        if not idx:
            continue
        off = [base[i]["pnl"] for i in idx]
        on = [variant[i]["pnl"] for i in idx]
        mean_off, _ = mean_ci(off)
        mean_on, _ = mean_ci(on)
        diff, half = paired_diff_ci(off, on)
        verdict = "неразличимо" if abs(diff) <= half else (
            "лучше" if diff > 0 else "ХУЖЕ"
        )
        print(f"{name:>9} ({len(idx):4d} окон): база {mean_off:+7.2f}, "
              f"вариант {mean_on:+7.2f}, Δ {diff:+6.2f} ± {half:.2f} ({verdict})")
        diffs[name] = (diff, half)
    return diffs


def run_unwind_compare(args) -> None:
    """
    Разгрузка ВЫКЛ (прежний симулятор) против ВКЛ (паритет с движком) на
    общих seed'ах. Отвечает на вопрос, насколько прежняя оценка урона
    тренда была завышена артефактом модели.
    """
    print(f"Сравнение разгрузки перекоса: {args.runs} прогонов, toxicity "
          f"{args.toxicity}, тренд {args.trend_prob}/{args.trend_strength}σ")
    print("-" * 62)
    base = run_level(args, args.toxicity, use_unwind=False)
    variant = run_level(args, args.toxicity, use_unwind=True)

    for name, results in (("без разгрузки", base), ("с разгрузкой", variant)):
        trend = [r for r in results if r["trending"]]
        calm = [r for r in results if not r["trending"]]
        print(f"{name}:")
        print(f"  спокойные : {group_stats(calm)}")
        print(f"  трендовые : {group_stats(trend)}")
    print("-" * 62)
    print("Парные разности (с разгрузкой − без):")
    print_paired_by_group(base, variant)


def run_net_sweep(args, caps: list[float]) -> None:
    """
    Свип RISK_MAX_NET_EXPOSURE на общих seed'ах: цена защиты в трендовых
    окнах, выраженная в обороте и PnL спокойных. Порог разгрузки едет
    вместе с лимитом: min(directional_max_net, лимит) — как требует
    кросс-валидация конфига реального бота.
    """
    print(f"Свип RISK_MAX_NET_EXPOSURE {caps}: {args.runs} прогонов, toxicity "
          f"{args.toxicity}, тренд {args.trend_prob}/{args.trend_strength}σ, "
          f"общие seed'ы")
    print("-" * 78)

    per_cap: dict[float, list[dict]] = {}
    for cap in caps:
        per_cap[cap] = run_level(args, args.toxicity, max_net=cap)

    print(f"{'лимит':>6} | {'все окна':^18} | {'спокойные':^18} | "
          f"{'трендовые':^18} | {'смерж.':>6} | {'разгр.':>6}")
    for cap in caps:
        results = per_cap[cap]
        calm = [r["pnl"] for r in results if not r["trending"]]
        trend = [r["pnl"] for r in results if r["trending"]]
        alls = [r["pnl"] for r in results]
        m_a, h_a = mean_ci(alls)
        m_c, h_c = mean_ci(calm)
        m_t, h_t = mean_ci(trend)
        merged = statistics.mean(r["merged"] for r in results if not r["trending"])
        unwound = statistics.mean(r["unwound"] for r in results)
        print(f"{cap:6.0f} | {m_a:+7.2f} ± {h_a:5.2f} | {m_c:+7.2f} ± {h_c:5.2f} | "
              f"{m_t:+7.2f} ± {h_t:5.2f} | {merged:6.1f} | {unwound:6.1f}")

    baseline = 120.0 if 120.0 in per_cap else caps[len(caps) // 2]
    print("-" * 78)
    print(f"Парные разности против лимита {baseline:.0f} (Δ = вариант − база):")
    for cap in caps:
        if cap == baseline:
            continue
        print(f"лимит {cap:.0f}:")
        print_paired_by_group(per_cap[baseline], per_cap[cap])
    print("-" * 78)

    means = {cap: mean_ci([r["pnl"] for r in per_cap[cap]])[0] for cap in caps}
    best = max(means, key=means.get)
    ordered = [means[c] for c in caps]
    monotone_up = all(a <= b for a, b in zip(ordered, ordered[1:]))
    monotone_down = all(a >= b for a, b in zip(ordered, ordered[1:]))
    if monotone_up or monotone_down:
        print(f"Суммарный PnL монотонен по лимиту ({'растёт' if monotone_up else 'падает'}); "
              f"внутреннего оптимума в этой сетке нет.")
    else:
        print(f"Максимум суммарного PnL в сетке — лимит {best:.0f} "
              f"({means[best]:+.2f}); значимость смотри по парным разностям выше.")


def run_regime_compare(args) -> None:
    """
    Одни и те же seed'ы, реакция на режим ВЫКЛ против ВКЛ. PnL отдельно по
    трендовым и спокойным окнам (группировка от seed, у обоих вариантов
    одинаковая), разности парные. Это и есть ответ на вопрос «стоит ли
    включать часть 3»: если в спокойных окнах реакция теряет больше, чем
    выигрывает в трендовых, итог отрицательный — и он будет напечатан
    так же честно, как положительный.
    """
    print(f"Сравнение реакции на режим: {args.runs} прогонов, toxicity "
          f"{args.toxicity}, тренд {args.trend_prob}/{args.trend_strength}σ")
    print("-" * 62)
    base = run_level(args, args.toxicity, use_regime=False)
    with_regime = run_level(args, args.toxicity, use_regime=True)

    act_t = statistics.mean(
        r["regime_trending_s"] for r in with_regime if r["trending"]
    ) if any(r["trending"] for r in with_regime) else 0.0
    act_c = statistics.mean(
        r["regime_trending_s"] for r in with_regime if not r["trending"]
    )
    print(f"Детектор в TRENDING: {act_t:.0f} с/окно в трендовых, "
          f"{act_c:.0f} с/окно в спокойных (ложные срабатывания)")
    print("-" * 62)

    total_diff = None
    group_diffs: dict[str, tuple[float, float]] = {}
    for name, selector in (("спокойные", False), ("трендовые", True), ("все", None)):
        idx = [
            i for i, r in enumerate(base)
            if selector is None or r["trending"] == selector
        ]
        if not idx:
            continue
        off = [base[i]["pnl"] for i in idx]
        on = [with_regime[i]["pnl"] for i in idx]
        mean_off, half_off = mean_ci(off)
        mean_on, _ = mean_ci(on)
        diff, half = paired_diff_ci(off, on)
        verdict = "неразличимо" if abs(diff) <= half else (
            "лучше" if diff > 0 else "ХУЖЕ"
        )
        print(f"{name:>9} ({len(idx):4d} окон): без реакции {mean_off:+7.2f}, "
              f"с реакцией {mean_on:+7.2f}, Δ {diff:+6.2f} ± {half:.2f} ({verdict})")
        group_diffs[name] = (diff, half)
        if selector is None:
            total_diff = (diff, half)

    print("-" * 62)
    calm = group_diffs.get("спокойные")
    trend = group_diffs.get("трендовые")
    if calm and trend and total_diff:
        if total_diff[0] > total_diff[1]:
            print("ИТОГ: реакция улучшает суммарный PnL статистически значимо.")
        elif total_diff[0] < -total_diff[1]:
            print("ИТОГ: реакция УХУДШАЕТ суммарный PnL — включать не стоит.")
        else:
            print("ИТОГ: суммарный эффект статистически неразличим.")
        if calm[0] < -calm[1]:
            print("В спокойных окнах реакция значимо ТЕРЯЕТ деньги — смотри,")
            print("покрывает ли это выигрыш в трендовых (строки выше).")


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
    ap.add_argument("--trend-prob", type=float, default=0.25,
                    help="доля окон с направленным дрейфом (0 = чистый GBM)")
    ap.add_argument("--trend-strength", type=float, default=1.5,
                    help="сила дрейфа: ожидаемый лог-ход за окно в сигмах окна")
    ap.add_argument("--regime", action="store_true",
                    help="включить реакцию котирования на режим (часть 3)")
    ap.add_argument("--regime-compare", action="store_true",
                    help="прогнать реакцию ВЫКЛ против ВКЛ на общих seed'ах "
                         "с парным сравнением по трендовым/спокойным окнам")
    ap.add_argument("--no-unwind", action="store_true",
                    help="выключить разгрузку перекоса (прежнее поведение "
                         "симулятора, БЕЗ паритета с движком)")
    ap.add_argument("--max-net", type=float, default=120.0,
                    help="RISK_MAX_NET_EXPOSURE; порог разгрузки едет вместе "
                         "с ним: min(directional_max_net, лимит)")
    ap.add_argument("--unwind-compare", action="store_true",
                    help="разгрузка ВЫКЛ против ВКЛ на общих seed'ах: "
                         "насколько прежняя оценка урона тренда была завышена")
    ap.add_argument("--net-sweep", type=str, default=None,
                    help="свип RISK_MAX_NET_EXPOSURE через запятую "
                         "(например 30,60,90,120,180) на общих seed'ах")
    ap.add_argument("--sweep", type=str, default=None,
                    help="уровни toxicity через запятую (например 0.2,0.35,0.5,0.65): "
                         "прогнать все на общих seed'ах и сравнить парно")
    ap.add_argument("--resolution", choices=["twap", "endpoint"], default="twap",
                    help="резолюция окна: twap — как живые рынки (дефолт); "
                         "endpoint — историческая модель конечной точки")
    ap.add_argument("--jobs", type=int, default=1,
                    help="процессов для параллельного прогона")
    args = ap.parse_args()

    print(__doc__)

    if args.regime_compare:
        run_regime_compare(args)
        return

    if args.unwind_compare:
        run_unwind_compare(args)
        return

    if args.net_sweep:
        caps = [float(x) for x in args.net_sweep.split(",") if x.strip()]
        if len(caps) < 2:
            raise SystemExit("--net-sweep требует минимум два значения")
        run_net_sweep(args, caps)
        return

    if args.sweep:
        levels = [float(x) for x in args.sweep.split(",") if x.strip()]
        if len(levels) < 2:
            raise SystemExit("--sweep требует минимум два уровня")
        run_sweep(args, levels)
        return

    print(f"Прогонов: {args.runs} | окно {args.window}s | sigma {args.sigma} | "
          f"toxicity {args.toxicity} | queue {args.queue} | "
          f"комиссия {args.fee_rate} | газ merge {args.merge_gas} | "
          f"тренд {args.trend_prob}/{args.trend_strength}σ")
    print("-" * 62)
    print_level_report(run_level(args, args.toxicity))
    print("-" * 62)
    print("НАПОМИНАНИЕ: это верхняя граница. Прогоните --sweep 0.2,0.35,0.5,0.65,")
    print("чтобы увидеть чувствительность к adverse selection с парным")
    print("сравнением уровней. Реальная toxicity против розничного бота")
    print("обычно ВЫШЕ, чем вы думаете.")


if __name__ == "__main__":
    main()
