"""
Детектор режима рынка: CALM / TRENDING / VOLATILE с гистерезисом.

Зачем: стратегия симметричного котирования устроена для бокового рынка.
В тренде поток тейкеров становится односторонним — одну нашу сторону
выносит, другая голодает, пары перестают складываться, и мы копим
проигрышный инвентарь (симулятор: в трендовых окнах доля пар падает
вдвое, PnL уходит в глубокий минус). Детектор существует, чтобы заметить
это ДО того, как позиция упрётся в риск-лимиты.

Три сигнала:

1. Односторонность НАШЕГО потока филлов за скользящее окно. Каждый филл
   даёт вклад в net со знаком (+YES/-NO с учётом стороны сделки);
   imbalance = |сумма| / сумма модулей, в [0..1]. Это самое прямое
   свидетельство: в тренде нас исполняют в одну сторону. Отсюда же
   crowded_side — сторона, которую нам ЗАСЫПАЮТ (в аптренде тейкеры
   покупают YES, снимая наши BUY-NO биды => crowded = NO).

2. Реализованная волатильность относительно своего же среднего: быстрая
   EWMA-дисперсия против медленной. Отношение >> 1 — рынок вошёл в
   волатильный всплеск, где наша модель и котировки не успевают.

3. Автокорреляция 1-секундных доходностей спота (lag-1). Положительная —
   движения продолжаются (momentum-каскад). Честная оговорка: тренд с
   ПОСТОЯННЫМ дрейфом автокорреляции не создаёт (приращения независимы),
   поэтому в синтетике симулятора вход в TRENDING происходит по потоку
   филлов; автокорреляция — подтверждающий сигнал для реального рынка,
   где тренды идут каскадами, и она ускоряет вход при мягкой
   односторонности.

Гистерезис: пороги входа строже порогов выхода, чтобы состояние не
дребезжало на границе. Приоритет VOLATILE > TRENDING > CALM.

Модуль не знает ни про SDK, ни про конфиг: все пороги — аргументы
конструктора (движок передаёт их из StrategySettings), время — явный
аргумент методов (тестируемость и симулятор с виртуальным временем).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .models import Outcome, Side

_LN2 = math.log(2.0)


class Regime(str, Enum):
    CALM = "calm"
    TRENDING = "trending"
    VOLATILE = "volatile"


@dataclass(slots=True)
class RegimeState:
    """Снимок состояния для котирования и логов."""

    regime: Regime
    # Сторона, которую нам засыпает поток (валидна в TRENDING, иначе None).
    crowded_side: Outcome | None
    imbalance: float
    fills_in_window: int
    vol_ratio: float | None
    autocorr: float | None


class _EwmaVar:
    """EWMA-дисперсия нормированных доходностей с полупериодом в секундах."""

    __slots__ = ("_lam_base", "value", "samples")

    def __init__(self, halflife_s: float) -> None:
        self._lam_base = math.exp(-_LN2 / max(halflife_s, 1e-6))
        self.value: float | None = None
        self.samples = 0

    def update(self, squared_return: float, dt: float) -> None:
        lam = self._lam_base ** dt
        if self.value is None:
            self.value = squared_return
        else:
            self.value = lam * self.value + (1.0 - lam) * squared_return
        self.samples += 1


def _autocorr_lag1(xs: list[float]) -> float | None:
    """Автокорреляция соседних элементов; None, если данных мало."""
    n = len(xs)
    if n < 3:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs)
    if var <= 0:
        return None
    cov = sum((xs[i] - mean) * (xs[i + 1] - mean) for i in range(n - 1))
    return cov / var


class RegimeDetector:
    """Копит сигналы одного актива и держит текущее состояние режима."""

    def __init__(
        self,
        *,
        window_s: float = 120.0,
        min_fills: int = 6,
        imbalance_enter: float = 0.70,
        imbalance_soft: float = 0.45,
        imbalance_exit: float = 0.40,
        autocorr_enter: float = 0.25,
        autocorr_bars: int = 90,
        autocorr_min_bars: int = 30,
        vol_ratio_enter: float = 1.8,
        vol_ratio_exit: float = 1.35,
        vol_fast_halflife_s: float = 20.0,
        vol_slow_halflife_s: float = 300.0,
        # Сигнал волатильности молчит, пока медленная база не прогрета:
        # EWMA с полупериодом 300 с после пары минут всё ещё взвешена в
        # пользу первых сэмплов, и отношение fast/slow на прогреве шумит
        # до ложных VOLATILE. Дефолт = полупериод медленной EWMA в тиках.
        vol_min_samples: int = 300,
        bar_s: float = 1.0,
    ) -> None:
        if not (imbalance_exit <= imbalance_soft <= imbalance_enter):
            raise ValueError("пороги односторонности: exit <= soft <= enter")
        if vol_ratio_exit >= vol_ratio_enter:
            raise ValueError("порог выхода из VOLATILE должен быть ниже порога входа")

        self.window_s = window_s
        self.min_fills = min_fills
        self.imbalance_enter = imbalance_enter
        self.imbalance_soft = imbalance_soft
        self.imbalance_exit = imbalance_exit
        self.autocorr_enter = autocorr_enter
        self.autocorr_min_bars = autocorr_min_bars
        self.vol_ratio_enter = vol_ratio_enter
        self.vol_ratio_exit = vol_ratio_exit
        self.vol_min_samples = vol_min_samples
        self.bar_s = bar_s

        # (ts, signed_size): знак — вклад филла в net (+ = копим YES).
        self._fills: deque[tuple[float, float]] = deque()
        self._fast = _EwmaVar(vol_fast_halflife_s)
        self._slow = _EwmaVar(vol_slow_halflife_s)
        self._last_price: float | None = None
        self._last_ts: float | None = None
        # 1-секундные бары для автокорреляции.
        self._bar_returns: deque[float] = deque(maxlen=autocorr_bars)
        self._bar_open_price: float | None = None
        self._bar_index: int | None = None

        self._regime = Regime.CALM

    # ------------------------------------------------------------ сигналы

    def on_fill(self, outcome: Outcome, side: Side, size: Decimal, ts: float) -> None:
        """Учесть наш филл. Знак = вклад в net-позицию (+YES / -NO)."""
        signed = float(size)
        if (outcome == "YES") != (side == "BUY"):
            signed = -signed
        self._fills.append((ts, signed))
        self._prune(ts)
        self._reevaluate()

    def on_spot(self, price: float, ts: float) -> None:
        """Учесть тик спота: вола (быстрая/медленная) и бары автокорреляции."""
        if price <= 0:
            return
        if self._last_price is not None and self._last_ts is not None:
            dt = ts - self._last_ts
            if 0 < dt <= 30.0:
                norm = math.log(price / self._last_price) / math.sqrt(dt)
                self._fast.update(norm * norm, dt)
                self._slow.update(norm * norm, dt)
        self._last_price, self._last_ts = price, ts

        bar = int(ts / self.bar_s)
        if self._bar_index is None:
            self._bar_index, self._bar_open_price = bar, price
        elif bar != self._bar_index:
            if self._bar_open_price and self._bar_open_price > 0:
                self._bar_returns.append(math.log(price / self._bar_open_price))
            self._bar_index, self._bar_open_price = bar, price

        self._prune(ts)
        self._reevaluate()

    # ------------------------------------------------------------ метрики

    def _prune(self, now: float) -> None:
        deadline = now - self.window_s
        while self._fills and self._fills[0][0] < deadline:
            self._fills.popleft()

    def _imbalance(self) -> tuple[float, int, Outcome | None]:
        total = sum(abs(s) for _, s in self._fills)
        if total <= 0:
            return 0.0, 0, None
        net = sum(s for _, s in self._fills)
        side: Outcome | None = None
        if net > 0:
            side = "YES"
        elif net < 0:
            side = "NO"
        return abs(net) / total, len(self._fills), side

    def _vol_ratio(self) -> float | None:
        if (
            self._fast.value is None
            or self._slow.value is None
            or self._slow.value <= 0
            or self._slow.samples < self.vol_min_samples
        ):
            return None
        return math.sqrt(self._fast.value / self._slow.value)

    def _autocorr(self) -> float | None:
        if len(self._bar_returns) < self.autocorr_min_bars:
            return None
        return _autocorr_lag1(list(self._bar_returns))

    # ------------------------------------------------------------ автомат

    def _reevaluate(self) -> None:
        vol = self._vol_ratio()
        imbalance, fills_n, _ = self._imbalance()
        autocorr = self._autocorr()

        # VOLATILE доминирует и удерживается собственным exit-порогом.
        if vol is not None:
            if self._regime == Regime.VOLATILE:
                if vol > self.vol_ratio_exit:
                    return
            elif vol >= self.vol_ratio_enter:
                self._regime = Regime.VOLATILE
                return

        trending_now = fills_n >= self.min_fills and (
            imbalance >= self.imbalance_enter
            or (
                imbalance >= self.imbalance_soft
                and autocorr is not None
                and autocorr >= self.autocorr_enter
            )
        )

        if self._regime == Regime.TRENDING:
            # Гистерезис: удерживаемся, пока односторонность выше exit.
            if trending_now or (fills_n > 0 and imbalance > self.imbalance_exit):
                return
            self._regime = Regime.CALM
        elif trending_now:
            self._regime = Regime.TRENDING
        else:
            self._regime = Regime.CALM

    # ------------------------------------------------------------- доступ

    @property
    def regime(self) -> Regime:
        return self._regime

    def state(self) -> RegimeState:
        imbalance, fills_n, side = self._imbalance()
        return RegimeState(
            regime=self._regime,
            crowded_side=side if self._regime == Regime.TRENDING else None,
            imbalance=imbalance,
            fills_in_window=fills_n,
            vol_ratio=self._vol_ratio(),
            autocorr=self._autocorr(),
        )

    def snapshot(self) -> dict:
        """Метрики для status-лога."""
        s = self.state()
        return {
            "regime": s.regime.value,
            "crowded": s.crowded_side,
            "imbalance": round(s.imbalance, 3),
            "fills": s.fills_in_window,
            "vol_ratio": round(s.vol_ratio, 3) if s.vol_ratio is not None else None,
            "autocorr": round(s.autocorr, 3) if s.autocorr is not None else None,
        }
