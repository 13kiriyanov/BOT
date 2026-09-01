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
    # Готовность вол-сигнала: без неё vol_ratio == None ГЕЙТИТСЯ, а не
    # отсутствует. Ложь = сигнал ещё прогревается и решений не принимает.
    vol_ready: bool = False


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
        # Минимум филлов в окне для входа в TRENDING. Калибр — собственный
        # темп бота: при ~20 филлах за 10-минутное окно на 120-секундное
        # окно приходится 4-6, и порог выше этого делает детектор слепым:
        # вход случался бы к середине тренда или никогда.
        min_fills: int = 4,
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
        # ГОТОВНОСТЬ вол-сигнала — ДВА независимых условия, оба обязательны.
        # 1) Минимум сэмплов: защита от решений на паре тиков.
        # 2) Минимум ПРОШЕДШЕГО времени данных: живой фид даёт 5-20 тиков/с,
        #    и 300 сэмплов набираются за полминуты — а медленная EWMA с
        #    полупериодом 300 с к этому моменту всё ещё прибита к первым
        #    сэмплам, отношение fast/slow систематически завышено (вживую:
        #    2.7-2.8 по обоим активам сразу после старта => ложный VOLATILE,
        #    который глушил котирование полностью). Один счётчик тиков от
        #    этого не защищает — тики не измеряют прогрев EWMA, его измеряет
        #    время. При elapsed >= полупериода медленной EWMA остаточный вес
        #    первого сэмпла <= 50%, и завышение отношения ограничено sqrt(2)
        #    < порога входа 1.8. Пока сигнал не готов — он НЕДОСТУПЕН
        #    (None), и режим по воле не назначается: гейт без данных — не
        #    гейт, а выключатель.
        vol_min_samples: int = 300,
        vol_min_elapsed_s: float | None = None,  # None => полупериод slow EWMA
        bar_s: float = 1.0,
        # Минимальное удержание TRENDING после входа. Реакция котирования
        # снимает/отодвигает заваленную сторону, то есть ДУШИТ сам поток
        # филлов, по которому тренд был обнаружен: без удержания детектор
        # осциллирует «вошёл -> подавил свидетельства -> вышел -> снова
        # завалило». Удержание разрывает эту петлю на время, за которое
        # тренд либо подтвердится новыми филлами, либо нет.
        min_hold_s: float = 45.0,
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
        self.vol_min_elapsed_s = (
            vol_slow_halflife_s if vol_min_elapsed_s is None else vol_min_elapsed_s
        )
        self.bar_s = bar_s
        self.min_hold_s = min_hold_s

        # (ts, signed_size): знак — вклад филла в net (+ = копим YES).
        self._fills: deque[tuple[float, float]] = deque()
        self._fast = _EwmaVar(vol_fast_halflife_s)
        self._slow = _EwmaVar(vol_slow_halflife_s)
        self._last_price: float | None = None
        self._last_ts: float | None = None
        self._first_spot_ts: float | None = None
        # 1-секундные бары для автокорреляции. Значение кэшируется и
        # пересчитывается только при закрытии нового бара: O(N) на каждый
        # тик превратил бы прогон симулятора из секунд в минуты.
        self._bar_returns: deque[float] = deque(maxlen=autocorr_bars)
        self._bar_open_price: float | None = None
        self._bar_index: int | None = None
        self._autocorr_cache: float | None = None
        self._autocorr_dirty = False

        self._regime = Regime.CALM
        self._entered_trending_at: float | None = None
        # Заваленная сторона на момент входа: пока реакция морит поток,
        # живой знак может обнулиться, а сторона реакции меняться не должна.
        self._crowded_hold: Outcome | None = None

    @classmethod
    def from_settings(cls, s) -> "RegimeDetector":  # noqa: ANN001 - StrategySettings
        """Собрать детектор из порогов StrategySettings (утиный доступ)."""
        return cls(
            window_s=s.regime_window_s,
            min_fills=s.regime_min_fills,
            imbalance_enter=s.regime_imbalance_enter,
            imbalance_soft=s.regime_imbalance_soft,
            imbalance_exit=s.regime_imbalance_exit,
            autocorr_enter=s.regime_autocorr_enter,
            vol_ratio_enter=s.regime_vol_ratio_enter,
            vol_ratio_exit=s.regime_vol_ratio_exit,
            vol_min_elapsed_s=s.regime_vol_min_elapsed_s,
            min_hold_s=s.regime_min_hold_s,
        )

    # ------------------------------------------------------------ сигналы

    def on_fill(self, outcome: Outcome, side: Side, size: Decimal, ts: float) -> None:
        """Учесть наш филл. Знак = вклад в net-позицию (+YES / -NO)."""
        signed = float(size)
        if (outcome == "YES") != (side == "BUY"):
            signed = -signed
        self._fills.append((ts, signed))
        self._prune(ts)
        self._reevaluate(ts)

    def on_spot(self, price: float, ts: float) -> None:
        """Учесть тик спота: вола (быстрая/медленная) и бары автокорреляции."""
        if price <= 0:
            return
        if self._first_spot_ts is None:
            self._first_spot_ts = ts
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
                self._autocorr_dirty = True
            self._bar_index, self._bar_open_price = bar, price

        self._prune(ts)
        self._reevaluate(ts)

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

    def _vol_ready(self) -> bool:
        """Оба условия прогрева: достаточно сэмплов И достаточно времени."""
        if self._slow.samples < self.vol_min_samples:
            return False
        if self._first_spot_ts is None or self._last_ts is None:
            return False
        return (self._last_ts - self._first_spot_ts) >= self.vol_min_elapsed_s

    def _vol_ratio_raw(self) -> float | None:
        """Отношение без гейта прогрева — ТОЛЬКО для наблюдаемости в логах."""
        if (
            self._fast.value is None
            or self._slow.value is None
            or self._slow.value <= 0
        ):
            return None
        return math.sqrt(self._fast.value / self._slow.value)

    def _vol_ratio(self) -> float | None:
        if not self._vol_ready():
            return None
        return self._vol_ratio_raw()

    def _autocorr(self) -> float | None:
        if len(self._bar_returns) < self.autocorr_min_bars:
            return None
        if self._autocorr_dirty or self._autocorr_cache is None:
            self._autocorr_cache = _autocorr_lag1(list(self._bar_returns))
            self._autocorr_dirty = False
        return self._autocorr_cache

    # ------------------------------------------------------------ автомат

    def _reevaluate(self, now: float) -> None:
        vol = self._vol_ratio()
        imbalance, fills_n, live_side = self._imbalance()
        autocorr = self._autocorr()

        # VOLATILE доминирует и удерживается собственным exit-порогом.
        if vol is not None:
            if self._regime == Regime.VOLATILE:
                if vol > self.vol_ratio_exit:
                    return
            elif vol >= self.vol_ratio_enter:
                self._regime = Regime.VOLATILE
                self._entered_trending_at = None
                self._crowded_hold = None
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
            if live_side is not None:
                self._crowded_hold = live_side
            # Минимальное удержание: реакция морит поток-свидетельство,
            # без него состояние осциллирует (см. комментарий к min_hold_s).
            entered = self._entered_trending_at
            if entered is not None and now - entered < self.min_hold_s:
                return
            # Гистерезис: удерживаемся, пока односторонность выше exit,
            # НО удержание тоже требует свидетельств. Один устаревающий
            # филл держит imbalance=1.0 и без этой планки пиннил бы
            # TRENDING на минуты после того, как поток кончился.
            hold_fills = max(2, self.min_fills // 2)
            if trending_now or (
                fills_n >= hold_fills and imbalance > self.imbalance_exit
            ):
                return
            self._regime = Regime.CALM
            self._entered_trending_at = None
            self._crowded_hold = None
        elif trending_now:
            self._regime = Regime.TRENDING
            self._entered_trending_at = now
            self._crowded_hold = live_side
        else:
            self._regime = Regime.CALM

    # ------------------------------------------------------------- доступ

    @property
    def regime(self) -> Regime:
        return self._regime

    def state(self) -> RegimeState:
        imbalance, fills_n, side = self._imbalance()
        crowded: Outcome | None = None
        if self._regime == Regime.TRENDING:
            crowded = side if side is not None else self._crowded_hold
        return RegimeState(
            regime=self._regime,
            crowded_side=crowded,
            imbalance=imbalance,
            fills_in_window=fills_n,
            vol_ratio=self._vol_ratio(),
            autocorr=self._autocorr(),
            vol_ready=self._vol_ready(),
        )

    def snapshot(self) -> dict:
        """Метрики для status-лога."""
        s = self.state()
        raw = self._vol_ratio_raw()
        return {
            "regime": s.regime.value,
            "crowded": s.crowded_side,
            "imbalance": round(s.imbalance, 3),
            "fills": s.fills_in_window,
            "vol_ratio": round(s.vol_ratio, 3) if s.vol_ratio is not None else None,
            # Сырое отношение видно и ДО готовности: по нему в логе видно,
            # что гейт прогрева реально сдержал (vol_ready=False, raw > 1.8).
            "vol_ratio_raw": round(raw, 3) if raw is not None else None,
            "vol_ready": s.vol_ready,
            "autocorr": round(s.autocorr, 3) if s.autocorr is not None else None,
        }
