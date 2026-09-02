"""
Модель справедливой цены YES для рынка «BTC Up or Down (TWAP)».

Математика
----------
ПРАВИЛА РЫНКА (подтверждены по описанию живых рынков): резолвится Up, если
TWAP Chainlink за окно [T0, T1] >= значения того же ряда в момент T0.
То есть рынок спрашивает про СРЕДНЕЕ ПО ВРЕМЕНИ, а не про конечную точку.

Пусть t — сейчас, W = T1 - T0 — длина окна, e = t - T0 — прошло,
tau = T1 - t — осталось, alpha = e / W, beta = 1 - alpha = tau / W.

    TWAP = alpha * A_real + beta * A_rem

где A_real — УЖЕ РЕАЛИЗОВАННОЕ среднее (накапливаем сами по фиду, класс
RealizedTwap), а A_rem = (1/tau) * ∫_t^{T1} S_u du — среднее остатка.
Реализованная часть — константа, случаен только остаток:

    P(TWAP >= K) = P( A_rem >= K_eff ),   K_eff = (K - alpha * A_real) / beta

Для GBM ln S_u = ln S_t + (m - sigma^2/2)(u - t) + sigma (W_u - W_t).
Среднее броуновского моста: (1/tau) ∫_0^tau W_u du ~ N(0, tau/3) —
классический результат, дисперсия среднего втрое меньше дисперсии конца.
Дрейф входит половиной: среднее линейно растущего сноса = половина сноса
конца. Отсюда лог-нормальное приближение остатка:

    ln A_rem ~ N( ln S_t + (m - sigma^2/2) * tau/2,  sigma^2 * tau/3 )

    P(A_rem >= K_eff) = Phi( d ),
    d = ( ln(S_t / K_eff) + (m - sigma^2/2) * tau_y/2 ) / ( sigma * sqrt(tau_y/3) )

Члены порядка sigma^2*tau (поправка Йенсена лог-среднего и т.п.) отброшены
сознательно: на нашем горизонте sigma^2*tau ~ 1e-6 — на порядки меньше
тика вероятности. Крайние случаи: K_eff <= 0 означает, что даже нулевые
будущие цены не опустят TWAP ниже K, — вероятность 1 независимо от спота.

Следствия против модели конечной точки:
- дисперсия остатка sigma^2*tau/3 меньше — модель УВЕРЕННЕЕ при том же
  отклонении;
- к концу окна beta -> 0 и K_eff уходит от K по мере накопления A_real:
  реализованная часть «запирает» исход, уверенность растёт быстрее, чем
  давала бы точка конца;
- одиночный шип цены почти не двигает среднее — рынок «медленнее» спота.

Практические поправки, без которых модель бесполезна на 5-15 минутах:

1. sigma оценивается EWMA по лог-доходностям ФИДА РЕЗОЛЮЦИИ (Chainlink
   TWAP-поток): моделируем тот ряд, по которому рынок рассчитается.
2. m берётся из momentum (EWMA доходностей), но с малым коэффициентом.
   На минутном горизонте momentum крипты слабый и шумный; ставить на него
   всерьёз — путь к сливу. Он нужен, чтобы не быть на неправильной стороне.
3. tau -> 0 делает модель сингулярной. Мы явно снижаем доверие к модели по
   мере приближения к экспирации И ограничиваем отклонение от рынка.
4. Итог смешивается с рыночным microprice. Рынок агрегирует информацию,
   которой у нас нет. Полностью игнорировать его — самоуверенность.
"""

from __future__ import annotations

import math
import time
from decimal import Decimal
from statistics import NormalDist

from .models import FairValue

SECONDS_PER_YEAR = 365.0 * 24 * 3600
_ND = NormalDist()


def _norm_cdf(x: float) -> float:
    """Стандартная нормальная CDF через erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class RealizedTwap:
    """
    Реализованная часть TWAP окна [start_ts, end_ts]: ступенчатый интеграл
    цены по времени (значение фида держится до следующего тика — ровно так
    устроен и сам поток TWAP).

    Честность покрытия: если первый тик пришёл ПОЗЖЕ start_ts больше чем на
    start_tolerance_s, начало окна не наблюдалось, и реализованное среднее
    невосстановимо — state() возвращает None, модель по рынку не работает.
    Проверка проваливается в сторону остановки, а не «донаблюдаем как есть».
    """

    __slots__ = (
        "start_ts", "end_ts", "start_tolerance_s",
        "_integral", "_last_price", "_last_ts", "_first_ts",
    )

    def __init__(
        self, start_ts: float, end_ts: float, start_tolerance_s: float = 3.0
    ) -> None:
        if end_ts <= start_ts:
            raise ValueError("окно TWAP должно иметь положительную длину")
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.start_tolerance_s = start_tolerance_s
        self._integral: float = 0.0
        self._last_price: float | None = None
        self._last_ts: float | None = None
        self._first_ts: float | None = None

    def update(self, price: float, ts: float) -> None:
        """Учесть тик фида. Вклад в интеграл — от предыдущего тика до этого."""
        if price <= 0:
            return
        if self._first_ts is None:
            self._first_ts = ts
        if (
            self._last_ts is not None
            and self._last_price is not None
            and ts > self._last_ts
        ):
            lo = max(self._last_ts, self.start_ts)
            hi = min(ts, self.end_ts)
            if hi > lo:
                self._integral += self._last_price * (hi - lo)
        if self._last_ts is None or ts >= self._last_ts:
            self._last_price, self._last_ts = price, ts

    @property
    def covers_window_start(self) -> bool:
        return (
            self._first_ts is not None
            and self._first_ts <= self.start_ts + self.start_tolerance_s
        )

    def state(self, now: float) -> tuple[float, float] | None:
        """
        (alpha, realized_avg) на момент now; None — начало окна не покрыто
        и реализованная часть неизвестна. При alpha == 0 (окно ещё не
        началось) второй элемент — последняя цена, математически он не
        используется: K_eff == K.
        """
        if not self.covers_window_start or self._last_price is None:
            return None
        effective_now = min(now, self.end_ts)
        elapsed = effective_now - self.start_ts
        if elapsed <= 0:
            return 0.0, self._last_price
        integral = self._integral
        assert self._last_ts is not None
        lo = max(self._last_ts, self.start_ts)
        if effective_now > lo:
            integral += self._last_price * (effective_now - lo)
        window = self.end_ts - self.start_ts
        return min(elapsed / window, 1.0), integral / elapsed


def twap_probability(
    spot: float,
    strike: float,
    seconds_left: float,
    sigma_annual: float,
    alpha: float = 0.0,
    realized_avg: float | None = None,
    drift_log: float = 0.0,
) -> float:
    """
    P(TWAP >= K) по формуле из докстринга модуля. Чистая функция — её же
    использует симулятор, чтобы математика бота и симуляции не расходились.

    drift_log — уже приведённый вклад сноса в лог-цену за ОСТАТОК окна
    (модель передаёт туда приглушённый momentum * tau/2, симулятор — 0).
    """
    if strike <= 0 or spot <= 0:
        return 0.5

    alpha = min(max(alpha, 0.0), 0.9999)
    beta = 1.0 - alpha
    if alpha > 0.0 and realized_avg is not None:
        k_eff = (strike - alpha * realized_avg) / beta
    else:
        k_eff = strike
    if k_eff <= 0:
        # Реализованная часть уже выше страйка настолько, что даже нулевые
        # будущие цены не опустят среднее ниже K.
        return 1.0

    tau_s = max(seconds_left, 2.0)
    tau_y = tau_s / SECONDS_PER_YEAR
    sigma = max(sigma_annual, 1e-4)
    vol_term = sigma * math.sqrt(tau_y / 3.0)
    if vol_term < 1e-9:
        return 1.0 if spot > k_eff else 0.0

    d = (
        math.log(spot / k_eff)
        + drift_log
        - 0.5 * sigma**2 * (tau_y / 2.0)
    ) / vol_term
    return _norm_cdf(d)


def implied_strike_twap(
    spot: float,
    market_prob: float,
    seconds_left: float,
    sigma_annual: float,
    alpha: float = 0.0,
    realized_avg: float | None = None,
) -> Decimal | None:
    """
    Обратить TWAP-модель: найти K, при котором P(TWAP >= K) == market_prob.

        z = Phi^-1(p);  v = sigma * sqrt(tau_y / 3)
        K_eff = S * exp(-z * v - sigma^2 * tau_y / 4)
        K = beta * K_eff + alpha * A_real
    """
    p = min(max(market_prob, 0.02), 0.98)
    tau_y = max(seconds_left, 5.0) / SECONDS_PER_YEAR
    sigma = max(sigma_annual, 1e-4)
    vol_term = sigma * math.sqrt(tau_y / 3.0)
    if vol_term < 1e-9 or spot <= 0:
        return None
    try:
        z = _ND.inv_cdf(p)
    except ValueError:
        return None
    k_eff = spot * math.exp(-z * vol_term - 0.5 * sigma**2 * (tau_y / 2.0))
    alpha = min(max(alpha, 0.0), 0.9999)
    if alpha > 0.0 and realized_avg is not None:
        strike = (1.0 - alpha) * k_eff + alpha * realized_avg
    else:
        strike = k_eff
    if strike <= 0 or not math.isfinite(strike):
        return None
    return Decimal(str(round(strike, 2)))


class VolatilityEstimator:
    """
    EWMA-оценка реализованной волатильности и momentum по тикам фида.

    ДВА РЕЖИМА, и перепутать их — значит сломать модель молча:

    - sample_interval_s == 0 (несглаженный ряд, Binance-спот): каждый тик,
      нормировка на sqrt(dt) — классика.
    - sample_interval_s > 0 (СГЛАЖЕННЫЙ ряд, поток Chainlink TWAP-60):
      секундные приращения скользящего среднего крошечные и сильно
      коррелированы — наивная EWMA занижает sigma в разы (на синтетике
      SMA-60 от GBM с sigma=0.55 наивная оценка даёт 0.075, x0.14), модель
      насыщается в 0/1 и fair прилипает к клипу max_model_deviation.
      Лечение: берём приращения только на лаге >= sample_interval_s и
      восстанавливаем масштаб поправкой скользящего среднего — для ряда
      SMA(W) дисперсия лаг-L приращения равна sigma^2*(L - W/3), а не
      sigma^2*L, поэтому норму делим на sqrt(1 - W/(3L)).
    """

    __slots__ = (
        "_vol_lambda",
        "_mom_lambda",
        "_var",
        "_momentum",
        "_last_price",
        "_last_ts",
        "_samples",
        "_vol_floor",
        "_sample_interval",
        "_ma_window",
        "_ready_samples",
        "_blend_samples",
    )

    def __init__(
        self,
        vol_halflife_s: float,
        momentum_halflife_s: float,
        vol_floor_annual: float,
        *,
        sample_interval_s: float = 0.0,
        ma_window_s: float = 0.0,
        ready_samples: int = 20,
    ) -> None:
        # EWMA-коэффициент из полупериода: lambda = exp(-ln2 / halflife)
        self._vol_lambda = math.exp(-math.log(2.0) / max(vol_halflife_s, 1e-6))
        self._mom_lambda = math.exp(-math.log(2.0) / max(momentum_halflife_s, 1e-6))
        self._var: float = (vol_floor_annual**2) / SECONDS_PER_YEAR  # дисперсия в сек
        self._momentum: float = 0.0
        self._last_price: float | None = None
        self._last_ts: float | None = None
        self._samples: int = 0
        self._vol_floor = vol_floor_annual
        if 0 < sample_interval_s < ma_window_s:
            raise ValueError("интервал выборки должен быть >= окна сглаживания")
        self._sample_interval = sample_interval_s
        self._ma_window = ma_window_s
        self._ready_samples = max(ready_samples, 2)
        self._blend_samples = max(3 * self._ready_samples, 10)

    def update(self, price: float, ts: float | None = None) -> None:
        """Скормить новый тик цены фида."""
        ts = ts if ts is not None else time.time()
        if price <= 0:
            return
        if self._last_price is None or self._last_ts is None:
            self._last_price, self._last_ts = price, ts
            return

        dt = max(ts - self._last_ts, 1e-3)
        # Лаг-выборка сглаженного ряда: тики чаще интервала ПРОПУСКАЕМ, не
        # сдвигая якорь — иначе снова меряем микроприращения среднего.
        if self._sample_interval > 0 and dt < self._sample_interval:
            return
        # Игнорируем дубликаты и слишком старые тики; порог масштабируется
        # интервалом выборки.
        if dt > max(30.0, self._sample_interval * 4):
            self._last_price, self._last_ts = price, ts
            return

        ret = math.log(price / self._last_price)
        # Нормируем на sqrt(dt) — приводим к дисперсии за 1 секунду.
        norm_ret = ret / math.sqrt(dt)
        # Поправка скользящего среднего (см. докстринг класса).
        if self._ma_window > 0 and dt >= self._ma_window:
            norm_ret /= math.sqrt(1.0 - self._ma_window / (3.0 * dt))

        # Адаптивный коэффициент: учитываем реальный интервал между тиками.
        lam_v = self._vol_lambda**dt
        lam_m = self._mom_lambda**dt

        self._var = lam_v * self._var + (1.0 - lam_v) * (norm_ret**2)
        # Momentum на лаг-выборке — дрейф за интервал; клип вклада в модели
        # ограничивает его шум, как и на тиковом режиме.
        self._momentum = lam_m * self._momentum + (1.0 - lam_m) * (ret / dt)

        self._last_price, self._last_ts = price, ts
        self._samples += 1

    @property
    def sigma_annual(self) -> float:
        """Годовая волатильность."""
        sigma = math.sqrt(max(self._var, 1e-12) * SECONDS_PER_YEAR)
        # Пока мало данных — подмешиваем floor, чтобы не переоценить уверенность.
        if self._samples < self._blend_samples:
            w = self._samples / float(self._blend_samples)
            sigma = w * sigma + (1.0 - w) * self._vol_floor
        return max(sigma, self._vol_floor * 0.25)

    @property
    def drift_per_second(self) -> float:
        """Оценённый momentum-дрейф лог-цены за секунду."""
        return self._momentum

    @property
    def ready(self) -> bool:
        return self._samples >= self._ready_samples

    @property
    def samples(self) -> int:
        return self._samples


class FairValueModel:
    """Считает справедливую вероятность YES и смешивает её с рынком."""

    def __init__(
        self,
        model_weight: Decimal,
        momentum_drift_coef: Decimal,
        max_model_deviation: Decimal,
    ) -> None:
        self.model_weight = float(model_weight)
        self.momentum_coef = float(momentum_drift_coef)
        self.max_deviation = float(max_model_deviation)

    def model_probability(
        self,
        spot: float,
        strike: float,
        seconds_left: float,
        sigma_annual: float,
        drift_per_second: float,
    ) -> float:
        """Чистая GBM-оценка P(S_T > K)."""
        if strike <= 0 or spot <= 0:
            return 0.5
        # Минимум 2 секунды — иначе sqrt(tau) -> 0 и d взрывается.
        tau_s = max(seconds_left, 2.0)
        tau_y = tau_s / SECONDS_PER_YEAR

        sigma = max(sigma_annual, 1e-4)
        vol_term = sigma * math.sqrt(tau_y)
        if vol_term < 1e-9:
            return 1.0 if spot > strike else 0.0

        # Дрейф: momentum за секунду * оставшиеся секунды, приглушённый коэффициентом.
        drift = self.momentum_coef * drift_per_second * tau_s
        # Ограничиваем вклад дрейфа одним стандартным отклонением — иначе
        # шумный momentum утащит fair value в крайности.
        drift = max(-vol_term, min(vol_term, drift))

        d = (math.log(spot / strike) + drift - 0.5 * sigma**2 * tau_y) / vol_term
        return _norm_cdf(d)

    def model_probability_twap(
        self,
        spot: float,
        strike: float,
        seconds_left: float,
        sigma_annual: float,
        drift_per_second: float,
        alpha: float,
        realized_avg: float | None,
    ) -> float:
        """P(TWAP >= K) с приглушённым momentum-сносом (см. докстринг модуля)."""
        tau_s = max(seconds_left, 2.0)
        tau_y = tau_s / SECONDS_PER_YEAR
        sigma = max(sigma_annual, 1e-4)
        vol_term = sigma * math.sqrt(tau_y / 3.0)
        # Снос входит в среднее остатка половиной горизонта; ограничиваем
        # его вклад одним стандартным отклонением ОСТАТКА, как и раньше.
        drift = self.momentum_coef * drift_per_second * (tau_s / 2.0)
        drift = max(-vol_term, min(vol_term, drift))
        return twap_probability(
            spot, strike, seconds_left, sigma_annual,
            alpha=alpha, realized_avg=realized_avg, drift_log=drift,
        )

    def confidence(self, seconds_left: float, vol_ready: bool) -> float:
        """
        Насколько доверять модели.
        Ближе к экспирации модель точнее по сути, но чувствительнее к
        микросекундной задержке фида — реальный бот там теряет, а не выигрывает.
        Поэтому доверие максимально в середине окна и падает по краям.
        """
        if not vol_ready:
            return 0.0
        if seconds_left < 20:
            return 0.15          # финальный рывок — рынок быстрее нас
        if seconds_left < 60:
            return 0.55
        if seconds_left > 900:
            return 0.6           # слишком далеко, вол-оценка шумит
        return 1.0

    def compute(
        self,
        *,
        spot: float,
        strike: float,
        seconds_left: float,
        market_mid: Decimal,
        sigma_annual: float,
        drift_per_second: float,
        vol_ready: bool,
        twap_alpha: float | None = None,
        twap_realized: float | None = None,
    ) -> FairValue:
        """
        Итоговый fair value: модель, смешанная с рынком и ограниченная.

        twap_alpha is not None — TWAP-режим (боевой): вероятность считается
        для среднего по окну с учётом реализованной части. None — модель
        конечной точки; в бою она не используется (рынки резолвятся по
        TWAP), оставлена для сравнительных прогонов симулятора.
        """
        if twap_alpha is not None:
            p_model = self.model_probability_twap(
                spot, strike, seconds_left, sigma_annual, drift_per_second,
                alpha=twap_alpha, realized_avg=twap_realized,
            )
        else:
            p_model = self.model_probability(
                spot, strike, seconds_left, sigma_annual, drift_per_second
            )
        mid = float(market_mid)
        conf = self.confidence(seconds_left, vol_ready)

        # Эффективный вес модели = базовый вес * доверие.
        w = self.model_weight * conf
        blended = w * p_model + (1.0 - w) * mid

        # Жёсткий клип: никогда не уходим от рынка дальше max_deviation.
        blended = max(mid - self.max_deviation, min(mid + self.max_deviation, blended))
        # И держимся внутри торгуемого диапазона.
        blended = max(0.01, min(0.99, blended))

        return FairValue(
            fair=Decimal(str(round(blended, 6))),
            model_prob=Decimal(str(round(p_model, 6))),
            market_mid=market_mid,
            edge=Decimal(str(round(p_model - mid, 6))),
            sigma_annual=Decimal(str(round(sigma_annual, 4))),
            confidence=Decimal(str(round(conf, 3))),
        )
