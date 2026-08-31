"""
Модель справедливой цены YES для рынка «BTC Up or Down к моменту T».

Математика
----------
Рынок спрашивает: будет ли S_T > K, где K — цена на момент открытия окна.
Под GBM с нулевым дрейфом:

    P(S_T > K) = Phi( (ln(S_t/K) + (mu - sigma^2/2) * tau) / (sigma * sqrt(tau)) )

где tau — доля года до экспирации, sigma — годовая волатильность.

Практические поправки, без которых модель бесполезна на 5-15 минутах:

1. sigma оценивается EWMA по логарифмическим доходностям спота с коротким
   полупериодом. На горизонте минут вол резко меняется — статичная не годится.
2. mu берётся из momentum (EWMA доходностей), но с малым коэффициентом.
   На минутном горизонте momentum крипты слабый и шумный; ставить на него
   всерьёз — путь к сливу. Он нужен, чтобы не быть на неправильной стороне.
3. tau -> 0 делает модель сингулярной: d -> +-inf, вероятность прыгает в 0/1.
   Мы явно снижаем доверие к модели по мере приближения к экспирации И
   ограничиваем максимальное отклонение от рынка.
4. Итог смешивается с рыночным microprice. Рынок агрегирует информацию,
   которой у нас нет. Полностью игнорировать его — самоуверенность.
"""

from __future__ import annotations

import math
import time
from decimal import Decimal

from .models import FairValue

SECONDS_PER_YEAR = 365.0 * 24 * 3600


def _norm_cdf(x: float) -> float:
    """Стандартная нормальная CDF через erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class VolatilityEstimator:
    """EWMA-оценка реализованной волатильности и momentum по тикам спота."""

    __slots__ = (
        "_vol_lambda",
        "_mom_lambda",
        "_var",
        "_momentum",
        "_last_price",
        "_last_ts",
        "_samples",
        "_vol_floor",
    )

    def __init__(
        self,
        vol_halflife_s: float,
        momentum_halflife_s: float,
        vol_floor_annual: float,
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

    def update(self, price: float, ts: float | None = None) -> None:
        """Скормить новый тик спот-цены."""
        ts = ts if ts is not None else time.time()
        if price <= 0:
            return
        if self._last_price is None or self._last_ts is None:
            self._last_price, self._last_ts = price, ts
            return

        dt = max(ts - self._last_ts, 1e-3)
        # Игнорируем дубликаты и слишком старые тики.
        if dt > 30.0:
            self._last_price, self._last_ts = price, ts
            return

        ret = math.log(price / self._last_price)
        # Нормируем на sqrt(dt) — приводим к дисперсии за 1 секунду.
        norm_ret = ret / math.sqrt(dt)

        # Адаптивный коэффициент: учитываем реальный интервал между тиками.
        lam_v = self._vol_lambda**dt
        lam_m = self._mom_lambda**dt

        self._var = lam_v * self._var + (1.0 - lam_v) * (norm_ret**2)
        self._momentum = lam_m * self._momentum + (1.0 - lam_m) * (ret / dt)

        self._last_price, self._last_ts = price, ts
        self._samples += 1

    @property
    def sigma_annual(self) -> float:
        """Годовая волатильность."""
        sigma = math.sqrt(max(self._var, 1e-12) * SECONDS_PER_YEAR)
        # Пока мало данных — подмешиваем floor, чтобы не переоценить уверенность.
        if self._samples < 60:
            w = self._samples / 60.0
            sigma = w * sigma + (1.0 - w) * self._vol_floor
        return max(sigma, self._vol_floor * 0.25)

    @property
    def drift_per_second(self) -> float:
        """Оценённый momentum-дрейф лог-цены за секунду."""
        return self._momentum

    @property
    def ready(self) -> bool:
        return self._samples >= 20

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
    ) -> FairValue:
        """Итоговый fair value: модель, смешанная с рынком и ограниченная."""
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
