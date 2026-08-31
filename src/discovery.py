"""
Поиск активных краткосрочных Up/Down рынков BTC/ETH и калибровка страйка.

ПРОБЛЕМА СТРАЙКА
----------------
Рынок «Bitcoin Up or Down 3PM ET» резолвится сравнением цены на момент
экспирации с ценой на момент открытия окна. Эта опорная цена (страйк) не
всегда доступна через API: если бот стартовал в середине окна, момент
открытия уже прошёл.

Три стратегии, по убыванию надёжности:
 1. Бот наблюдал открытие окна -> берём записанный спот.
 2. Страйк указан в описании рынка -> парсим число.
 3. Инвертируем модель: находим K такое, что GBM-вероятность совпадает с
    текущим рыночным mid. Модель калибруется на рынок в момент t0, и дальше
    edge возникает из РАСХОЖДЕНИЯ движения спота и движения рынка.
    Это не хак: именно так и делают на практике — торгуют изменение, а не
    уровень, потому что уровень зависит от неизвестного нам параметра.
"""

from __future__ import annotations

import logging
import math
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from statistics import NormalDist

from polymarket import AsyncSecureClient

from .models import ONE, ZERO, TargetMarket

log = logging.getLogger("polybot.discovery")

SECONDS_PER_YEAR = 365.0 * 24 * 3600
_ND = NormalDist()

# Определение актива по тексту вопроса/слага.
ASSET_PATTERNS = {
    "BTC": re.compile(r"\b(bitcoin|btc)\b", re.I),
    "ETH": re.compile(r"\b(ethereum|eth)\b", re.I),
}
# Числа вида "$67,432.10" или "67432.1" в описании.
PRICE_RE = re.compile(r"\$?\s*([0-9]{3,7}(?:,[0-9]{3})*(?:\.[0-9]+)?)")


def _parse_ts(value) -> float | None:  # noqa: ANN001
    """Привести дату из API к unix-времени."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    try:
        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def detect_asset(text: str) -> str | None:
    for asset, pattern in ASSET_PATTERNS.items():
        if pattern.search(text):
            return asset
    return None


def parse_strike_from_text(text: str, spot_hint: float | None) -> Decimal | None:
    """
    Вытащить опорную цену из описания. Берём число, ближайшее к текущему споту
    (описание содержит и другие числа — даты, проценты, номера).
    """
    if not text:
        return None
    candidates: list[float] = []
    for match in PRICE_RE.finditer(text):
        try:
            candidates.append(float(match.group(1).replace(",", "")))
        except ValueError:
            continue
    if not candidates:
        return None
    if spot_hint is None:
        return None
    # Допускаем отклонение не больше 10% от спота — иначе это не страйк.
    plausible = [c for c in candidates if abs(c - spot_hint) / spot_hint < 0.10]
    if not plausible:
        return None
    best = min(plausible, key=lambda c: abs(c - spot_hint))
    return Decimal(str(best))


def implied_strike(
    spot: float, market_prob: float, seconds_left: float, sigma_annual: float
) -> Decimal | None:
    """
    Обратить GBM: найти K, при котором P(S_T > K) == market_prob.

        d = (ln(S/K) - 0.5*sigma^2*tau) / (sigma*sqrt(tau)) = Phi^-1(p)
        =>  K = S * exp(-Phi^-1(p) * sigma * sqrt(tau) - 0.5*sigma^2*tau)
    """
    p = min(max(market_prob, 0.02), 0.98)  # клип, иначе inv_cdf -> +-inf
    tau_y = max(seconds_left, 5.0) / SECONDS_PER_YEAR
    sigma = max(sigma_annual, 1e-4)
    vol_term = sigma * math.sqrt(tau_y)
    if vol_term < 1e-9:
        return None
    try:
        z = _ND.inv_cdf(p)
    except (ValueError, InvalidOperation):
        return None
    k = spot * math.exp(-z * vol_term - 0.5 * sigma**2 * tau_y)
    if k <= 0 or not math.isfinite(k):
        return None
    return Decimal(str(round(k, 2)))


class MarketDiscovery:
    """Периодически ищет подходящие рынки и калибрует страйки."""

    def __init__(
        self,
        client: AsyncSecureClient,
        *,
        series_slugs: list[str],
        title_keywords: list[str],
        min_seconds: int,
        max_seconds: int,
        max_markets: int,
        fallback_fee_rate: Decimal = ZERO,
    ) -> None:
        self.client = client
        self.series_slugs = series_slugs
        self.title_keywords = title_keywords
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.max_markets = max_markets
        self.fallback_fee_rate = fallback_fee_rate
        # Спот, записанный на момент открытия окна: condition_id -> price.
        self._observed_open: dict[str, Decimal] = {}
        self._strike_cache: dict[str, Decimal] = {}

    def observe_window_open(self, condition_id: str, spot: Decimal) -> None:
        """Вызывается движком, когда рынок появился до старта своего окна."""
        self._observed_open.setdefault(condition_id, spot)

    # ------------------------------------------------------------- поиск

    async def _candidates_from_series(self) -> list:
        """Основной путь: рынки из заданных серий."""
        events = []
        for slug in self.series_slugs:
            try:
                paginator = self.client.list_series(slug=slug, closed=False, page_size=5)
                async for series in paginator:
                    for ev in getattr(series, "events", None) or []:
                        events.append(ev)
            except Exception as exc:  # noqa: BLE001
                log.debug("Серия %s недоступна: %s", slug, exc)
        return events

    async def _candidates_from_search(self) -> list:
        """Резерв: полнотекстовый поиск, если серии не отдали событий."""
        events = []
        for kw in self.title_keywords:
            try:
                paginator = self.client.list_events(
                    title_search=kw, closed=False, order="endDate",
                    ascending=True, page_size=20,
                )
                async for ev in paginator:
                    events.append(ev)
            except Exception as exc:  # noqa: BLE001
                log.debug("Поиск '%s' не удался: %s", kw, exc)
        return events

    async def find_markets(self, spot_prices: dict[str, float]) -> list[TargetMarket]:
        """Вернуть список торгуемых прямо сейчас рынков."""
        events = await self._candidates_from_series()
        if not events:
            events = await self._candidates_from_search()
        if not events:
            log.warning("Активных Up/Down рынков не найдено")
            return []

        now = time.time()
        found: list[TargetMarket] = []
        seen: set[str] = set()

        for ev in events:
            for m in getattr(ev, "markets", None) or []:
                target = self._build_target(m, ev, now, spot_prices)
                if target and target.condition_id not in seen:
                    seen.add(target.condition_id)
                    found.append(target)

        # Ближайшие к экспирации — приоритетнее: там выше активность.
        found.sort(key=lambda t: t.seconds_left)
        return found[: self.max_markets]

    def _build_target(
        self, m, ev, now: float, spot_prices: dict[str, float]
    ) -> TargetMarket | None:  # noqa: ANN001
        state = getattr(m, "state", None)
        if state is None or not state.active or state.closed or state.archived:
            return None
        if not state.accepting_orders or not state.enable_order_book:
            return None

        end_ts = _parse_ts(getattr(state, "end_date", None))
        if end_ts is None:
            return None
        left = end_ts - now
        if not (self.min_seconds <= left <= self.max_seconds):
            return None

        outcomes = getattr(m, "outcomes", None)
        if outcomes is None or not outcomes.yes or not outcomes.no:
            return None
        yes_token = str(outcomes.yes.token_id)
        no_token = str(outcomes.no.token_id)
        if not yes_token or not no_token:
            return None

        text = f"{getattr(m, 'question', '')} {getattr(m, 'slug', '')} {getattr(ev, 'title', '')}"
        asset = detect_asset(text)
        if asset is None:
            return None

        trading = getattr(m, "trading", None)
        tick = Decimal(str(getattr(trading, "minimum_tick_size", None) or "0.01"))
        min_size = Decimal(str(getattr(trading, "minimum_order_size", None) or "5"))
        rewards = getattr(m, "rewards", None)
        fee_rate, fee_exponent = self._resolve_fees(trading, str(getattr(m, "slug", "")))

        target = TargetMarket(
            condition_id=str(m.condition_id),
            slug=str(getattr(m, "slug", "")),
            question=str(getattr(m, "question", "")),
            yes_token_id=yes_token,
            no_token_id=no_token,
            end_ts=end_ts,
            tick_size=tick,
            min_order_size=min_size,
            neg_risk=bool(getattr(state, "neg_risk", False)),
            asset=asset,
            fees_enabled=bool(getattr(trading, "fees_enabled", False)),
            fee_rate=fee_rate,
            fee_exponent=fee_exponent,
            rewards_max_spread=(
                Decimal(str(rewards.rewards_max_spread))
                if rewards and getattr(rewards, "rewards_max_spread", None)
                else None
            ),
            rewards_min_size=(
                Decimal(str(rewards.rewards_min_size))
                if rewards and getattr(rewards, "rewards_min_size", None)
                else None
            ),
        )
        target.strike = self._resolve_strike(target, m, spot_prices)
        return target

    def _resolve_fees(self, trading, slug: str) -> tuple[Decimal, Decimal]:  # noqa: ANN001
        """
        Ставка комиссии, которую заплатим МЫ, и её экспонента.

        Все наши ордера post_only=True, то есть мы всегда мейкер. Поэтому
        расписание, помеченное taker_only, нас не касается — для нас такой
        рынок бесплатный. Если же комиссии включены, а расписание не пришло,
        берём консервативную ставку из конфига: недооценка комиссии означает
        котирование с отрицательной чистой маржой, и заметить это по логам
        почти невозможно.
        """
        if not bool(getattr(trading, "fees_enabled", False)):
            return ZERO, ZERO

        schedule = getattr(trading, "fee_schedule", None)
        if schedule is None:
            log.warning(
                "[%s] fees_enabled=true, но расписание комиссий не отдано — "
                "считаю по консервативной ставке %s",
                slug, self.fallback_fee_rate,
            )
            return self.fallback_fee_rate, ONE

        if bool(getattr(schedule, "taker_only", False)):
            log.info("[%s] Комиссия taker-only, а мы всегда мейкер => 0", slug)
            return ZERO, ZERO

        raw_rate = getattr(schedule, "rate", None)
        raw_exponent = getattr(schedule, "exponent", None)
        rate = Decimal(str(raw_rate)) if raw_rate is not None else self.fallback_fee_rate
        exponent = Decimal(str(raw_exponent)) if raw_exponent is not None else ONE
        log.info("[%s] Комиссия рынка: rate=%s exponent=%s", slug, rate, exponent)
        return max(rate, ZERO), max(exponent, ZERO)

    def _resolve_strike(
        self, target: TargetMarket, m, spot_prices: dict[str, float]
    ) -> Decimal | None:  # noqa: ANN001
        """Три стратегии определения страйка, по убыванию надёжности."""
        cid = target.condition_id
        if cid in self._strike_cache:
            return self._strike_cache[cid]

        spot = spot_prices.get(target.asset)

        # 1. Наблюдали открытие окна.
        if cid in self._observed_open:
            strike = self._observed_open[cid]
            log.info("[%s] Страйк из наблюдения открытия: %s", target.slug, strike)
            self._strike_cache[cid] = strike
            return strike

        # 2. Парсим описание.
        parsed = parse_strike_from_text(getattr(m, "description", "") or "", spot)
        if parsed is not None:
            log.info("[%s] Страйк из описания: %s", target.slug, parsed)
            self._strike_cache[cid] = parsed
            return parsed

        # 3. Инвертируем модель по рыночной цене.
        prices = getattr(m, "prices", None)
        mid = None
        if prices is not None:
            bb = getattr(prices, "best_bid", None)
            ba = getattr(prices, "best_ask", None)
            if bb is not None and ba is not None:
                mid = (float(bb) + float(ba)) / 2.0
        if spot is not None and mid is not None:
            # Используем консервативную вол-оценку для калибровки.
            strike = implied_strike(spot, mid, target.seconds_left, 0.45)
            if strike is not None:
                log.info(
                    "[%s] Страйк калиброван по рынку (mid=%.3f): %s",
                    target.slug, mid, strike,
                )
                self._strike_cache[cid] = strike
                return strike

        log.warning("[%s] Страйк не определён — directional отключён", target.slug)
        return None

    def forget(self, condition_id: str) -> None:
        self._strike_cache.pop(condition_id, None)
        self._observed_open.pop(condition_id, None)
