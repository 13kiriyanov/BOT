"""
Поиск активных краткосрочных Up/Down рынков BTC/ETH и калибровка страйка.

ПРОБЛЕМА СТРАЙКА
----------------
Рынок `btc-updown-5m-<ts>` резолвится сравнением TWAP Chainlink за окно со
значением того же ряда в НАЧАЛЕ окна — это и есть страйк. Через API он не
отдаётся: если бот стартовал в середине окна, момент начала уже прошёл.

Стратегии, по убыванию надёжности:
 1. Бот наблюдал начало окна -> записанное значение фида резолюции
    (движок ловит пересечение start_ts и зовёт observe_window_open).
 2. Страйк указан в описании рынка -> парсим число.
 3. Инверсия TWAP-модели по живому рынку — в ДВИЖКЕ
    (implied_strike_twap из fair_value.py, нужна реализованная часть окна):
    находим K, при котором модельная вероятность совпадает с рыночным mid,
    и дальше edge возникает из РАСХОЖДЕНИЯ движения фида и движения рынка.
    Это не хак: торгуют изменение, а не уровень, потому что уровень
    зависит от неизвестного нам параметра.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from polymarket import AsyncSecureClient

from .models import ONE, ZERO, TargetMarket

log = logging.getLogger("polybot.discovery")

# Определение актива по тексту вопроса/слага.
ASSET_PATTERNS = {
    "BTC": re.compile(r"\b(bitcoin|btc)\b", re.I),
    "ETH": re.compile(r"\b(ethereum|eth)\b", re.I),
}
# Числа вида "$67,432.10" или "67432.1" в описании.
PRICE_RE = re.compile(r"\$?\s*([0-9]{3,7}(?:,[0-9]{3})*(?:\.[0-9]+)?)")
# Unix-время начала окна в хвосте слага: btc-updown-5m-1788276000.
SLUG_START_TS_RE = re.compile(r"-(\d{10})$")
# Окно потока TWAP из описания рынка: ссылка вида
# https://data.chain.link/streams/btc-usd-twap-60s-streams.
DESC_TWAP_WINDOW_RE = re.compile(r"twap-(\d{2,3})s", re.I)
# Длительность окна из слага: btc-updown-5m-..., eth-updown-15m-..., -4h-.
SLUG_DURATION_RE = re.compile(r"-(\d+)(m|h)-\d{10}$")
# Окно TWAP по длительности рынка — анонс Polymarket (7 августа 2026):
# 30s для 5-минутных, 60s для 15-минутных и 4-часовых. Резерв, если в
# описании ссылки на поток нет.
TWAP_WINDOW_BY_DURATION_S = {300: 30, 900: 60, 4 * 3600: 60}
SUPPORTED_TWAP_WINDOWS = (30, 60)


def parse_twap_window_s(description: str, slug: str) -> int | None:
    """
    Окно потока Chainlink TWAP, по которому резолвится рынок.

    1. Явная ссылка в описании (`...-twap-30s-streams`) — истина.
    2. Иначе по длительности окна из слага (таблица анонса).
    None — определить нельзя (рынок без модели, а не с чужим рядом).
    """
    match = DESC_TWAP_WINDOW_RE.search(description or "")
    if match:
        window = int(match.group(1))
        return window if window in SUPPORTED_TWAP_WINDOWS else None
    duration = SLUG_DURATION_RE.search(slug or "")
    if duration:
        amount, unit = int(duration.group(1)), duration.group(2)
        seconds = amount * (60 if unit == "m" else 3600)
        return TWAP_WINDOW_BY_DURATION_S.get(seconds)
    return None
# Ранняя остановка поиска по endDate: берём события, истекающие не позже
# max_seconds + этот запас. Дальше по ascending-порядку — только позже.
SEARCH_END_LOOKAHEAD_S = 120.0
# Предохранитель от бесконечной выборки, если сортировка не сработала.
SEARCH_MAX_EVENTS = 200


def parse_slug_start_ts(slug: str) -> float | None:
    """Достать unix-время начала окна из хвоста слага; None — его там нет."""
    match = SLUG_START_TS_RE.search(slug or "")
    if not match:
        return None
    ts = float(match.group(1))
    # Диапазон правдоподобия: 2017..2038. Чужие десятизначные числа мимо.
    if not (1.5e9 <= ts <= 2.2e9):
        return None
    return ts


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


def _active_rewards(rewards) -> tuple[Decimal | None, str | None]:  # noqa: ANN001
    """
    Суммарная дневная ставка наград по действующим записям clobRewards
    (end_date пустой или не раньше сегодняшнего дня) и самая поздняя дата
    окончания. (None, None) — программы у рынка нет.
    """
    entries = getattr(rewards, "clob_rewards", None) or []
    if not entries:
        return None, None
    today = datetime.now(timezone.utc).date()
    total = Decimal("0")
    latest_end: str | None = None
    for entry in entries:
        end = getattr(entry, "end_date", None)
        if end is not None and end < today:
            continue
        rate = getattr(entry, "rewards_daily_rate", None)
        if rate is not None:
            total += Decimal(str(rate))
        if end is not None and (latest_end is None or str(end) > latest_end):
            latest_end = str(end)
    return (total if total > 0 else None), latest_end


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


# Инверсия модели «конечная точка» жила здесь до подтверждения TWAP-правил
# резолюции. Рынок ценит СРЕДНЕЕ по окну — правильная инверсия теперь
# implied_strike_twap в fair_value.py (её зовёт движок с реализованной
# частью окна из RealizedTwap).


@dataclass(slots=True)
class DiscoveryFunnel:
    """
    Счётчики одной итерации поиска: сколько кандидатов пришло и на каком
    фильтре сколько отсеялось. «Рынков=0» без этой раскладки неотличимо
    от «поиск сломался» — а это разные аварии с разными починками.
    """

    search_events: int = 0      # событий из полнотекстового поиска (основной)
    series_events: int = 0      # событий из заданных серий (резерв)
    markets_in: int = 0         # рынков внутри пришедших событий
    drop_state: int = 0         # not active / closed / archived / нет state
    drop_accepting: int = 0     # accepting_orders=False
    drop_book: int = 0          # enable_order_book=False
    drop_no_end: int = 0        # нет даты экспирации
    drop_window_far: int = 0    # до экспирации больше max (ещё рано)
    drop_window_near: int = 0   # до экспирации меньше min (уже поздно)
    drop_no_tokens: int = 0     # нет token_id одной из ног
    drop_no_asset: int = 0      # не распознан BTC/ETH
    drop_no_twap_window: int = 0  # не определить окно потока резолюции
    drop_dup: int = 0           # дубль condition_id
    drop_overflow: int = 0      # прошли всё, но не влезли в max_markets
    accepted: int = 0           # итог: сколько рынков торгуем

    def describe(self) -> str:
        """Одна строка для INFO-лога discovery_loop."""
        return (
            f"событий: поиск={self.search_events} серии={self.series_events}; "
            f"рынков={self.markets_in}, взято={self.accepted}; отсев: "
            f"состояние={self.drop_state} приём_ордеров={self.drop_accepting} "
            f"стакан={self.drop_book} без_даты={self.drop_no_end} "
            f"окно_рано={self.drop_window_far} окно_поздно={self.drop_window_near} "
            f"токены={self.drop_no_tokens} актив={self.drop_no_asset} "
            f"окно_twap={self.drop_no_twap_window} "
            f"дубли={self.drop_dup} сверх_лимита={self.drop_overflow}"
        )


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
        # Воронка последнего find_markets — её печатает discovery_loop.
        self.last_funnel: DiscoveryFunnel = DiscoveryFunnel()

    def observe_window_open(self, condition_id: str, spot: Decimal) -> None:
        """Вызывается движком, когда рынок появился до старта своего окна."""
        self._observed_open.setdefault(condition_id, spot)

    # ------------------------------------------------------------- поиск

    async def _candidates_from_search(self) -> list:
        """
        ОСНОВНОЙ путь: полнотекстовый поиск по заголовкам. На живом API
        именно он находит updown-рынки; запросы по слагам серий стабильно
        возвращают ноль (см. _candidates_from_series — теперь это резерв).
        """
        events = []
        # Выборка отсортирована по endDate по возрастанию, а торгуем мы
        # только окно до max_seconds: всё, что истекает позже, не нужно.
        # Без ранней остановки поиск сливал ~900 событий (десятки страниц
        # API) каждый цикл discovery.
        cutoff = time.time() + self.max_seconds + SEARCH_END_LOOKAHEAD_S
        for kw in self.title_keywords:
            try:
                paginator = self.client.list_events(
                    title_search=kw, closed=False, order="endDate",
                    ascending=True, page_size=20,
                )
                taken = 0
                async for ev in paginator.iter_items():
                    end_ts = _parse_ts(
                        getattr(getattr(ev, "schedule", None), "end_date", None)
                    )
                    if end_ts is not None and end_ts > cutoff:
                        break
                    events.append(ev)
                    taken += 1
                    if taken >= SEARCH_MAX_EVENTS:
                        log.warning(
                            "Поиск '%s': упёрся в предохранитель %d событий — "
                            "сортировка по endDate не сработала?",
                            kw, SEARCH_MAX_EVENTS,
                        )
                        break
            except Exception as exc:  # noqa: BLE001
                log.warning("Поиск по заголовку '%s' не удался: %s", kw, exc)
        return events

    async def _candidates_from_series(self) -> list:
        """Резерв: рынки из заданных серий, если поиск не дал ничего."""
        events = []
        for slug in self.series_slugs:
            try:
                paginator = self.client.list_series(slug=slug, closed=False, page_size=5)
                # Пагинатор итерируется СТРАНИЦАМИ (Page); элементы — только
                # через iter_items(). См. канарейку в tests/test_units.py.
                n_series = 0
                async for series in paginator.iter_items():
                    n_series += 1
                    for ev in getattr(series, "events", None) or []:
                        events.append(ev)
                if n_series == 0:
                    log.warning(
                        "Серия '%s': API вернул ноль открытых серий — слаг "
                        "устарел или серия закрыта (проверь diag.py)", slug,
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("Серия '%s' недоступна: %s", slug, exc)
        return events

    async def find_markets(self, spot_prices: dict[str, float]) -> list[TargetMarket]:
        """Вернуть список торгуемых прямо сейчас рынков."""
        funnel = DiscoveryFunnel()
        self.last_funnel = funnel

        events = await self._candidates_from_search()
        funnel.search_events = len(events)
        if not events and self.series_slugs:
            events = await self._candidates_from_series()
            funnel.series_events = len(events)
        if not events:
            log.warning(
                "Активных Up/Down рынков не найдено: поиск по %s%s вернул "
                "ноль событий", self.title_keywords,
                f" и серии {self.series_slugs}" if self.series_slugs else "",
            )
            return []

        now = time.time()
        found: list[TargetMarket] = []
        seen: set[str] = set()

        for ev in events:
            for m in getattr(ev, "markets", None) or []:
                funnel.markets_in += 1
                target = self._build_target(m, ev, now, spot_prices, funnel)
                if target is None:
                    continue
                if target.condition_id in seen:
                    funnel.drop_dup += 1
                    continue
                seen.add(target.condition_id)
                found.append(target)

        # Ближайшие к экспирации — приоритетнее: там выше активность.
        found.sort(key=lambda t: t.seconds_left)
        result = found[: self.max_markets]
        funnel.drop_overflow = len(found) - len(result)
        funnel.accepted = len(result)
        return result

    def _build_target(
        self, m, ev, now: float, spot_prices: dict[str, float],
        funnel: DiscoveryFunnel,
    ) -> TargetMarket | None:  # noqa: ANN001
        state = getattr(m, "state", None)
        if state is None or not state.active or state.closed or state.archived:
            funnel.drop_state += 1
            return None
        if not state.accepting_orders:
            funnel.drop_accepting += 1
            return None
        if not state.enable_order_book:
            funnel.drop_book += 1
            return None

        end_ts = _parse_ts(getattr(state, "end_date", None))
        if end_ts is None:
            funnel.drop_no_end += 1
            return None
        left = end_ts - now
        if left > self.max_seconds:
            funnel.drop_window_far += 1
            return None
        if left < self.min_seconds:
            funnel.drop_window_near += 1
            return None

        outcomes = getattr(m, "outcomes", None)
        if outcomes is None or not outcomes.yes or not outcomes.no:
            funnel.drop_no_tokens += 1
            return None
        yes_token = str(outcomes.yes.token_id or "")
        no_token = str(outcomes.no.token_id or "")
        if not yes_token or not no_token:
            funnel.drop_no_tokens += 1
            return None

        text = f"{getattr(m, 'question', '')} {getattr(m, 'slug', '')} {getattr(ev, 'title', '')}"
        asset = detect_asset(text)
        if asset is None:
            funnel.drop_no_asset += 1
            return None

        slug = str(getattr(m, "slug", ""))
        description = str(getattr(m, "description", "") or "")
        # Ряд резолюции ЭТОГО рынка. Не определить — не торгуем: кормить
        # модель чужим окном хуже, чем пропустить рынок.
        twap_window = parse_twap_window_s(description, slug)
        if twap_window is None:
            funnel.drop_no_twap_window += 1
            return None

        trading = getattr(m, "trading", None)
        tick = Decimal(str(getattr(trading, "minimum_tick_size", None) or "0.01"))
        min_size = Decimal(str(getattr(trading, "minimum_order_size", None) or "5"))
        rewards = getattr(m, "rewards", None)
        fee_rate, fee_exponent = self._resolve_fees(trading, slug)
        daily_rate, rewards_end = _active_rewards(rewards)

        target = TargetMarket(
            condition_id=str(m.condition_id),
            slug=slug,
            question=str(getattr(m, "question", "")),
            yes_token_id=yes_token,
            no_token_id=no_token,
            end_ts=end_ts,
            start_ts=parse_slug_start_ts(slug),
            twap_window_s=twap_window,
            rewards_daily_rate=daily_rate,
            rewards_end_date=rewards_end,
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
        """Стратегии определения страйка, по убыванию надёжности."""
        cid = target.condition_id
        if cid in self._strike_cache:
            return self._strike_cache[cid]

        spot = spot_prices.get(target.asset)

        # 1. Наблюдали открытие окна (сюда же пишет движок через
        #    observe_window_open, поймав момент start_ts на живом споте).
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

        # 3. Инверсию модели по рыночному mid здесь НЕ делаем. Раньше она
        #    калибровалась по prices из Gamma REST (кэшированным и легко
        #    устаревшим) с захардкоженной сигмой 0.45 — и кэшировала ошибку
        #    навсегда: на живом фиде модель уходила на 0.68 от рынка. Теперь
        #    калибрует движок — отложенно, по живому стакану, свежему споту
        #    и живой вол-оценке (см. engine._try_calibrate_strike), а сюда
        #    результат приходит через set_strike.
        log.debug(
            "[%s] Страйк при discovery не определён — отложенная калибровка "
            "по живым данным", target.slug,
        )
        return None

    # ------------------------------------------------ страйк: API движка

    def set_strike(self, condition_id: str, strike: Decimal) -> None:
        """Запомнить страйк, откалиброванный движком по живым данным."""
        self._strike_cache[condition_id] = strike

    def invalidate_strike(self, condition_id: str) -> None:
        """
        Сторож расхождения признал страйк ложным: забыть все его источники,
        чтобы пересборка рынка не восстановила то же неверное значение.
        """
        self._strike_cache.pop(condition_id, None)
        self._observed_open.pop(condition_id, None)

    def forget(self, condition_id: str) -> None:
        self._strike_cache.pop(condition_id, None)
        self._observed_open.pop(condition_id, None)
