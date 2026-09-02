#!/usr/bin/env python3
"""
Диагностика поиска рынков: что РЕАЛЬНО отдаёт Gamma API Polymarket.

Зачем: «рынков=0» в статусе бота неотличимо от «слаги серий устарели»,
«серии больше не содержат рынков» и «фильтры отсеивают всё». Этот скрипт
разбирает вопрос по слоям и печатает факты, на которые можно смотреть.

Что делает:
 1. Печатает настройки поиска бота (слаги, ключевые слова, окно).
 2. Повторяет ТОЧНЫЕ запросы бота по series_slugs и показывает, что пришло.
 3. Сканирует ВСЕ открытые серии и печатает те, чьи слаг/заголовок содержат
    up-or-down / updown / up-down / twap — здесь видно актуальные слаги,
    включая новые «Up/Down TWAP» серии.
 4. Пробует list_events по нескольким вариантам заголовка.
 5. Сводит найденные рынки в таблицу: slug, время до экспирации,
    accepting_orders, enable_order_book, token_id обеих ног — и вердикт
    каждого фильтра бота.
 6. Прогоняет НАСТОЯЩИЙ MarketDiscovery.find_markets() с настройками бота
    и печатает его воронку фильтров.

Безопасность: только чтение публичных данных через AsyncPublicClient —
у этого клиента нет ни ключей, ни методов ордеров, отправить ордер он
не может физически. Работает без .env; DRY_RUN значения не имеет.

Запуск:
    python diag.py                      # всё сразу
    python diag.py --max-series 500     # быстрее: короче скан серий
    python diag.py --horizon-hours 6    # уже: таблица ближайших рынков

Коды выхода: 0 — бот взял бы хотя бы один рынок прямо сейчас; 2 — ни
одного (смотри таблицу и воронку: там написано, почему); 1 — сбой скрипта.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from decimal import Decimal

from polymarket import AsyncPublicClient

from src.config import StrategySettings
from src.discovery import (
    PRICE_RE,
    MarketDiscovery,
    _parse_ts,
    detect_asset,
    parse_slug_start_ts,
    parse_twap_window_s,
)

# Предел page_size у Gamma для list_series (больше — UserInputError/400).
SERIES_PAGE_SIZE = 50

# Маркеры интересных серий/событий в слагах и заголовках.
MARKERS = ("up-or-down", "up or down", "up/down", "updown", "up-down", "twap")

# Дополнительные варианты заголовков для list_events, сверх настроенных.
EXTRA_TITLE_PROBES = (
    "Up or Down",
    "TWAP",
    "Bitcoin TWAP",
    "Ethereum TWAP",
    "Bitcoin Up or Down",
    "Ethereum Up or Down",
)


def matches_markers(*texts: str | None) -> bool:
    for text in texts:
        if not text:
            continue
        low = text.lower()
        if any(marker in low for marker in MARKERS):
            return True
    return False


def fmt_left(seconds: float | None) -> str:
    """Время до экспирации в человеческом виде."""
    if seconds is None:
        return "нет даты"
    if seconds < 0:
        return f"истёк {-seconds / 60:.0f}м назад"
    if seconds < 3600:
        return f"{int(seconds // 60):d}м {int(seconds % 60):02d}с"
    return f"{seconds / 3600:.1f}ч"


def fmt_bool(value: object) -> str:
    if value is True:
        return "да"
    if value is False:
        return "НЕТ"
    return "n/a"


def market_verdict(m, now: float, cfg: StrategySettings) -> str:  # noqa: ANN001
    """Тот же порядок фильтров, что в MarketDiscovery._build_target."""
    state = getattr(m, "state", None)
    if state is None or not state.active or state.closed or state.archived:
        return "отсев: закрыт/неактивен"
    if not state.accepting_orders:
        return "отсев: не принимает ордера"
    if not state.enable_order_book:
        return "отсев: нет стакана CLOB"
    end_ts = _parse_ts(getattr(state, "end_date", None))
    if end_ts is None:
        return "отсев: нет даты экспирации"
    left = end_ts - now
    if left > cfg.max_seconds_to_expiry:
        return f"отсев: рано (окно начнётся через {fmt_left(left - cfg.max_seconds_to_expiry)})"
    if left < cfg.min_seconds_to_expiry:
        return "отсев: поздно (окно почти истекло)"
    outcomes = getattr(m, "outcomes", None)
    if (outcomes is None or not outcomes.yes or not outcomes.no
            or not outcomes.yes.token_id or not outcomes.no.token_id):
        return "отсев: нет token_id ног"
    text = f"{getattr(m, 'question', '')} {getattr(m, 'slug', '')}"
    if detect_asset(text) is None:
        return "отсев: актив не распознан (не BTC/ETH)"
    return "✓ БОТ ВЗЯЛ БЫ"


def print_market(m, ev_title: str, now: float, cfg: StrategySettings) -> bool:  # noqa: ANN001
    """Напечатать рынок с полями, которые просят чаще всего. True = взял бы."""
    state = getattr(m, "state", None)
    end_ts = _parse_ts(getattr(state, "end_date", None)) if state else None
    left = None if end_ts is None else end_ts - now
    outcomes = getattr(m, "outcomes", None)
    yes = getattr(outcomes, "yes", None)
    no = getattr(outcomes, "no", None)
    verdict = market_verdict(m, now, cfg)

    print(f"  {getattr(m, 'slug', None) or getattr(m, 'question', '?')}")
    print(f"    событие: {ev_title}")
    print(
        f"    до экспирации: {fmt_left(left)} | "
        f"accepting_orders: {fmt_bool(getattr(state, 'accepting_orders', None))} | "
        f"enable_order_book: {fmt_bool(getattr(state, 'enable_order_book', None))} | "
        f"active: {fmt_bool(getattr(state, 'active', None))}"
    )
    yes_label = getattr(yes, "label", "?")
    no_label = getattr(no, "label", "?")
    print(f"    token {yes_label}: {getattr(yes, 'token_id', None) or '—'}")
    print(f"    token {no_label}:  {getattr(no, 'token_id', None) or '—'}")

    # Подсказки для страйка: время старта окна из слага и числа из описания.
    # По ним видно, отдаёт ли API опорную цену прямо в полях рынка.
    start_ts = parse_slug_start_ts(str(getattr(m, "slug", "") or ""))
    if start_ts is not None:
        delta = now - start_ts
        when = f"{delta:.0f}с назад" if delta >= 0 else f"через {-delta:.0f}с"
        print(f"    старт окна из слага: {start_ts:.0f} ({when})")
    description = str(getattr(m, "description", "") or "")
    numbers = [match.group(1) for match in PRICE_RE.finditer(description)][:5]
    if numbers:
        print(f"    числа в описании (кандидаты в страйк): {', '.join(numbers)}")
    elif description:
        print("    описание есть, но чисел (страйка) в нём нет")

    # Ряд резолюции: 30s для 5-минутных, 60s для 15-минутных/4-часовых.
    slug = str(getattr(m, "slug", "") or "")
    window = parse_twap_window_s(description, slug)
    if window is None:
        print("    окно TWAP резолюции: НЕ ОПРЕДЕЛЕНО — бот такой рынок отсеет")
    else:
        source = "из описания" if "twap-" in description.lower() else "по длительности слага"
        print(f"    окно TWAP резолюции: {window}s ({source})")

    print_rewards(m)
    print(f"    вердикт: {verdict}")
    return verdict.startswith("✓")


def print_rewards(m) -> None:  # noqa: ANN001
    """Реальные параметры программы ликвидити-наград рынка (Gamma)."""
    rewards = getattr(m, "rewards", None)
    if rewards is None:
        print("    награды: поле rewards не отдано")
        return
    max_spread = getattr(rewards, "rewards_max_spread", None)
    min_size = getattr(rewards, "rewards_min_size", None)
    holding = getattr(rewards, "holding_rewards_enabled", None)
    print(
        f"    награды: max_spread={max_spread}¢ min_size={min_size} "
        f"holding_rewards={fmt_bool(holding)}"
    )
    entries = getattr(rewards, "clob_rewards", None) or []
    if not entries:
        print("      clobRewards: пусто — рынок НЕ в программе наград (или данные не отданы)")
        return
    for entry in entries:
        start = getattr(entry, "start_date", None)
        end = getattr(entry, "end_date", None)
        asset = str(getattr(entry, "asset_address", "") or "")
        print(
            f"      ставка {getattr(entry, 'rewards_daily_rate', '?')} /день, "
            f"amount={getattr(entry, 'rewards_amount', '?')}, "
            f"{start} → {end or 'бессрочно'}, актив {asset[:10]}…"
        )


# «In-game multiplier» c формулы наград: для крипторынков в docs не назван,
# 3 — распространённая цифра, НЕ проверена. Используется только для нижней
# границы конкуренции.
REWARD_C_ASSUMED = Decimal("3")


def book_reward_scores(
    levels_bid, levels_ask, max_spread_c, min_size  # noqa: ANN001
) -> tuple[Decimal, Decimal, Decimal] | None:
    """
    Очки всей книги YES по формуле программы наград: Q_one (биды) и Q_two
    (аски) в пределах max_spread от mid, только уровни размером >= min_size.
    Mid — по лучшим уровням не мельче min_size («size-cutoff-adjusted
    midpoint» — наша интерпретация docs). Порядок уровней не важен.
    Возвращает (mid, q_one, q_two) или None, если после отсева с одной из
    сторон уровней нет — тогда очков в книге ни у кого.
    """
    min_size = Decimal(str(min_size))
    v = Decimal(str(max_spread_c))
    bids = [(Decimal(str(p)), Decimal(str(z))) for p, z in levels_bid
            if Decimal(str(z)) >= min_size]
    asks = [(Decimal(str(p)), Decimal(str(z))) for p, z in levels_ask
            if Decimal(str(z)) >= min_size]
    if not bids or not asks or v <= 0:
        return None
    best_bid = max(p for p, _ in bids)
    best_ask = min(p for p, _ in asks)
    mid = (best_bid + best_ask) / 2

    def score(distance_c: Decimal, size: Decimal) -> Decimal:
        distance_c = max(distance_c, Decimal("0"))
        if distance_c > v:
            return Decimal("0")
        return ((v - distance_c) / v) ** 2 * size

    q_one = sum((score((mid - p) * 100, z) for p, z in bids), Decimal("0"))
    q_two = sum((score((p - mid) * 100, z) for p, z in asks), Decimal("0"))
    return mid, q_one, q_two


async def print_book_competition(client: AsyncPublicClient, t) -> None:  # noqa: ANN001
    """
    Конкуренция за награды по живому стакану YES (наших ордеров в нём нет —
    diag ничего не выставляет). Сумму Q_min других мейкеров по книге точно
    не восстановить (Q_min считается на мейкера), но границы есть:
    max(Q_one, Q_two)/c <= sum Q_min <= Q_one + Q_two в диапазоне mid
    0.10–0.90. Это и есть число для --reward-competition в simulate.py.
    """
    if t.rewards_max_spread is None or t.rewards_min_size is None:
        print("    конкуренция: у рынка нет параметров наград — стакан не оцениваю")
        return
    try:
        book = await client.get_order_book(token_id=t.yes_token_id)
    except Exception as exc:  # noqa: BLE001
        print(f"    конкуренция: стакан не прочитан ({exc})")
        return
    scored = book_reward_scores(
        [(lvl.price, lvl.size) for lvl in book.bids],
        [(lvl.price, lvl.size) for lvl in book.asks],
        t.rewards_max_spread, t.rewards_min_size,
    )
    if scored is None:
        print("    конкуренция: в стакане YES нет уровней >= min_size с обеих сторон — "
              "очков сейчас ни у кого")
        return
    mid, q_one, q_two = scored
    lo = max(q_one, q_two) / REWARD_C_ASSUMED
    hi = q_one + q_two
    print(
        f"    стакан YES: mid(adj)={mid:.3f} | Q_one(биды)={q_one:.1f} "
        f"Q_two(аски)={q_two:.1f} очков в пределах {t.rewards_max_spread}¢ | "
        f"сумма Q_min других мейкеров ∈ [{lo:.1f}; {hi:.1f}] "
        f"(низ при c={REWARD_C_ASSUMED}) → --reward-competition для simulate.py"
    )


def load_strategy_settings() -> StrategySettings:
    """Настройки бота; без .env или при битом .env — дефолты."""
    try:
        return StrategySettings()
    except Exception as exc:  # noqa: BLE001
        print(f"! Конфиг не прочитан ({exc}) — использую дефолты кода.")
        return StrategySettings.model_construct()


async def section_exact_bot_queries(
    client: AsyncPublicClient, cfg: StrategySettings
) -> list:
    """Раздел 2: точные запросы бота по слагам серий (резервный путь)."""
    print("\n=== 2. Резервный путь бота: list_series(slug=...) =============")
    if not cfg.series_slugs:
        print("  (слаги серий не настроены — основной путь бота title_search, "
              "резерв отключён; это дефолт)")
        return []
    events: list = []
    for slug in cfg.series_slugs:
        try:
            n_series = 0
            n_events = 0
            n_markets = 0
            paginator = client.list_series(slug=slug, closed=False, page_size=5)
            async for series in paginator.iter_items():
                n_series += 1
                for ev in getattr(series, "events", None) or []:
                    n_events += 1
                    events.append(ev)
                    n_markets += len(getattr(ev, "markets", None) or [])
            status = "OK" if n_markets else "ПУСТО ДЛЯ БОТА"
            print(
                f"  slug='{slug}': серий={n_series} событий={n_events} "
                f"рынков в событиях={n_markets}  [{status}]"
            )
            if n_series == 0:
                print("    -> слаг не находит ни одной открытой серии: устарел или серия закрыта")
            elif n_markets == 0 and n_events == 0:
                print("    -> серия есть, но открытых событий в ней нет")
            elif n_markets == 0:
                print("    -> события есть, но БЕЗ встроенных рынков — путь через серии мёртв")
        except Exception as exc:  # noqa: BLE001
            print(f"  slug='{slug}': ЗАПРОС НЕ УДАЛСЯ: {exc}")
    return events


async def section_scan_series(
    client: AsyncPublicClient, max_series: int
) -> list:
    """Раздел 3: скан всех открытых серий, фильтр по маркерам."""
    print(f"\n=== 3. Скан открытых серий (до {max_series}), маркеры {MARKERS} ===")
    matched: list = []
    scanned = 0
    try:
        # Gamma принимает page_size не больше 50 для серий («page_size must
        # be at most 50»); дальше пагинатор сам идёт по страницам.
        paginator = client.list_series(closed=False, page_size=SERIES_PAGE_SIZE)
        async for series in paginator.iter_items():
            scanned += 1
            if matches_markers(getattr(series, "slug", None), getattr(series, "title", None)):
                matched.append(series)
            if scanned >= max_series:
                print(f"  (скан оборван на {max_series} сериях — есть ещё; --max-series поднимет предел)")
                break
    except Exception as exc:  # noqa: BLE001
        print(f"  СКАН НЕ УДАЛСЯ после {scanned} серий: {exc}")

    print(f"  просмотрено серий: {scanned}, совпало с маркерами: {len(matched)}")
    for s in matched:
        n_events = len(getattr(s, "events", None) or [])
        vol = getattr(s, "volume_24hr", None)
        vol_txt = f" vol24h={vol:.0f}" if isinstance(vol, Decimal) else ""
        print(
            f"  - slug='{getattr(s, 'slug', '?')}' | '{getattr(s, 'title', '?')}' | "
            f"recurrence={getattr(s, 'recurrence', None)} | "
            f"active={fmt_bool(getattr(s, 'active', None))} "
            f"closed={fmt_bool(getattr(s, 'closed', None))} | "
            f"событий={n_events}{vol_txt}"
        )
    if not matched:
        print("  ! Ни одна открытая серия не содержит маркеров — ищи по событиям (раздел 4)")
    return matched


async def section_title_probes(
    client: AsyncPublicClient, cfg: StrategySettings
) -> list:
    """Раздел 4: list_events по вариантам заголовка (ОСНОВНОЙ путь бота)."""
    print("\n=== 4. ОСНОВНОЙ путь: list_events(title_search=...) ============")
    probes: list[str] = []
    for probe in list(cfg.title_keywords) + list(EXTRA_TITLE_PROBES):
        if probe.lower() not in {p.lower() for p in probes}:
            probes.append(probe)

    events: list = []
    seen_event_ids: set[str] = set()
    for probe in probes:
        try:
            batch: list = []
            paginator = client.list_events(
                title_search=probe, closed=False, order="endDate",
                ascending=True, page_size=20,
            )
            async for ev in paginator.iter_items():
                batch.append(ev)
                if len(batch) >= 40:
                    break
            print(f"  '{probe}': событий={len(batch)}{'+' if len(batch) >= 40 else ''}")
            for ev in batch[:5]:
                end = getattr(getattr(ev, "schedule", None), "end_date", None)
                print(
                    f"      {getattr(ev, 'title', '?')} | slug={getattr(ev, 'slug', '?')} | "
                    f"конец={end} | рынков={len(getattr(ev, 'markets', None) or [])}"
                )
            for ev in batch:
                ev_id = str(getattr(ev, "id", id(ev)))
                if ev_id not in seen_event_ids:
                    seen_event_ids.add(ev_id)
                    events.append(ev)
        except Exception as exc:  # noqa: BLE001
            print(f"  '{probe}': ЗАПРОС НЕ УДАЛСЯ: {exc}")
    return events


def section_candidates(
    all_events: list, cfg: StrategySettings, horizon_hours: float
) -> int:
    """Раздел 5: сводная таблица рынков и вердикты фильтров бота."""
    print(f"\n=== 5. Рынки-кандидаты с экспирацией в ближайшие {horizon_hours:g}ч ===")
    now = time.time()
    rows: list[tuple[float, object, str]] = []
    seen: set[str] = set()
    skipped_far = 0
    for ev in all_events:
        for m in getattr(ev, "markets", None) or []:
            key = str(getattr(m, "condition_id", None) or getattr(m, "id", id(m)))
            if key in seen:
                continue
            seen.add(key)
            end_ts = _parse_ts(getattr(getattr(m, "state", None), "end_date", None))
            # Рынки без даты НЕ прячем: отсутствие даты — само по себе диагноз.
            if end_ts is not None and end_ts - now > horizon_hours * 3600:
                skipped_far += 1
                continue
            left = float("inf") if end_ts is None else end_ts - now
            rows.append((left, m, str(getattr(ev, "title", "") or "")))

    rows.sort(key=lambda r: r[0])
    takeable = 0
    for left, m, ev_title in rows[:40]:
        takeable += print_market(m, ev_title, now, cfg)
    if len(rows) > 40:
        print(f"  ... и ещё {len(rows) - 40} рынков (обрезано)")
    if skipped_far:
        print(f"  (не показано {skipped_far} рынков с экспирацией дальше горизонта)")
    if not rows:
        print("  Рынков в горизонте нет вовсе.")
    print(f"  Итого в горизонте: {len(rows)}, из них бот взял бы прямо сейчас: {takeable}")
    return takeable


async def section_parity(client: AsyncPublicClient, cfg: StrategySettings) -> list:
    """Раздел 6: настоящий MarketDiscovery с настройками бота. Возвращает взятые рынки."""
    print("\n=== 6. Паритет: MarketDiscovery.find_markets() как в боте =======")
    discovery = MarketDiscovery(
        client,  # type: ignore[arg-type] — нужные методы у публичного клиента те же
        series_slugs=list(cfg.series_slugs),
        title_keywords=list(cfg.title_keywords),
        min_seconds=cfg.min_seconds_to_expiry,
        max_seconds=cfg.max_seconds_to_expiry,
        max_markets=cfg.max_concurrent_markets,
        fallback_fee_rate=cfg.fallback_fee_rate,
    )
    try:
        # Спота здесь нет — страйк не калибруется, это нормально для диага.
        targets = await discovery.find_markets({})
    except Exception as exc:  # noqa: BLE001
        print(f"  find_markets УПАЛ: {exc}")
        return []
    print(f"  воронка: {discovery.last_funnel.describe()}")
    for t in targets:
        if t.rewards_daily_rate is None:
            rewards_txt = "нет"
        else:
            rewards_txt = (f"{t.rewards_daily_rate}/день до "
                           f"{t.rewards_end_date or 'бессрочно'}")
        print(
            f"  ВЗЯТ: {t.slug} | {t.asset} | до экспирации {t.seconds_left:.0f}с | "
            f"tick={t.tick_size} | комиссия={'есть' if t.fees_enabled else 'нет'} | "
            f"окно TWAP={t.twap_window_s}s | награды: {rewards_txt} | "
            f"reward_max_spread={t.rewards_max_spread} min_size={t.rewards_min_size}"
        )
        await print_book_competition(client, t)
    if not targets:
        print("  Бот не взял ни одного рынка — причина видна в воронке выше.")
    return targets


def _fmt_reward_config(cfg) -> str:  # noqa: ANN001
    start = getattr(cfg, "start_date", None)
    end = getattr(cfg, "end_date", None)
    fmt = lambda d: d.strftime("%Y-%m-%d") if d is not None else "бессрочно"  # noqa: E731
    return (f"{getattr(cfg, 'rate_per_day', '?')}/день "
            f"({fmt(start)} → {fmt(end)}, всего {getattr(cfg, 'total_rewards', '?')})")


async def section_rewards(client: AsyncPublicClient, targets: list) -> dict:
    """
    Раздел 7: активные программы наград ПО CLOB, а не по Gamma.

    Gamma отдаёт clobRewards в карточке рынка, и вживую он оказался пуст у
    всех взятых рынков. У CLOB есть свой источник истины — эндпоинты
    /rewards/markets/current (все рынки с активной программой) и
    /rewards/markets/{condition_id} (конфиг конкретного рынка, включая
    market_competitiveness). Если и они пусты — программы для этих рынков
    нет, и экономика бота — только торговый PnL (README).
    Возвращает сводку {condition_id: ставка/день по CLOB или None}.
    """
    print("\n=== 7. Награды по CLOB: /rewards/markets/current, /rewards/markets/{cid} ===")
    verdict: dict[str, object] = {}
    active: dict[str, object] = {}
    n_active = 0
    try:
        async for reward in client.list_current_rewards().iter_items():
            n_active += 1
            active[str(getattr(reward, "condition_id", ""))] = reward
            if n_active >= 5000:
                print("  (список активных программ оборван на 5000)")
                break
        print(f"  рынков с активной программой на CLOB (всего по бирже): {n_active}")
    except Exception as exc:  # noqa: BLE001
        print(f"  list_current_rewards НЕ УДАЛСЯ: {exc}")

    if not targets:
        print("  взятых рынков нет — сверять не с чем")
        return verdict

    for t in targets:
        gamma_rate = t.rewards_daily_rate
        current = active.get(str(t.condition_id))
        configs: list = []
        competitiveness = None
        try:
            async for m in client.list_market_rewards(condition_id=t.condition_id).iter_items():
                configs.extend(getattr(m, "rewards_config", None) or [])
                competitiveness = getattr(m, "market_competitiveness", competitiveness)
        except Exception as exc:  # noqa: BLE001
            print(f"  {t.slug}: list_market_rewards НЕ УДАЛСЯ: {exc}")

        clob_rate = None
        if current is not None:
            clob_rate = getattr(current, "total_daily_rate", None)
            if clob_rate is None:
                clob_rate = sum(
                    (getattr(c, "rate_per_day", Decimal("0")) for c in
                     getattr(current, "rewards_config", None) or []),
                    Decimal("0"),
                ) or None
        if clob_rate is None and configs:
            clob_rate = sum((getattr(c, "rate_per_day", Decimal("0")) for c in configs),
                            Decimal("0")) or None
        verdict[str(t.condition_id)] = clob_rate

        print(f"  {t.slug}:")
        print(f"    Gamma clobRewards: {gamma_rate if gamma_rate is not None else 'пусто'}")
        if current is None:
            print("    CLOB /rewards/markets/current: рынка НЕТ в списке активных программ")
        else:
            print(
                f"    CLOB current: total={getattr(current, 'total_daily_rate', None)} "
                f"native={getattr(current, 'native_daily_rate', None)} "
                f"sponsored={getattr(current, 'sponsored_daily_rate', None)} "
                f"(спонсоров {getattr(current, 'sponsors_count', None)}) | "
                f"max_spread={getattr(current, 'rewards_max_spread', None)}¢ "
                f"min_size={getattr(current, 'rewards_min_size', None)}"
            )
            for cfg in getattr(current, "rewards_config", None) or []:
                print(f"      конфиг: {_fmt_reward_config(cfg)}")
        if configs:
            print(f"    CLOB /rewards/markets/{{cid}}: competitiveness={competitiveness}")
            for cfg in configs:
                print(f"      конфиг: {_fmt_reward_config(cfg)}")
        else:
            print("    CLOB /rewards/markets/{cid}: конфигов наград нет")

    with_clob = [cid for cid, rate in verdict.items() if rate]
    with_gamma = [t for t in targets if t.rewards_daily_rate]
    if not with_clob and not with_gamma:
        print(
            "  ВЕРДИКТ: программа наград для взятых рынков НЕ АКТИВНА ни по Gamma, "
            "ни по CLOB. Экономика бота — только торговый PnL: раздел README про "
            "награды описывает то, чего сейчас нет."
        )
    elif with_clob and not with_gamma:
        print(
            "  ВЕРДИКТ: CLOB отдаёт ставки, Gamma — нет: discovery читает Gamma и "
            "оставляет TargetMarket.rewards_daily_rate пустым. Ставки для "
            "simulate.py --reward-daily-rate брать из строк CLOB выше."
        )
    else:
        print("  ВЕРДИКТ: программа активна — ставки выше, подставляй в simulate.py.")
    return verdict


async def run(args: argparse.Namespace) -> int:
    cfg = load_strategy_settings()

    print("=== 1. Настройки поиска бота ===================================")
    print(f"  series_slugs:   {cfg.series_slugs}")
    print(f"  title_keywords: {cfg.title_keywords}")
    print(
        f"  окно торговли:  от {cfg.min_seconds_to_expiry}с "
        f"до {cfg.max_seconds_to_expiry}с до экспирации"
    )

    async with AsyncPublicClient() as client:
        bot_events = await section_exact_bot_queries(client, cfg)
        matched_series = await section_scan_series(client, args.max_series)
        probe_events = await section_title_probes(client, cfg)

        series_events: list = []
        for s in matched_series:
            series_events.extend(getattr(s, "events", None) or [])

        takeable = section_candidates(
            bot_events + series_events + probe_events, cfg, args.horizon_hours
        )
        targets = await section_parity(client, cfg)
        parity = len(targets)
        await section_rewards(client, targets)

    print("\n=== ИТОГ ========================================================")
    if parity > 0:
        print(f"  Бот видит {parity} рынков со своими текущими настройками — поиск работает.")
        return 0
    if takeable > 0:
        print(
            "  Подходящие рынки НА БИРЖЕ ЕСТЬ, но настройки бота их не находят: "
            "возьми правильные слаги/заголовки из разделов 3-4 и пропиши в "
            "STRAT_SERIES_SLUGS / STRAT_TITLE_KEYWORDS."
        )
        return 2
    print(
        "  Подходящих рынков не нашлось ни по настройкам бота, ни по маркерам. "
        "Смотри разделы 3-5: либо серии переименованы сильнее (поправь MARKERS), "
        "либо сейчас нет открытого окна в заданном горизонте."
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Диагностика поиска рынков Polymarket")
    parser.add_argument("--max-series", type=int, default=2000,
                        help="потолок скана открытых серий (раздел 3)")
    parser.add_argument("--horizon-hours", type=float, default=24.0,
                        help="горизонт таблицы рынков-кандидатов (раздел 5)")
    args = parser.parse_args()

    # Тёплые логи discovery (WARNING о слагах и т.п.) — в stdout, к месту.
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="  [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    sys.exit(main())
