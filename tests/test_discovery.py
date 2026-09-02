"""
Тесты поиска рынков на заглушке, повторяющей ФОРМУ SDK буквально.

Дважды один и тот же класс бага («форма вызова — тоже контракт»): сначала
неawaited subscribe(), теперь итерация пагинатора. `async for x in list_*()`
отдаёт СТРАНИЦЫ (Page), а не элементы; у Page нет полей события или рынка,
и защитные getattr(..., None) or [] молча превращали весь ответ API в ноль
кандидатов — при любых слагах. Здесь пагинатор НАСТОЯЩИЙ (из SDK), рынки —
настоящие gamma-модели из плоских словарей, как их отдаёт API. Возврат к
итерации без .iter_items() роняет эти тесты нулём найденных рынков.

Сеть и ключи не нужны; пакет polymarket нужен для моделей и пагинатора.
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("polymarket", reason="нужны gamma-модели и пагинатор SDK")

from polymarket.models.gamma.event import Event  # noqa: E402
from polymarket.models.gamma.series import Series  # noqa: E402
from polymarket.pagination import AsyncPaginator, Page  # noqa: E402

from src.discovery import MarketDiscovery  # noqa: E402

D = Decimal


def iso_in(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def market_dict(n: int = 1, **over) -> dict:
    """Плоский словарь рынка — та же форма, что в JSON Gamma API."""
    base = {
        "id": str(n),
        "slug": f"bitcoin-up-or-down-window-{n}",
        "conditionId": "0x" + f"{n:02x}" * 32,
        "question": "Bitcoin Up or Down?",
        "active": True, "closed": False, "archived": False,
        "acceptingOrders": True, "enableOrderBook": True,
        "endDate": iso_in(300),
        "outcomes": '["Up","Down"]',
        "clobTokenIds": f'["{n}11","{n}22"]',
        "orderPriceMinTickSize": "0.01",
        "orderMinSize": "5",
        # Ссылка на поток резолюции — как в описании живых рынков.
        "description": ("Resolves per Chainlink TWAP: "
                        "https://data.chain.link/streams/btc-usd-twap-60s-streams"),
    }
    base.update(over)
    return base


def gamma_event(markets: list[dict], **over) -> Event:
    base = {
        "id": "10", "slug": "bitcoin-up-or-down-sep-1",
        "title": "Bitcoin Up or Down", "markets": markets,
    }
    base.update(over)
    return Event.model_validate(base)


def gamma_series(events: list[Event], **over) -> Series:
    base = {
        "id": "5", "slug": "bitcoin-up-or-down",
        "title": "Bitcoin Up or Down", "recurrence": "hourly",
    }
    base.update(over)
    series = Series.model_validate(base)
    # События подставляем уже валидированными: у Series поле опционально.
    return series.model_copy(update={"events": tuple(events)})


def paginate(
    items: list, page_size: int = 1, fetch_log: list | None = None
) -> AsyncPaginator:
    """
    НАСТОЯЩИЙ пагинатор SDK поверх Page, по одному элементу на страницу:
    и забытый .iter_items(), и чтение только первой страницы — провал.
    fetch_log считает обращения к API — им проверяется ранняя остановка.
    """

    async def fetch(cursor: str | None) -> Page:
        if fetch_log is not None:
            fetch_log.append(cursor)
        idx = int(cursor or 0)
        has_more = idx + page_size < len(items)
        return Page(
            items=tuple(items[idx:idx + page_size]),
            has_more=has_more,
            next_cursor=str(idx + page_size) if has_more else None,
        )

    return AsyncPaginator(fetch)


class FakeGammaClient:
    """list_series/list_events как у SDK: пагинаторы страниц, без await."""

    def __init__(
        self,
        series: dict[str, list[Series]] | None = None,
        events: list[Event] | None = None,
        series_error: Exception | None = None,
        events_error: Exception | None = None,
        rewards_current: list | None = None,
        rewards_market: dict[str, list] | None = None,
    ) -> None:
        self._series = series or {}
        self._events = events or []
        self._series_error = series_error
        self._events_error = events_error
        self._rewards_current = rewards_current or []
        self._rewards_market = rewards_market or {}
        self.series_queries: list[str] = []
        self.series_page_sizes: list[int] = []
        self.event_queries: list[str] = []
        self.event_fetches: list = []      # обращения к страницам list_events

    def list_series(
        self, *, slug: str | None = None, closed: bool, page_size: int
    ) -> AsyncPaginator:
        self.series_queries.append(slug or "*")
        self.series_page_sizes.append(page_size)
        if page_size > 50:
            # Как Gamma/SDK вживую: «page_size must be at most 50».
            raise ValueError("page_size must be at most 50")
        if self._series_error is not None:
            raise self._series_error
        if slug is None:
            # Скан всех открытых серий (так делает diag.py, раздел 3).
            all_series = [s for group in self._series.values() for s in group]
            return paginate(all_series)
        return paginate(self._series.get(slug, []))

    def list_events(self, *, title_search: str, **kw) -> AsyncPaginator:
        self.event_queries.append(title_search)
        if self._events_error is not None:
            raise self._events_error
        return paginate(self._events, fetch_log=self.event_fetches)

    # Награды по CLOB — тоже пагинаторы (публичный клиент SDK).
    def list_current_rewards(self, *, sponsored: bool | None = None) -> AsyncPaginator:
        return paginate(self._rewards_current)

    def list_market_rewards(
        self, *, condition_id: str, sponsored: bool | None = None
    ) -> AsyncPaginator:
        return paginate(self._rewards_market.get(condition_id, []))


def make_discovery(client: FakeGammaClient, **over) -> MarketDiscovery:
    kw = dict(
        series_slugs=["bitcoin-up-or-down", "ethereum-up-or-down"],
        title_keywords=["Bitcoin Up or Down"],
        min_seconds=90, max_seconds=900, max_markets=4,
    )
    kw.update(over)
    return MarketDiscovery(client, **kw)  # type: ignore[arg-type]


def find(disc: MarketDiscovery, spots: dict[str, float] | None = None):
    return asyncio.run(disc.find_markets(spots or {}))


# ------------------------------------------------------------------- поиск


def test_search_is_primary_and_extracts_markets_through_pages():
    """
    ГЛАВНЫЙ путь — title_search (на живом API только он находит updown-
    рынки): рынок доезжает из event -> target, серии не опрашиваются вовсе.
    """
    ev = gamma_event([market_dict(1)])
    client = FakeGammaClient(series={}, events=[ev])
    disc = make_discovery(client)

    found = find(disc)

    assert len(found) == 1
    t = found[0]
    assert (t.yes_token_id, t.no_token_id) == ("111", "122")
    assert t.asset == "BTC"
    assert 250 <= t.seconds_left <= 300
    assert client.event_queries == ["Bitcoin Up or Down"]
    assert client.series_queries == []          # резерв не понадобился
    assert disc.last_funnel.search_events == 1
    assert disc.last_funnel.series_events == 0
    assert disc.last_funnel.accepted == 1


def test_series_are_fallback_when_search_is_empty():
    """Поиск пуст -> резервный путь через серии, тоже постранично."""
    series = gamma_series([gamma_event([market_dict(2)])])
    client = FakeGammaClient(series={"bitcoin-up-or-down": [series]}, events=[])
    disc = make_discovery(client)

    found = find(disc)

    assert len(found) == 1
    # Сначала спросили поиск (по каждому ключу), потом серии.
    assert client.event_queries == ["Bitcoin Up or Down"]
    assert client.series_queries == ["bitcoin-up-or-down", "ethereum-up-or-down"]
    assert disc.last_funnel.search_events == 0
    assert disc.last_funnel.series_events == 1


def test_empty_series_slugs_skip_fallback_silently(caplog):
    """Пустые series_slugs (дефолт) — ни запросов, ни WARNING про серии."""
    client = FakeGammaClient(series={}, events=[])
    disc = make_discovery(client, series_slugs=[])

    with caplog.at_level(logging.WARNING, logger="polybot.discovery"):
        found = find(disc)

    assert found == []
    assert client.series_queries == []
    assert not any("Серия" in r.message for r in caplog.records)
    # Итоговое предупреждение о пустом результате — есть, и без серий.
    summary = [r.message for r in caplog.records if "не найдено" in r.message]
    assert summary and "серии" not in summary[0]


def test_search_stops_early_by_end_date():
    """
    Выборка поиска отсортирована по endDate: событие за горизонтом окна
    обрывает пагинацию. Без этого бот сливал ~900 событий каждый цикл.
    """
    near1 = gamma_event([market_dict(1)], id="1", endDate=iso_in(300))
    near2 = gamma_event([market_dict(2, endDate=iso_in(400))], id="2",
                        endDate=iso_in(400))
    far = [
        gamma_event([market_dict(3 + i, endDate=iso_in(7200))], id=str(3 + i),
                    endDate=iso_in(7200 + i))
        for i in range(4)
    ]
    client = FakeGammaClient(events=[near1, near2, *far])
    disc = make_discovery(client, title_keywords=["Bitcoin Up or Down"])

    found = find(disc)

    # Два близких события взяты, дальняя часть выборки не выкачивалась:
    # три обращения к API (near1, near2, первое дальнее — на нём стоп).
    assert disc.last_funnel.search_events == 2
    assert len(client.event_fetches) == 3
    assert len(found) == 2


def test_multipage_series_are_not_truncated():
    """Кандидаты со второй страницы не теряются (page_size=1 в заглушке)."""
    s1 = gamma_series([gamma_event([market_dict(1)])])
    s2 = gamma_series(
        [gamma_event([market_dict(2, slug="ethereum-up-or-down-w",
                                  question="Ethereum Up or Down?")],
                     title="Ethereum Up or Down")],
        slug="ethereum-up-or-down", title="Ethereum Up or Down",
    )
    client = FakeGammaClient(series={
        "bitcoin-up-or-down": [s1, s2],   # две серии = две страницы
    })
    disc = make_discovery(client)

    found = find(disc)
    assert {t.asset for t in found} == {"BTC", "ETH"}


# ------------------------------------------------------------------ воронка


def test_funnel_attributes_every_drop_to_its_filter():
    markets = [
        market_dict(1),                                        # пройдёт
        market_dict(2, closed=True),                           # state
        market_dict(3, acceptingOrders=False),                 # приём ордеров
        market_dict(4, enableOrderBook=False),                 # стакан
        market_dict(5, endDate=None),                          # нет даты
        market_dict(6, endDate=iso_in(3600)),                  # окно: рано
        market_dict(7, endDate=iso_in(30)),                    # окно: поздно
        market_dict(8, clobTokenIds="[]"),                     # нет токенов
        market_dict(9, slug="solana-up-or-down-w",
                    question="Solana Up or Down?"),            # актив
        market_dict(1),                                        # дубль
    ]
    ev = gamma_event([m for m in markets], title="Crypto Up or Down")
    client = FakeGammaClient(series={"bitcoin-up-or-down": [gamma_series([ev])]})
    disc = make_discovery(client)

    found = find(disc)
    f = disc.last_funnel

    assert len(found) == 1
    assert f.markets_in == 10
    assert f.drop_state == 1
    assert f.drop_accepting == 1
    assert f.drop_book == 1
    assert f.drop_no_end == 1
    assert f.drop_window_far == 1
    assert f.drop_window_near == 1
    assert f.drop_no_tokens == 1
    assert f.drop_no_asset == 1
    assert f.drop_dup == 1
    assert f.accepted == 1
    # Сумма раскладки обязана сходиться с числом пришедших рынков.
    dropped = (f.drop_state + f.drop_accepting + f.drop_book + f.drop_no_end
               + f.drop_window_far + f.drop_window_near + f.drop_no_tokens
               + f.drop_no_asset + f.drop_dup + f.drop_overflow)
    assert dropped + f.accepted == f.markets_in
    # Строка для лога несёт все счётчики.
    line = f.describe()
    assert "рынков=10" in line and "взято=1" in line and "окно_рано=1" in line


def test_max_markets_overflow_is_counted_not_silent():
    ms = [market_dict(1), market_dict(2, endDate=iso_in(400))]
    ev = gamma_event(ms)
    client = FakeGammaClient(series={"bitcoin-up-or-down": [gamma_series([ev])]})
    disc = make_discovery(client, max_markets=1)

    found = find(disc)
    assert len(found) == 1
    assert disc.last_funnel.drop_overflow == 1
    # Берём ближайший к экспирации.
    assert found[0].seconds_left <= 310


# ------------------------------------------------------------------- логи


def test_search_api_error_warns_and_falls_back_to_series(caplog):
    """Сломанный поиск обязан кричать WARNING, а резерв через серии — спасать."""
    series = gamma_series([gamma_event([market_dict(3)])])
    client = FakeGammaClient(
        series={"bitcoin-up-or-down": [series]},
        events_error=RuntimeError("HTTP 500"),
    )
    disc = make_discovery(client)

    with caplog.at_level(logging.WARNING, logger="polybot.discovery"):
        found = find(disc)

    assert len(found) == 1  # резервный путь спас
    warned = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Поиск по заголовку" in m and "HTTP 500" in m for m in warned)


def test_series_api_error_warns(caplog):
    """И сломанный резерв тоже кричит WARNING, а не шепчет debug."""
    client = FakeGammaClient(series_error=RuntimeError("HTTP 502"), events=[])
    disc = make_discovery(client)

    with caplog.at_level(logging.WARNING, logger="polybot.discovery"):
        found = find(disc)

    assert found == []
    assert any(
        "недоступна" in r.message and "HTTP 502" in r.message
        for r in caplog.records if r.levelno == logging.WARNING
    )


def test_stale_slug_warns_even_without_exception(caplog):
    """API отвечает, но серий ноль — это «слаг устарел», и это WARNING."""
    client = FakeGammaClient(series={}, events=[])
    disc = make_discovery(client)

    with caplog.at_level(logging.WARNING, logger="polybot.discovery"):
        found = find(disc)

    assert found == []
    warned = " | ".join(
        r.message for r in caplog.records if r.levelno == logging.WARNING
    )
    assert "ноль открытых серий" in warned
    assert "не найдено" in warned


def test_search_api_error_warns(caplog):
    client = FakeGammaClient(series={}, events_error=RuntimeError("timeout"))
    disc = make_discovery(client)

    with caplog.at_level(logging.WARNING, logger="polybot.discovery"):
        found = find(disc)

    assert found == []
    assert any(
        "Поиск по заголовку" in r.message and "timeout" in r.message
        for r in caplog.records if r.levelno == logging.WARNING
    )


# ------------------------------------------------------------------ diag.py


def _fake_public(monkeypatch, fake: FakeGammaClient) -> None:
    import diag

    class FakePublicClient:
        async def __aenter__(self):
            return fake

        async def __aexit__(self, *exc) -> None:
            return None

    monkeypatch.setattr(diag, "AsyncPublicClient", FakePublicClient)


def test_diag_smoke_offline(monkeypatch, capsys):
    """
    diag.py на том же фейке: скрипт проходит все разделы без сети, печатает
    token_id обеих ног, вердикты фильтров и воронку, и выходит с кодом 0,
    когда бот взял бы рынок. Ордеров он не шлёт — у клиента их просто нет.
    """
    import argparse

    import diag

    fake = FakeGammaClient(
        series={}, events=[gamma_event([market_dict(1, slug="btc-updown-5m-1788276000")])]
    )
    _fake_public(monkeypatch, fake)

    code = asyncio.run(diag.run(argparse.Namespace(max_series=50, horizon_hours=24.0)))
    out = capsys.readouterr().out

    assert code == 0
    assert "111" in out and "122" in out          # token_id обеих ног
    assert "accepting_orders: да" in out
    assert "enable_order_book: да" in out
    assert "БОТ ВЗЯЛ БЫ" in out
    assert "воронка" in out
    assert "btc-updown-5m-1788276000" in out


def test_diag_exit_code_2_when_bot_settings_find_nothing(monkeypatch, capsys):
    """Рынки на бирже есть (в TWAP-серии), но поиск бота их не видит."""
    import argparse

    import diag

    series = gamma_series(
        [gamma_event([market_dict(2, slug="bitcoin-up-or-down-twap-1")],
                     title="Bitcoin Up or Down TWAP")],
        slug="bitcoin-up-or-down-twap", title="Bitcoin Up or Down TWAP",
    )
    fake = FakeGammaClient(series={"bitcoin-up-or-down-twap": [series]}, events=[])
    _fake_public(monkeypatch, fake)

    code = asyncio.run(diag.run(argparse.Namespace(max_series=50, horizon_hours=24.0)))
    out = capsys.readouterr().out

    assert code == 2
    # Скан по маркерам актуальный слаг всё равно нашёл и показал.
    assert "bitcoin-up-or-down-twap" in out
    assert "слаги" in out.lower()


def test_twap_window_parsing_description_then_slug_duration():
    """
    Окно ряда резолюции: ссылка в описании — истина; иначе по длительности
    из слага (5m -> 30s, 15m/4h -> 60s по анонсу); иначе None.
    """
    from src.discovery import parse_twap_window_s

    link30 = "https://data.chain.link/streams/btc-usd-twap-30s-streams"
    link60 = "https://data.chain.link/streams/eth-usd-twap-60s-streams"
    assert parse_twap_window_s(link30, "btc-updown-15m-1788276000") == 30  # описание важнее
    assert parse_twap_window_s(link60, "btc-updown-5m-1788276000") == 60
    assert parse_twap_window_s("", "btc-updown-5m-1788276000") == 30
    assert parse_twap_window_s("", "eth-updown-15m-1788276000") == 60
    assert parse_twap_window_s("", "btc-updown-4h-1788276000") == 60
    assert parse_twap_window_s("", "bitcoin-up-or-down-3pm-et") is None
    assert parse_twap_window_s("twap-45s", "btc-updown-5m-1788276000") is None  # чужое окно


def test_market_without_twap_window_is_dropped_and_counted():
    """Не знаем ряд резолюции — не торгуем (не кормим модель чужим окном)."""
    ev = gamma_event([
        market_dict(1, description=""),                              # нет ни ссылки, ни длительности
        market_dict(2, slug="btc-updown-5m-1788276000", description=""),  # по слагу -> 30s
    ])
    client = FakeGammaClient(events=[ev])
    disc = make_discovery(client)

    found = find(disc)

    assert [t.twap_window_s for t in found] == [30]
    assert disc.last_funnel.drop_no_twap_window == 1


def test_reward_program_fields_flow_into_target():
    ev = gamma_event([market_dict(
        1, rewardsMinSize="20", rewardsMaxSpread="3.5",
        clobRewards=[{"id": "77", "conditionId": "0x" + "01" * 32,
                      "assetAddress": "0xusdc", "rewardsAmount": "0",
                      "rewardsDailyRate": "150", "startDate": "2026-08-07",
                      "endDate": "2099-12-31"}],
    )])
    client = FakeGammaClient(events=[ev])
    found = find(make_discovery(client))

    assert len(found) == 1
    t = found[0]
    assert t.rewards_max_spread == D("3.5")
    assert t.rewards_min_size == D("20")
    assert t.rewards_daily_rate == D("150")
    assert t.rewards_end_date == "2099-12-31"

    # Истёкшая программа не считается действующей.
    ev_old = gamma_event([market_dict(
        3, clobRewards=[{"id": "78", "conditionId": "0x" + "03" * 32,
                         "assetAddress": "0xusdc", "rewardsAmount": "0",
                         "rewardsDailyRate": "150", "startDate": "2026-08-07",
                         "endDate": "2026-08-31"}],
    )])
    found_old = find(make_discovery(FakeGammaClient(events=[ev_old])))
    assert found_old[0].rewards_daily_rate is None


def test_slug_start_ts_parsing():
    """Хвост слага btc-updown-5m-<ts> — unix-время начала окна."""
    from src.discovery import parse_slug_start_ts

    assert parse_slug_start_ts("btc-updown-5m-1788276000") == 1788276000.0
    assert parse_slug_start_ts("eth-updown-15m-1788277500") == 1788277500.0
    assert parse_slug_start_ts("bitcoin-up-or-down-3pm-et") is None
    assert parse_slug_start_ts("market-1234") is None          # не unix-время
    assert parse_slug_start_ts("") is None

    # И конец цепочки: рынок из поиска несёт start_ts в TargetMarket.
    ev = gamma_event([market_dict(1, slug="btc-updown-5m-1788276000")])
    client = FakeGammaClient(events=[ev])
    disc = make_discovery(client)
    found = find(disc)
    assert found and found[0].start_ts == 1788276000.0


def test_book_reward_scores_filters_dust_and_spread():
    """
    diag.py: очки книги по формуле наград — пыль мельче min_size не входит
    ни в mid, ни в очки; уровни дальше max_spread дают ноль; порядок
    уровней не важен; пустая после отсева сторона — None (очков ни у кого).
    """
    from diag import book_reward_scores

    bids = [(D("0.49"), D("5")), (D("0.48"), D("100")), (D("0.40"), D("200"))]
    asks = [(D("0.51"), D("5")), (D("0.52"), D("50"))]
    scored = book_reward_scores(bids, asks, D("3.5"), D("20"))
    assert scored is not None
    mid, q_one, q_two = scored
    assert mid == D("0.50")  # пыль 0.49/0.51 не сдвинула mid
    weight = ((D("3.5") - 2) / D("3.5")) ** 2  # 2¢ от mid
    assert float(q_one) == pytest.approx(float(weight * 100))  # бид 0.40 (10¢) — ноль
    assert float(q_two) == pytest.approx(float(weight * 50))
    assert book_reward_scores(list(reversed(bids)), list(reversed(asks)),
                              D("3.5"), D("20")) == scored
    assert book_reward_scores(bids, [(D("0.52"), D("10"))], D("3.5"), D("20")) is None


def test_diag_series_scan_uses_gamma_page_limit(monkeypatch, capsys):
    """Скан серий: page_size не больше 50 — иначе Gamma отвечает ошибкой."""
    import argparse

    import diag

    fake = FakeGammaClient(series={}, events=[])
    _fake_public(monkeypatch, fake)
    asyncio.run(diag.run(argparse.Namespace(max_series=50, horizon_hours=24.0)))
    out = capsys.readouterr().out
    assert "*" in fake.series_queries
    assert all(size <= 50 for size in fake.series_page_sizes)
    assert "СКАН НЕ УДАЛСЯ" not in out


def test_diag_rewards_section_reads_clob_not_only_gamma(monkeypatch, capsys):
    """
    Раздел 7: Gamma clobRewards пуст, но CLOB /rewards/markets/current и
    /rewards/markets/{cid} отдают ставку — diag печатает её и вердикт
    «CLOB отдаёт ставки, Gamma — нет». Без CLOB-ставок — вердикт «программа
    НЕ АКТИВНА». Модели — настоящие SDK-модели наград.
    """
    import argparse

    import diag
    from polymarket.models.clob.rewards import CurrentReward, MarketReward

    cid = "0x" + "ab" * 32
    current = CurrentReward.model_validate({
        "condition_id": cid, "rewards_max_spread": 3.5, "rewards_min_size": "20",
        "rewards_config": [{"asset_address": "0xusdc", "start_date": 1756684800000,
                            "end_date": None, "rate_per_day": "33.6"}],
        "native_daily_rate": "33.6", "total_daily_rate": "33.6", "sponsors_count": 0,
    })
    market_reward = MarketReward.model_validate({
        "condition_id": cid, "question": "Bitcoin Up or Down?",
        "market_slug": "btc-updown-5m-1788276000", "rewards_max_spread": 3.5,
        "rewards_min_size": "20", "market_competitiveness": 1.7,
        "tokens": [{"token_id": "111", "outcome": "Up", "price": "0.51"}],
        "rewards_config": [{"asset_address": "0xusdc", "start_date": 1756684800000,
                            "end_date": None, "rate_per_day": "33.6"}],
    })
    ev = gamma_event([market_dict(1, slug="btc-updown-5m-1788276000", conditionId=cid)])
    fake = FakeGammaClient(
        series={}, events=[ev],
        rewards_current=[current], rewards_market={cid: [market_reward]},
    )
    _fake_public(monkeypatch, fake)
    code = asyncio.run(diag.run(argparse.Namespace(max_series=50, horizon_hours=24.0)))
    out = capsys.readouterr().out
    assert code == 0
    assert "рынков с активной программой на CLOB (всего по бирже): 1" in out
    assert "33.6/день" in out
    assert "competitiveness=1.7" in out
    assert "CLOB отдаёт ставки, Gamma — нет" in out

    # Без ставок ни в одном источнике — программа неактивна, и это сказано прямо.
    fake = FakeGammaClient(series={}, events=[ev])
    _fake_public(monkeypatch, fake)
    asyncio.run(diag.run(argparse.Namespace(max_series=50, horizon_hours=24.0)))
    out = capsys.readouterr().out
    assert "рынка НЕТ в списке активных программ" in out
    assert "НЕ АКТИВНА ни по Gamma, ни по CLOB" in out
