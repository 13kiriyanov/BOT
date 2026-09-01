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
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("polymarket", reason="нужны gamma-модели и пагинатор SDK")

from polymarket.models.gamma.event import Event  # noqa: E402
from polymarket.models.gamma.series import Series  # noqa: E402
from polymarket.pagination import AsyncPaginator, Page  # noqa: E402

from src.discovery import MarketDiscovery  # noqa: E402


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


def paginate(items: list, page_size: int = 1) -> AsyncPaginator:
    """
    НАСТОЯЩИЙ пагинатор SDK поверх Page, по одному элементу на страницу:
    и забытый .iter_items(), и чтение только первой страницы — провал.
    """

    async def fetch(cursor: str | None) -> Page:
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
    ) -> None:
        self._series = series or {}
        self._events = events or []
        self._series_error = series_error
        self._events_error = events_error
        self.series_queries: list[str] = []
        self.event_queries: list[str] = []

    def list_series(
        self, *, slug: str | None = None, closed: bool, page_size: int
    ) -> AsyncPaginator:
        self.series_queries.append(slug or "*")
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
        return paginate(self._events)


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


def test_series_path_extracts_markets_through_pages():
    """ГЛАВНЫЙ тест бага: рынок доезжает из series -> event -> target."""
    series = gamma_series([gamma_event([market_dict(1)])])
    client = FakeGammaClient(series={"bitcoin-up-or-down": [series]})
    disc = make_discovery(client)

    found = find(disc)

    assert len(found) == 1
    t = found[0]
    assert (t.yes_token_id, t.no_token_id) == ("111", "122")
    assert t.asset == "BTC"
    assert 250 <= t.seconds_left <= 300
    assert disc.last_funnel.series_events == 1
    assert disc.last_funnel.accepted == 1
    # Поиск по заголовку не понадобился.
    assert client.event_queries == []


def test_search_fallback_extracts_markets_through_pages():
    """Серии пусты -> резервный полнотекстовый путь, тоже постранично."""
    ev = gamma_event([market_dict(2)])
    client = FakeGammaClient(series={}, events=[ev])
    disc = make_discovery(client)

    found = find(disc)

    assert len(found) == 1
    assert client.event_queries == ["Bitcoin Up or Down"]
    assert disc.last_funnel.series_events == 0
    assert disc.last_funnel.search_events == 1


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


def test_series_api_error_warns_and_falls_back_to_search(caplog):
    """Сломанный поиск обязан кричать WARNING, а не шептать debug."""
    ev = gamma_event([market_dict(3)])
    client = FakeGammaClient(series_error=RuntimeError("HTTP 500"), events=[ev])
    disc = make_discovery(client)

    with caplog.at_level(logging.WARNING, logger="polybot.discovery"):
        found = find(disc)

    assert len(found) == 1  # резервный путь спас
    warned = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("недоступна" in m and "HTTP 500" in m for m in warned)


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


def test_diag_smoke_offline(monkeypatch, capsys):
    """
    diag.py на том же фейке: скрипт проходит все разделы без сети, печатает
    token_id обеих ног, вердикты фильтров и воронку, и выходит с кодом 0,
    когда бот взял бы рынок. Ордеров он не шлёт — у клиента их просто нет.
    """
    import argparse

    import diag

    series = gamma_series([gamma_event([market_dict(1)])])
    fake = FakeGammaClient(series={"bitcoin-up-or-down": [series]})

    class FakePublicClient:
        async def __aenter__(self):
            return fake

        async def __aexit__(self, *exc) -> None:
            return None

    monkeypatch.setattr(diag, "AsyncPublicClient", FakePublicClient)

    code = asyncio.run(diag.run(argparse.Namespace(max_series=50, horizon_hours=24.0)))
    out = capsys.readouterr().out

    assert code == 0
    assert "111" in out and "122" in out          # token_id обеих ног
    assert "accepting_orders: да" in out
    assert "enable_order_book: да" in out
    assert "БОТ ВЗЯЛ БЫ" in out
    assert "воронка" in out
    # Скан серий нашёл слаг по маркеру.
    assert "bitcoin-up-or-down" in out


def test_diag_exit_code_2_when_bot_settings_find_nothing(monkeypatch, capsys):
    """Рынки на бирже есть, но слаг бота другой — diag говорит именно это."""
    import argparse

    import diag

    series = gamma_series(
        [gamma_event([market_dict(2, slug="bitcoin-up-or-down-twap-1")],
                     title="Bitcoin Up or Down TWAP")],
        slug="bitcoin-up-or-down-twap", title="Bitcoin Up or Down TWAP",
    )
    fake = FakeGammaClient(series={"bitcoin-up-or-down-twap": [series]})

    class FakePublicClient:
        async def __aenter__(self):
            return fake

        async def __aexit__(self, *exc) -> None:
            return None

    monkeypatch.setattr(diag, "AsyncPublicClient", FakePublicClient)

    code = asyncio.run(diag.run(argparse.Namespace(max_series=50, horizon_hours=24.0)))
    out = capsys.readouterr().out

    assert code == 2
    # Скан по маркерам актуальный слаг всё равно нашёл и показал.
    assert "bitcoin-up-or-down-twap" in out
    assert "СЕРИИ" in out.upper() or "слаги" in out.lower()
