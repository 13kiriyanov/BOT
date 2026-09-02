"""
Смоук-тесты стримов на заглушке, повторяющей ФОРМУ SDK.

Багу «async with client.subscribe(...) без await» было достаточно одной
секунды живого запуска, но 106 тестов его не поймали: стримы тестировались
на заглушках, чья форма отличалась от SDK. Здесь заглушка повторяет контракт
SDK буквально — subscribe() это КОРУТИНА, возвращающая handle, и только
handle является async context manager'ом и async-итератором. Забытый await
роняет каждый из этих тестов тем же TypeError, что и живой запуск.

Сеть и ключи не нужны; пакет polymarket нужен для импорта модулей.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

pytest.importorskip("polymarket", reason="модули стримов импортируют типы SDK")

from src.execution import OrderManager  # noqa: E402
from src.models import Fill, LiveOrder  # noqa: E402
from src.orderbook import OrderBookManager  # noqa: E402
from src.price_feed import SpotFeed  # noqa: E402

D = Decimal


class Obj:
    def __init__(self, **fields) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


class FakeHandle:
    """Как SubscriptionHandle SDK: сам себе async-CM и async-итератор."""

    def __init__(self, events: list) -> None:
        self._events = list(events)

    async def __aenter__(self) -> "FakeHandle":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def __aiter__(self) -> "FakeHandle":
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def close(self) -> None:
        return None


class FakeStreamClient:
    """
    subscribe() — корутина, как в SDK. Первая подписка отдаёт события;
    вторая (реконнект потребителя после исчерпания) останавливает его
    через stopper, чтобы бесконечный цикл переподключения завершился.
    """

    def __init__(self, events: list) -> None:
        self._events = events
        self.calls = 0
        self.stopper = None
        self.specs: list = []

    async def subscribe(self, spec):  # noqa: ANN001 - спец SDK
        self.calls += 1
        self.specs.append(spec)
        if self.calls == 1:
            return FakeHandle(self._events)
        if self.stopper is not None:
            self.stopper()
        return FakeHandle([])


async def drive(coro) -> None:
    await asyncio.wait_for(coro, timeout=5.0)


def test_user_stream_consumes_events_through_sdk_shaped_subscribe():
    """execution.run_user_stream: филл доходит до колбэка, а не до TypeError."""
    fills: list[Fill] = []

    async def on_fill(fill: Fill) -> None:
        fills.append(fill)

    events = [
        Obj(type="trade", payload=Obj(
            id="t-1", status="MATCHED", token_id="tok_yes", side="SELL",
            price="0.45", size="35", taker_order_id="tk", trader_side="MAKER",
            maker_orders=[Obj(order_id="our-1", token_id="tok_yes", side="BUY",
                              price="0.49", matched_amount="20", fee_rate_bps=None)],
        )),
    ]
    client = FakeStreamClient(events)
    om = OrderManager(
        client,  # type: ignore[arg-type]
        dry_run=True, requote_threshold_ticks=1, order_ttl_s=0, on_fill=on_fill,
    )
    om.register_market("0xcond", "tok_yes", "tok_no")
    om._track(LiveOrder("our-1", "tok_yes", "BUY", D("0.49"), D("20")))
    client.stopper = om.stop

    asyncio.run(drive(om.run_user_stream()))

    assert len(fills) == 1
    assert (fills[0].side, fills[0].price) == ("BUY", D("0.49"))
    assert client.calls >= 2


def test_price_feed_consumes_real_chainlink_twap_events():
    """
    price_feed.run_polymarket: события — НАСТОЯЩИЕ модели RTDS SDK для
    потока резолюции (Chainlink TWAP-60s), символы в форме провода
    ('btc/usd') доезжают до self.price(). Заглушка с «удобным» символом
    прятала бы живой баг фильтра, как это уже случилось с binance-топиком.
    """
    from polymarket.models.rtds_events import CryptoPricesChainlinkTwapEvent

    def twap_event(symbol: str, value: str, e18: str) -> object:
        return CryptoPricesChainlinkTwapEvent.model_validate({
            "type": "update", "timestamp": 1788276000000,
            "payload": {"symbol": symbol, "timestamp": 1788276000,
                        "value": value, "full_accuracy_value": e18,
                        "window_s": 60},
        })

    ev_btc = twap_event("btc/usd", "100500.5", "100500500000000000000000")
    ev_eth = twap_event("eth/usd", "4200.25", "4200250000000000000000")
    client = FakeStreamClient([ev_btc, ev_eth])
    feed = SpotFeed(vol_halflife_s=45.0, momentum_halflife_s=8.0,
                    vol_floor_annual=D("0.30"))
    client.stopper = feed.stop

    asyncio.run(drive(feed.run_polymarket(client)))  # type: ignore[arg-type]

    assert feed.price("BTC") == 100500.5
    assert feed.price("ETH") == 4200.25
    assert client.calls >= 2
    # Подписка обязана быть ТОПИКОВОЙ (symbols=None): клиентский фильтр
    # symbols в SDK сравнивает строки точно, несовпадение формата молча
    # убивает фид. И на ОБА окна резолюции разом (последовательность спек):
    # 30s для 5-минутных рынков, 60s для 15-минутных и 4-часовых.
    assert client.specs
    specs = list(client.specs[0])
    assert [s.window_seconds for s in specs] == [30, 60]
    assert all(s.symbols is None for s in specs)


def test_price_feed_keeps_twap_windows_as_separate_series():
    """
    Тики 30s и 60s одного актива — РАЗНЫЕ ряды: 5-минутный рынок читает
    30s, и 60s-значение его не подменяет (и наоборот). Запрос без окна —
    справочный, отдаёт любой имеющийся ряд.
    """
    from polymarket.models.rtds_events import CryptoPricesChainlinkTwapEvent

    def ev(window: int, value: str, e18: str) -> object:
        return CryptoPricesChainlinkTwapEvent.model_validate({
            "type": "update", "timestamp": 1788276000000,
            "payload": {"symbol": "btc/usd", "timestamp": 1788276000,
                        "value": value, "full_accuracy_value": e18,
                        "window_s": window},
        })

    client = FakeStreamClient([
        ev(30, "109510.0", "109510000000000000000000"),
        ev(60, "109490.0", "109490000000000000000000"),
    ])
    feed = SpotFeed(vol_halflife_s=45.0, momentum_halflife_s=8.0,
                    vol_floor_annual=D("0.30"))
    client.stopper = feed.stop
    asyncio.run(drive(feed.run_polymarket(client)))  # type: ignore[arg-type]

    assert feed.price("BTC", 30) == 109510.0
    assert feed.price("BTC", 60) == 109490.0
    assert feed.price("BTC") is not None          # справочный запрос
    assert feed.price("ETH", 30) is None           # чужого ряда нет — None
    assert feed.is_stale("ETH", 4.0, 30)


def _twap_event(symbol: str, value: str, e18: str) -> object:
    from polymarket.models.rtds_events import CryptoPricesChainlinkTwapEvent

    return CryptoPricesChainlinkTwapEvent.model_validate({
        "type": "update", "timestamp": 1788276000000,
        "payload": {"symbol": symbol, "timestamp": 1788276000,
                    "value": value, "full_accuracy_value": e18,
                    "window_s": 60},
    })


class RecordingFeed(SpotFeed):
    """SpotFeed, записывающий каждый вызов ingest — для проверки отбора."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ingested: list[tuple[str, float]] = []

    def ingest(self, asset: str, price: float, ts: float | None = None,
               window: int | None = None) -> None:
        self.ingested.append((asset, price))
        super().ingest(asset, price, ts, window)


def test_price_feed_drops_foreign_assets_from_topic_stream(caplog):
    """
    Топиковая подписка приносит ВСЕ активы потока. В SpotFeed имеют право
    попасть только BTC/ETH: чужая цена (вживую — zec/usd за 838.76,
    скормленный первой строкой лога) не должна доходить ни до ingest, ни
    до строки «фид жив». Первый тик логируется ПО КАЖДОМУ нужному активу.
    """
    import logging as _logging

    events = [
        _twap_event("zec/usd", "838.76", "838760000000000000000"),
        _twap_event("sol/usd", "153.20", "153200000000000000000"),
        _twap_event("btc/usd", "109500.5", "109500500000000000000000"),
        _twap_event("zec/usd", "838.80", "838800000000000000000"),
        _twap_event("eth/usd", "4200.25", "4200250000000000000000"),
    ]
    client = FakeStreamClient(events)
    feed = RecordingFeed(vol_halflife_s=45.0, momentum_halflife_s=8.0,
                         vol_floor_annual=D("0.30"))
    client.stopper = feed.stop

    with caplog.at_level(_logging.DEBUG, logger="polybot.price"):
        asyncio.run(drive(feed.run_polymarket(client)))  # type: ignore[arg-type]

    # ingest вызван ТОЛЬКО для наших активов — по разу на каждый.
    assert feed.ingested == [("BTC", 109500.5), ("ETH", 4200.25)]
    assert feed.price("BTC") == 109500.5
    assert feed.price("ETH") == 4200.25

    infos = [r.message for r in caplog.records if r.levelno == _logging.INFO]
    # Первый тик залогирован по каждому активу; общий «первый тик» (который
    # вживую наврал, показав zec/usd) больше не существует.
    assert any("первый тик BTC" in m for m in infos)
    assert any("первый тик ETH" in m for m in infos)
    assert not any("Крипто-фид жив" in m for m in infos)
    assert not any("zec" in m.lower() for m in infos)
    # Чужие символы ушли в DEBUG, по разу на символ.
    debugs = [r.message for r in caplog.records if r.levelno == _logging.DEBUG]
    assert sum("чужой символ" in m for m in debugs) == 2  # zec и sol

    # Оценщики переведены в режим сглаженного ряда (лаг-выборка >= окна).
    assert feed._est("BTC", 60)._sample_interval == 60.0


def test_orderbook_consumes_snapshots_through_sdk_shaped_subscribe():
    """orderbook.run: снапшот книги доезжает до локального зеркала."""
    events = [
        Obj(type="book", payload=Obj(
            token_id="tok_yes",
            bids=[Obj(price="0.49", size="100")],
            asks=[Obj(price="0.51", size="80")],
        )),
    ]
    client = FakeStreamClient(events)
    books = OrderBookManager()
    books.set_tokens({"tok_yes"})
    client.stopper = books.stop

    asyncio.run(drive(books.run(client)))  # type: ignore[arg-type]

    book = books.book("tok_yes")
    assert book is not None
    assert book.best_bid == D("0.49")
    assert book.best_ask == D("0.51")
    assert client.calls >= 2
