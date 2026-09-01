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

    async def subscribe(self, spec):  # noqa: ANN001 - спец SDK
        self.calls += 1
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


def test_price_feed_consumes_ticks_through_sdk_shaped_subscribe():
    """price_feed.run_polymarket: тик спота доезжает до self.price()."""
    events = [Obj(payload=Obj(symbol="BTCUSDT", value="100500.5"))]
    client = FakeStreamClient(events)
    feed = SpotFeed(vol_halflife_s=45.0, momentum_halflife_s=8.0,
                    vol_floor_annual=D("0.30"))
    client.stopper = feed.stop

    asyncio.run(drive(feed.run_polymarket(client)))  # type: ignore[arg-type]

    assert feed.price("BTC") == 100500.5
    assert client.calls >= 2


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
