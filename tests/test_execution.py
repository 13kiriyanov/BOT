"""
Тесты обработки user-stream: дедупликация трейдов и атрибуция наших ног.

Формат user-канала описывает трейд с точки зрения ТЕЙКЕРА: верхнеуровневые
side/price/size — его. Наша нога (мы всегда post_only-мейкер) лежит в
maker_orders со своими side/price/matched_amount. Эти тесты фиксируют, что
execution.py читает именно её, а события-повторы (смена статуса трейда,
переигрывание истории после реконнекта) не задваивают позицию.

Клиент SDK не нужен — сюда не доходит ни один сетевой вызов; пакет
polymarket нужен только для импорта execution.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

pytest.importorskip("polymarket", reason="execution импортирует типы SDK")

from src import execution as execution_mod  # noqa: E402
from src.execution import OrderManager  # noqa: E402
from src.models import Fill, LiveOrder  # noqa: E402

D = Decimal


class Obj:
    """Объект с произвольными полями — как payload события SDK."""

    def __init__(self, **fields) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


def make_manager(fills: list[Fill]) -> OrderManager:
    async def on_fill(fill: Fill) -> None:
        fills.append(fill)

    om = OrderManager(
        object(),  # type: ignore[arg-type] - до клиента здесь не доходит
        dry_run=True,
        requote_threshold_ticks=1,
        order_ttl_s=0,
        on_fill=on_fill,
    )
    om.register_market("0xcond", "tok_yes", "tok_no")
    return om


def trade_event(**payload_fields) -> Obj:
    return Obj(type="trade", payload=Obj(**payload_fields))


def our_maker_trade(status: str = "MATCHED", trade_id: str = "trade-1") -> Obj:
    """Трейд, где нашу BUY-YES котировку исполнил тейкер-продавец."""
    return trade_event(
        id=trade_id,
        status=status,
        # Верхнеуровневые поля — ТЕЙКЕРА: он продаёт, и цена у него своя.
        token_id="tok_yes",
        side="SELL",
        price="0.45",
        size="35",
        taker_order_id="taker-1",
        trader_side="MAKER",
        maker_orders=[
            Obj(
                order_id="our-1",
                token_id="tok_yes",
                side="BUY",
                price="0.49",
                matched_amount="20",
                fee_rate_bps=None,
            ),
            Obj(  # чужой мейкер в том же трейде
                order_id="alien-7",
                token_id="tok_yes",
                side="BUY",
                price="0.48",
                matched_amount="15",
                fee_rate_bps=None,
            ),
        ],
    )


def test_maker_fill_uses_our_leg_not_takers():
    """
    Наш филл берётся из maker_orders: сторона и цена НАШЕГО ордера.
    Взять верхнеуровневые side/price значит перевернуть знак позиции.
    """
    fills: list[Fill] = []
    om = make_manager(fills)
    om._track(LiveOrder("our-1", "tok_yes", "BUY", D("0.49"), D("20")))

    asyncio.run(om._handle_user_event(our_maker_trade()))

    assert len(fills) == 1
    fill = fills[0]
    assert (fill.side, fill.price, fill.size) == ("BUY", D("0.49"), D("20"))
    assert fill.condition_id == "0xcond"
    assert fill.trade_id == "trade-1"


def test_duplicate_trade_events_are_counted_once():
    """
    User-stream шлёт событие на каждую смену статуса (MATCHED -> MINED ->
    CONFIRMED) и повторяет историю после реконнекта. Без дедупа по trade id
    каждый филл задваивается — и позиция, и PnL, и риск-лимиты.
    """
    fills: list[Fill] = []
    om = make_manager(fills)
    om._track(LiveOrder("our-1", "tok_yes", "BUY", D("0.49"), D("20")))

    for status in ("MATCHED", "MINED", "CONFIRMED", "MATCHED"):
        asyncio.run(om._handle_user_event(our_maker_trade(status=status)))

    assert len(fills) == 1

    # Другой трейд с другим id — не дубль.
    asyncio.run(om._handle_user_event(our_maker_trade(trade_id="trade-2")))
    assert len(fills) == 2


def test_attribution_survives_order_untrack():
    """
    Order-событие со статусом MATCHED снимает ордер с учёта РАНЬШЕ, чем
    приходит trade-событие. Атрибуция обязана работать по памяти о своих
    order id, а не по живым ордерам.
    """
    fills: list[Fill] = []
    om = make_manager(fills)
    om._track(LiveOrder("our-1", "tok_yes", "BUY", D("0.49"), D("20")))
    om._untrack("our-1")  # ордер уже полностью исполнен и снят

    asyncio.run(om._handle_user_event(our_maker_trade()))
    assert len(fills) == 1 and fills[0].side == "BUY"


def test_foreign_trade_is_not_booked():
    """
    Трейд, где ни одна нога не наша, не бронируется: верхнеуровневые поля
    тейкерские, и учесть их «как есть» значит рискнуть знаком позиции.
    Расхождение, если оно настоящее, скорректирует сверка позиций.
    """
    fills: list[Fill] = []
    om = make_manager(fills)  # наших ордеров не помним вовсе

    asyncio.run(om._handle_user_event(our_maker_trade()))
    assert fills == []


def test_taker_fallback_uses_top_level_fields():
    """Если тейкер — мы (при post_only не должно случаться), поля наши."""
    fills: list[Fill] = []
    om = make_manager(fills)
    om._remember_order("taker-1")

    event = trade_event(
        id="trade-9",
        status="MATCHED",
        token_id="tok_yes",
        side="SELL",
        price="0.45",
        size="35",
        taker_order_id="taker-1",
        trader_side="TAKER",
        maker_orders=[],
    )
    asyncio.run(om._handle_user_event(event))

    assert len(fills) == 1
    assert (fills[0].side, fills[0].price, fills[0].size) == ("SELL", D("0.45"), D("35"))


def test_seen_trades_lru_is_bounded(monkeypatch):
    """Память дедупликации ограничена и не течёт на длинной сессии."""
    monkeypatch.setattr(execution_mod, "SEEN_TRADES_MAX", 8)
    fills: list[Fill] = []
    om = make_manager(fills)
    om._track(LiveOrder("our-1", "tok_yes", "BUY", D("0.49"), D("20")))

    for i in range(50):
        asyncio.run(om._handle_user_event(our_maker_trade(trade_id=f"t-{i}")))

    assert len(om._seen_trades) <= 8
    assert len(fills) == 50  # ограничение памяти не съело уникальные трейды


def test_rejected_statuses_are_ignored():
    """FAILED/RETRYING не бронируются (сверка позиций — страховка)."""
    fills: list[Fill] = []
    om = make_manager(fills)
    om._track(LiveOrder("our-1", "tok_yes", "BUY", D("0.49"), D("20")))

    asyncio.run(om._handle_user_event(our_maker_trade(status="FAILED")))
    asyncio.run(om._handle_user_event(our_maker_trade(status="RETRYING")))
    assert fills == []

    # А валидный статус после отклонённых — бронируется (id не был потрачен).
    asyncio.run(om._handle_user_event(our_maker_trade(status="MATCHED")))
    assert len(fills) == 1


def test_gtd_ttl_is_clamped_to_exchange_minimum():
    """
    SDK отклоняет GTD с expiration ближе now+180s. TTL ниже минимума молча
    оставил бы бота вообще без ордеров — клампим с предупреждением.
    """
    om = OrderManager(
        object(),  # type: ignore[arg-type]
        dry_run=True,
        requote_threshold_ticks=1,
        order_ttl_s=60,
    )
    assert om.order_ttl_s == execution_mod.MIN_GTD_TTL_S

    gtc = OrderManager(
        object(),  # type: ignore[arg-type]
        dry_run=True,
        requote_threshold_ticks=1,
        order_ttl_s=0,
    )
    assert gtc.order_ttl_s == 0  # GTC остаётся GTC
