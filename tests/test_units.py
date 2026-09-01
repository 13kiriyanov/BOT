"""
Аудит единиц на границе с SDK. Каждый вызов self.client.*, принимающий или
возвращающий числа, закреплён тестом, который сверяет НАШИ ожидания единиц
с самим SDK — его константами, валидаторами и моделями ответов.

Зачем: баг класса «не те единицы» не падает — он молча делает не то.
merge_positions(amount=25) вместо 25_000_000 мержил одну миллионную пачки,
платил полный газ и проходил все тесты; order_ttl_s=60 против биржевого
минимума 180 молча оставлял бота вообще без ордеров. Эти тесты — канарейки:
если при обновлении SDK какой-то из них упал (в том числе ImportError на
приватном пути), единицы этого вызова надо перепроверить руками, а не
чинить тест до зелёного.

Карта «вызов -> единицы -> тест» — в CLAUDE.md, раздел «Единицы на границе
с SDK».

Что НЕ проверяется здесь и не проверяемо без живой биржи: что сама биржа
трактует эти поля так же, как клиентский SDK. Мы фиксируем контракт
«наш код <-> polymarket-client»; контракт «SDK <-> биржа» — зона
ответственности SDK.
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

pytest.importorskip("polymarket", reason="аудит сверяется с самим SDK")

from src.models import MIN_GTD_TTL_S, POSITION_DECIMALS, shares_to_base_units  # noqa: E402

D = Decimal

CONDITION = "0x" + "ab" * 32


# --------------------------------------------------- merge_positions(amount)


def test_merge_amount_is_base_units_and_scale_matches_sdk():
    """
    engine.merge_loop: merge_positions(amount=...) ждёт БАЗОВЫЕ ЕДИНИЦЫ
    ERC-1155 (6 знаков), сверяемые с балансом токенов on-chain. Наша
    константа масштаба обязана совпадать с константой SDK.
    """
    from polymarket._internal.actions.relayer.positions import _TOKEN_DECIMALS

    assert int(POSITION_DECIMALS) == _TOKEN_DECIMALS == 1_000_000
    assert shares_to_base_units(D("25")) == 25 * _TOKEN_DECIMALS


def test_sdk_own_position_to_merge_amount_uses_same_scale():
    """SDK сам конвертирует Position.size (shares) в amount умножением на 1e6."""
    from polymarket._internal.actions.relayer import positions as relayer_positions

    # Функция-конвертер SDK — та же операция, что shares_to_base_units.
    src_text = open(relayer_positions.__file__).read()
    assert "position.size * Decimal(_TOKEN_DECIMALS)" in src_text


# --------------------------------------- create_limit_order(price, size, ...)


def test_limit_order_price_and_size_are_human_units():
    """
    execution.place_batch: price — цена в USDC (0..1), size — shares.
    Валидатор SDK хранит их как есть, а в wei переводит сам при подписании
    (parse_amount, 6 знаков). Передавать сюда базовые единицы нельзя.
    """
    from polymarket._internal.actions.orders.limit import validate_limit_order_params
    from polymarket._internal.actions.orders.math import parse_amount

    params = validate_limit_order_params(
        token_id="123", price="0.49", size="20", side="BUY", post_only=True
    )
    assert params.price == D("0.49")   # не отскалировано — человеческие единицы
    assert params.size == D("20")
    # Скалирование в базовые единицы SDK делает сам, этой функцией:
    assert parse_amount(D("0.49")) == 490_000
    assert parse_amount(D("20")) == 20_000_000


def test_limit_order_expiration_is_unix_seconds_with_exchange_minimum():
    """
    execution.place_batch: expiration — unix-СЕКУНДЫ, и не ближе, чем
    now + 180. Короче — UserInputError, то есть ордер вообще не подпишется.
    Наш MIN_GTD_TTL_S обязан покрывать минимум SDK с запасом.
    """
    from polymarket._internal.actions.orders.limit import (
        _MIN_EXPIRATION_BUFFER_S,
        validate_limit_order_params,
    )
    from polymarket.errors import UserInputError

    assert MIN_GTD_TTL_S >= _MIN_EXPIRATION_BUFFER_S + 10

    with pytest.raises(UserInputError):
        validate_limit_order_params(
            token_id="123", price="0.49", size="20", side="BUY",
            expiration=int(time.time()) + 60,   # старый дефолт конфига
        )
    # Наш эффективный TTL проходит валидацию.
    validate_limit_order_params(
        token_id="123", price="0.49", size="20", side="BUY",
        expiration=int(time.time()) + MIN_GTD_TTL_S,
    )


def test_config_default_ttl_is_valid_for_exchange():
    """Дефолт конфига не должен снова опуститься ниже биржевого минимума."""
    from src.config import StrategySettings

    ttl = StrategySettings().order_ttl_s
    assert ttl == 0 or ttl >= MIN_GTD_TTL_S


# ------------------------------------------------- list_positions() -> Position


def test_position_model_returns_human_units():
    """
    engine._fetch_positions / _sync_positions_once: Position.size — shares,
    avg_price — USDC за share. Это ЧЕЛОВЕЧЕСКИЕ единицы: их можно класть в
    MarketPosition как есть, но НЕЛЬЗЯ передавать в merge_positions(amount).
    """
    from polymarket.models.data.portfolio import Position

    p = Position.model_validate(
        {
            "conditionId": CONDITION, "asset": "123", "size": "12.5",
            "avgPrice": "0.42", "outcome": "Up", "outcomeIndex": 0,
        }
    )
    assert p.size == D("12.5")
    assert p.avg_price == D("0.42")
    assert p.outcome == "Up"          # Up/Down-серии: ярлык не Yes/No


# --------------------------------------------- list_open_orders() -> OpenOrder


def test_open_order_model_returns_human_units():
    """
    execution.sync_open_orders: price/original_size/size_matched — USDC-цена
    и shares, как в наших LiveOrder. Сверка не требует пересчёта единиц.
    """
    from polymarket.models.clob.account import OpenOrder

    o = OpenOrder.model_validate(
        {
            "id": "o1", "market": CONDITION, "asset_id": "123",
            "owner": "me", "maker_address": "0x0", "side": "BUY",
            "price": "0.49", "original_size": "20", "size_matched": "5",
            "outcome": "Yes", "order_type": "GTC", "status": "LIVE",
            "created_at": int(time.time()),
        }
    )
    assert o.price == D("0.49")
    assert o.original_size == D("20")
    assert o.size_matched == D("5")


# ------------------------------------- get_balance_allowance() -> BalanceAllowance


def test_balance_allowance_is_base_units_not_usdc():
    """
    engine._check_balance: balance приходит в БАЗОВЫХ единицах (int, 6
    знаков), а не в USDC — отображение обязано делить на POSITION_DECIMALS.
    """
    from polymarket.models.clob.account import BalanceAllowance

    b = BalanceAllowance.model_validate(
        {"balance": "500000000", "allowances": {"exchange": "1000000"}}
    )
    assert b.balance == 500_000_000
    assert isinstance(b.balance, int)
    assert D(b.balance) / POSITION_DECIMALS == D("500")


# ------------------------------------------------------- subscribe(spec)


def test_subscribe_is_a_coroutine_returning_handle():
    """
    price_feed.run_polymarket / orderbook.run / execution.run_user_stream:
    subscribe() — КОРУТИНА (async def), а не фабрика контекст-менеджера.
    `async with client.subscribe(...)` без await падает с «coroutine object
    does not support the asynchronous context manager protocol» — все три
    стрима умерли бы на старте. Правильная форма:

        async with await client.subscribe(spec) as stream:
            async for event in stream: ...

    Возвращаемый SubscriptionHandle — сам себе async-CM и async-итератор.
    Если канарейка упала после обновления SDK, перепроверь форму вызова
    во всех трёх модулях руками.
    """
    import inspect

    from polymarket import AsyncSecureClient
    from polymarket._internal.streams.handle import SubscriptionHandle

    assert inspect.iscoroutinefunction(AsyncSecureClient.subscribe)
    for attr in ("__aenter__", "__aexit__", "__aiter__", "__anext__", "close"):
        assert hasattr(SubscriptionHandle, attr), attr

    # Пагинаторы, наоборот, НЕ корутины: их зовут без await и сразу
    # итерируют. Перепутать эти две формы — симметричный способ сломаться.
    for name in ("list_positions", "list_open_orders", "list_series", "list_events"):
        assert not inspect.iscoroutinefunction(getattr(AsyncSecureClient, name)), name


# ----------------------------------------------- user-stream: событие trade


def test_user_trade_payload_units_and_maker_shape():
    """
    execution._extract_own_fills: и верхнеуровневые price/size, и
    maker_orders[].price/matched_amount — человеческие единицы. Форма
    события: верхнеуровневые side/price — ТЕЙКЕРА, наша мейкерская нога —
    в maker_orders со своими side/price/matched_amount и своим order_id.
    """
    from polymarket.models.clob.user_events import UserTradePayload

    t = UserTradePayload.model_validate(
        {
            "id": "t1", "taker_order_id": "tk1", "market": CONDITION,
            "asset_id": "123", "side": "sell", "size": "35", "price": "0.45",
            "status": "MATCHED", "owner": "me", "trader_side": "maker",
            "maker_orders": [
                {
                    "order_id": "our-1", "owner": "me", "asset_id": "123",
                    "side": "buy", "price": "0.49", "matched_amount": "20",
                }
            ],
        }
    )
    assert (t.side, t.price, t.size) == ("SELL", D("0.45"), D("35"))
    assert t.trader_side == "MAKER"   # поле различает нашу роль — формат
    mo = t.maker_orders[0]            # события НЕ нормализован под получателя
    assert (mo.side, mo.price, mo.matched_amount) == ("BUY", D("0.49"), D("20"))
    assert mo.order_id == "our-1"
