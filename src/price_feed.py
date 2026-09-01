"""
Фид цены BTC/ETH для модели.

Два источника:
1. 'polymarket' — Chainlink TWAP-60s через RTDS SDK
   (CryptoPricesChainlinkTwapSpec, топик 'prices.crypto.chainlink.twap').
   ЭТО ИСТОЧНИК РЕЗОЛЮЦИИ: по правилам updown-рынков исход считается по
   TWAP Chainlink («not according to any other sources or spot markets»),
   и страйк — значение этого же ряда в начале окна. Никакого базисного
   риска между моделью и расчётом рынка.
2. 'binance' — прямой WebSocket Binance. РЕЗЕРВ с базисным риском: это
   другой ряд, чем тот, по которому рынок рассчитается; при подключении
   логируется WARNING.

Оба варианта обновляют один и тот же VolatilityEstimator. В варианте
'polymarket' сигма оценивается по самому TWAP-ряду — сглаженному, то есть
менее волатильному, чем спот. Это не ошибка, а самосогласованность:
TWAP-модель fair value моделирует именно этот ряд.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from decimal import Decimal
from typing import Callable

import websockets
from polymarket import AsyncSecureClient
from polymarket.streams import CryptoPricesChainlinkTwapSpec

from .fair_value import VolatilityEstimator

log = logging.getLogger("polybot.price")

# Символы Polymarket RTDS -> внутреннее имя актива. Ключи — НОРМАЛИЗОВАННАЯ
# форма (upper, без разделителей): формат символа по проводу не зафиксирован
# документацией SDK, а клиентский фильтр подписки сравнивает строки ТОЧНО
# (см. normalize_rtds_symbol ниже). Допускаем и USDT- (топик binance), и
# USD-пары (chainlink-стиль) — коллизий между ними нет.
POLY_SYMBOLS = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "BTCUSD": "BTC", "ETHUSD": "ETH"}
BINANCE_STREAMS = {"btcusdt@bookTicker": "BTC", "ethusdt@bookTicker": "ETH"}


def normalize_rtds_symbol(symbol: str) -> str:
    """
    Привести символ RTDS к канонической форме: upper, без '/', '-', '_'.

    Зачем: SDK фильтрует события по symbols НА КЛИЕНТЕ точным сравнением
    строк, без нормализации регистра (в отличие от equity-фильтра, где есть
    .lower()). Подписка с symbols=["BTCUSDT"] при wire-символе "btcusdt"
    устанавливается без ошибок, но КАЖДОЕ событие молча отбрасывается ещё в
    SDK — ingest() не вызывается ни разу, бот блокирует все рынки по
    «спот-фид протух». Поэтому мы подписываемся на весь топик (symbols не
    передаём) и фильтруем сами, толерантно к регистру и разделителям.
    Канарейка: test_crypto_prices_payload_shape_and_filter_contract.
    """
    return symbol.replace("/", "").replace("-", "").replace("_", "").upper()


class SpotFeed:
    """Держит последнюю цену и волатильность по каждому активу."""

    def __init__(
        self,
        *,
        vol_halflife_s: float,
        momentum_halflife_s: float,
        vol_floor_annual: Decimal,
    ) -> None:
        self._prices: dict[str, float] = {}
        self._updated: dict[str, float] = {}
        self._vol: dict[str, VolatilityEstimator] = {}
        # Подписчики на каждый тик (движок кормит ими детектор режима).
        self._listeners: list[Callable[[str, float, float], None]] = []
        self._cfg = (vol_halflife_s, momentum_halflife_s, float(vol_floor_annual))
        self._stop = asyncio.Event()

    def _est(self, asset: str) -> VolatilityEstimator:
        if asset not in self._vol:
            self._vol[asset] = VolatilityEstimator(*self._cfg)
        return self._vol[asset]

    # ------------------------------------------------------------------ API

    def price(self, asset: str) -> float | None:
        return self._prices.get(asset)

    def sigma(self, asset: str) -> float:
        return self._est(asset).sigma_annual

    def drift(self, asset: str) -> float:
        return self._est(asset).drift_per_second

    def vol_ready(self, asset: str) -> bool:
        return self._est(asset).ready

    def is_stale(self, asset: str, timeout_s: float) -> bool:
        ts = self._updated.get(asset)
        return ts is None or (time.time() - ts) > timeout_s

    def add_listener(self, callback: Callable[[str, float, float], None]) -> None:
        """Подписаться на тики: callback(asset, price, ts)."""
        self._listeners.append(callback)

    def ingest(self, asset: str, price: float, ts: float | None = None) -> None:
        ts = ts or time.time()
        self._prices[asset] = price
        self._updated[asset] = ts
        self._est(asset).update(price, ts)
        for callback in self._listeners:
            try:
                callback(asset, price, ts)
            except Exception as exc:  # noqa: BLE001 - слушатель не роняет фид
                log.error("Слушатель тиков упал: %s", exc)

    def stop(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- runners

    async def run_polymarket(self, client: AsyncSecureClient) -> None:
        """Подписка на поток резолюции (Chainlink TWAP-60s) с переподключением."""
        # Окно 60 секунд — ровно тот поток, на который ссылаются правила
        # рынков (btc-usd-twap-60s-streams). БЕЗ symbols: клиентский фильтр
        # SDK сравнивает символы точно, и несовпадение формата
        # (регистр/разделитель) молча убивает весь фид. Символы этого топика
        # по докстрингу SDK — нижний регистр со слэшем ('btc/usd'), их
        # покрывает normalize_rtds_symbol; фильтрация на нашей стороне.
        spec = CryptoPricesChainlinkTwapSpec(window_seconds=60)
        backoff = 1.0
        first_tick_logged = False
        unmatched_seen: set[str] = set()
        while not self._stop.is_set():
            try:
                log.info(
                    "Подключаюсь к потоку резолюции Chainlink TWAP-60s "
                    "(prices.crypto.chainlink.twap)"
                )
                # subscribe() — корутина: await возвращает handle-CM.
                async with await client.subscribe(spec) as stream:
                    backoff = 1.0
                    async for event in stream:
                        if self._stop.is_set():
                            break
                        payload = getattr(event, "payload", None)
                        if payload is None:
                            continue
                        symbol = getattr(payload, "symbol", None)
                        value = getattr(payload, "value", None)
                        if symbol is None or value is None:
                            continue
                        # Защита от чужого окна TWAP, если фильтр SDK
                        # пропустит его при обновлении.
                        window = getattr(payload, "window_seconds", None)
                        if window is not None and window != 60:
                            continue
                        if not first_tick_logged:
                            # Живое доказательство формата символа по проводу.
                            first_tick_logged = True
                            log.info(
                                "Крипто-фид жив: первый тик symbol=%r value=%s",
                                symbol, value,
                            )
                        norm = normalize_rtds_symbol(str(symbol))
                        asset = POLY_SYMBOLS.get(norm)
                        if asset:
                            self.ingest(asset, float(value))
                        elif norm not in unmatched_seen:
                            unmatched_seen.add(norm)
                            log.debug("Крипто-фид: чужой символ %r — пропускаю", symbol)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - фид не должен ронять бота
                log.warning("Price stream упал: %s. Переподключение через %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def run_binance(self, ws_url: str) -> None:
        """Прямой WebSocket Binance (bookTicker — mid по лучшим bid/ask)."""
        log.warning(
            "PRICE_SOURCE=binance — РЕЗЕРВНЫЙ источник с базисным риском: "
            "рынки резолвятся по Chainlink TWAP, а не по Binance-споту. "
            "Модель и страйк будут считаться по другому ряду, чем расчёт "
            "рынка. Для боевой торговли используй PRICE_SOURCE=polymarket."
        )
        url = f"{ws_url}?streams={'/'.join(BINANCE_STREAMS)}"
        backoff = 1.0
        while not self._stop.is_set():
            try:
                log.info("Подключаюсь к Binance WS")
                async with websockets.connect(url, ping_interval=15) as ws:
                    backoff = 1.0
                    while not self._stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        msg = json.loads(raw)
                        data = msg.get("data", {})
                        stream = msg.get("stream", "")
                        asset = BINANCE_STREAMS.get(stream)
                        if not asset:
                            continue
                        bid, ask = data.get("b"), data.get("a")
                        if bid and ask:
                            self.ingest(asset, (float(bid) + float(ask)) / 2.0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("Binance WS упал: %s. Переподключение через %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
