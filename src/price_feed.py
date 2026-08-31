"""
Фид спот-цены BTC/ETH.

Два источника:
1. 'polymarket' — встроенный RTDS-стрим SDK (topic 'prices.crypto.binance').
   Предпочтителен: это ровно те цены, по которым Polymarket резолвит рынки,
   значит нет базисного риска между твоей моделью и расчётом рынка.
2. 'binance'  — прямой WebSocket Binance. Ниже задержка, но появляется риск
   расхождения с источником резолюции Polymarket.

Оба варианта обновляют один и тот же VolatilityEstimator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from decimal import Decimal

import websockets
from polymarket import AsyncSecureClient
from polymarket.streams import CryptoPricesSpec

from .fair_value import VolatilityEstimator

log = logging.getLogger("polybot.price")

# Символы Polymarket RTDS -> внутреннее имя актива.
POLY_SYMBOLS = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
BINANCE_STREAMS = {"btcusdt@bookTicker": "BTC", "ethusdt@bookTicker": "ETH"}


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

    def ingest(self, asset: str, price: float, ts: float | None = None) -> None:
        ts = ts or time.time()
        self._prices[asset] = price
        self._updated[asset] = ts
        self._est(asset).update(price, ts)

    def stop(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- runners

    async def run_polymarket(self, client: AsyncSecureClient) -> None:
        """Подписка на встроенный крипто-фид Polymarket с авто-переподключением."""
        spec = CryptoPricesSpec(
            topic="prices.crypto.binance", symbols=list(POLY_SYMBOLS.keys())
        )
        backoff = 1.0
        while not self._stop.is_set():
            try:
                log.info("Подключаюсь к Polymarket crypto price stream")
                async with client.subscribe(spec) as stream:
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
                        asset = POLY_SYMBOLS.get(str(symbol).upper())
                        if asset:
                            self.ingest(asset, float(value))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - фид не должен ронять бота
                log.warning("Price stream упал: %s. Переподключение через %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def run_binance(self, ws_url: str) -> None:
        """Прямой WebSocket Binance (bookTicker — mid по лучшим bid/ask)."""
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
