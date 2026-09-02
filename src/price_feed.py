"""
Фид цены BTC/ETH для модели.

Два источника:
1. 'polymarket' — Chainlink TWAP через RTDS SDK, ОБА окна сразу
   (CryptoPricesChainlinkTwapSpec 30s и 60s, топик
   'prices.crypto.chainlink.twap'). ЭТО ИСТОЧНИК РЕЗОЛЮЦИИ: по правилам
   updown-рынков исход считается по TWAP Chainlink («not according to any
   other sources or spot markets»), и страйк — значение этого же ряда в
   начале окна. 5-минутные рынки резолвятся по 30s-ряду, 15-минутные и
   4-часовые — по 60s (анонс Polymarket от 7.08.2026); каждый рынок читает
   ряд своего окна. Никакого базисного риска между моделью и расчётом.
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


def asset_for_symbol(symbol: str) -> str | None:
    """
    Наш актив по wire-символу; None — ЧУЖОЙ символ (zec/usd, sol/usd, ...).

    Топиковая подписка приносит ВСЕ активы потока — в SpotFeed имеют право
    попасть только BTC и ETH. Всё, что вернуло None, отбрасывается до
    ingest(): чужая цена, скормленная модели BTC/ETH, ломает fair молча.
    """
    return POLY_SYMBOLS.get(normalize_rtds_symbol(symbol))


# Активы, тики которых фид ОБЯЗАН отдавать; отсутствие любого из них через
# FEED_ASSETS_WARN_AFTER_S после подключения — WARNING в лог.
EXPECTED_ASSETS = ("BTC", "ETH")
FEED_ASSETS_WARN_AFTER_S = 30.0


# Окна потока Chainlink TWAP, на которые подписываемся: 30s резолвит
# 5-минутные рынки, 60s — 15-минутные и 4-часовые. Каждый рынок читает
# ряд СВОЕГО окна (TargetMarket.twap_window_s).
TWAP_WINDOWS = (30, 60)

# Слушатель тиков: callback(asset, price, ts, window). window None —
# несглаженный источник (Binance), общий для всех окон.
TickListener = Callable[[str, float, float, "int | None"], None]


class SpotFeed:
    """
    Последняя цена и волатильность по каждому ряду (актив, окно TWAP).

    Ряды хранятся по ключу (asset, window). Запрос с окном, которого нет,
    падает на общий ряд без окна (Binance) — и только на него: чужое окно
    TWAP не подменяет нужное (5-минутный рынок с рядом 60s считал бы страйк
    и среднее по другому процессу, чем его резолюция).
    """

    def __init__(
        self,
        *,
        vol_halflife_s: float,
        momentum_halflife_s: float,
        vol_floor_annual: Decimal,
    ) -> None:
        self._prices: dict[tuple[str, int | None], float] = {}
        self._updated: dict[tuple[str, int | None], float] = {}
        self._vol: dict[tuple[str, int | None], VolatilityEstimator] = {}
        # Подписчики на каждый тик (движок кормит ими детектор режима и
        # накопители TWAP).
        self._listeners: list[TickListener] = []
        self._cfg = (vol_halflife_s, momentum_halflife_s, float(vol_floor_annual))
        # Kwargs оценщика под источник: пустые = тиковый режим (Binance).
        self._est_kwargs: dict = {}
        self._stop = asyncio.Event()

    def _key(self, asset: str, window: int | None) -> tuple[str, int | None]:
        """
        Ключ ряда с резервом. Конкретное окно -> только оно или общий ряд
        без окна (Binance); чужое окно TWAP его НЕ подменяет. Запрос без
        окна (справочный спот для discovery) — любой имеющийся ряд актива.
        """
        if (asset, window) in self._prices:
            return asset, window
        if window is not None and (asset, None) in self._prices:
            return asset, None
        if window is None:
            for known in TWAP_WINDOWS[::-1]:
                if (asset, known) in self._prices:
                    return asset, known
        return asset, window

    def _est(self, asset: str, window: int | None = None) -> VolatilityEstimator:
        key = self._key(asset, window)
        if key not in self._vol:
            kwargs = dict(self._est_kwargs)
            if key[1] is not None and "ma_window_s" in kwargs:
                # Поправка скользящего среднего — под окно КОНКРЕТНОГО ряда.
                kwargs["ma_window_s"] = float(key[1])
                kwargs["sample_interval_s"] = max(
                    kwargs["sample_interval_s"], float(key[1])
                )
            self._vol[key] = VolatilityEstimator(*self._cfg, **kwargs)
        return self._vol[key]

    def _use_resolution_feed_estimators(self) -> None:
        """
        Перевести оценщики в режим сглаженного ряда (потоки Chainlink TWAP):
        лаг-выборка 60с + поправка скользящего среднего под окно ряда,
        полупериод EWMA растянут под редкие сэмплы (45с при выборке раз в
        60с означал бы полную замену дисперсии каждым сэмплом). Готовность —
        12 сэмплов (~12 минут фида); до неё модель живёт на floor-
        подмешанной sigma. Вызывается run_polymarket ДО первого ingest.
        """
        self._est_kwargs = dict(
            sample_interval_s=60.0, ma_window_s=60.0, ready_samples=12,
        )
        vol_halflife_s, momentum_halflife_s, vol_floor = self._cfg
        self._cfg = (max(vol_halflife_s, 600.0), momentum_halflife_s, vol_floor)
        self._vol.clear()

    # ------------------------------------------------------------------ API

    def price(self, asset: str, window: int | None = None) -> float | None:
        return self._prices.get(self._key(asset, window))

    def sigma(self, asset: str, window: int | None = None) -> float:
        return self._est(asset, window).sigma_annual

    def drift(self, asset: str, window: int | None = None) -> float:
        return self._est(asset, window).drift_per_second

    def vol_ready(self, asset: str, window: int | None = None) -> bool:
        return self._est(asset, window).ready

    def is_stale(
        self, asset: str, timeout_s: float, window: int | None = None
    ) -> bool:
        ts = self._updated.get(self._key(asset, window))
        return ts is None or (time.time() - ts) > timeout_s

    def add_listener(self, callback: TickListener) -> None:
        """Подписаться на тики: callback(asset, price, ts, window)."""
        self._listeners.append(callback)

    def ingest(
        self,
        asset: str,
        price: float,
        ts: float | None = None,
        window: int | None = None,
    ) -> None:
        ts = ts or time.time()
        key = (asset, window)
        self._prices[key] = price
        self._updated[key] = ts
        self._est(asset, window).update(price, ts)
        for callback in self._listeners:
            try:
                callback(asset, price, ts, window)
            except Exception as exc:  # noqa: BLE001 - слушатель не роняет фид
                log.error("Слушатель тиков упал: %s", exc)

    def stop(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- runners

    async def run_polymarket(self, client: AsyncSecureClient) -> None:
        """Подписка на потоки резолюции (Chainlink TWAP 30s и 60s)."""
        # ОБА окна одной подпиской (SDK принимает последовательность спек и
        # отдаёт MergedSubscriptionHandle — тот же CM/итератор): 30s резолвит
        # 5-минутные рынки, 60s — 15-минутные и 4-часовые. Окно события —
        # payload.window_seconds, по нему тик кладётся в СВОЙ ряд. БЕЗ
        # symbols: клиентский фильтр SDK сравнивает символы точно, и
        # несовпадение формата (регистр/разделитель) молча убивает весь
        # фид. Символы топика — нижний регистр со слэшем ('btc/usd'), их
        # покрывает normalize_rtds_symbol; фильтрация на нашей стороне.
        specs = [CryptoPricesChainlinkTwapSpec(window_seconds=w) for w in TWAP_WINDOWS]
        self._use_resolution_feed_estimators()
        backoff = 1.0
        # Живое доказательство покрытия: первый тик логируется ПО КАЖДОМУ
        # ряду (актив, окно) отдельно. Один общий «первый тик» врал: топиковая
        # подписка начинается с чужого символа (вживую — zec/usd), строка
        # выглядела как жизнь фида, а BTC/ETH могли не прийти вовсе.
        series_seen: set[tuple[str, int]] = set()
        foreign_seen: set[str] = set()
        missing_warned = False
        connected_at: float | None = None
        while not self._stop.is_set():
            try:
                log.info(
                    "Подключаюсь к потокам резолюции Chainlink TWAP %s "
                    "(prices.crypto.chainlink.twap)",
                    "/".join(f"{w}s" for w in TWAP_WINDOWS),
                )
                # subscribe() — корутина: await возвращает handle-CM.
                async with await client.subscribe(specs) as stream:
                    backoff = 1.0
                    connected_at = time.time()
                    async for event in stream:
                        if self._stop.is_set():
                            break
                        payload = getattr(event, "payload", None)
                        if payload is None:
                            continue
                        symbol = getattr(payload, "symbol", None)
                        value = getattr(payload, "value", None)
                        window = getattr(payload, "window_seconds", None)
                        if symbol is None or value is None:
                            continue
                        # Окно обязано быть одним из подписанных: тик без
                        # окна или с чужим окном не кладём никуда.
                        if window not in TWAP_WINDOWS:
                            continue

                        # ФИЛЬТР АКТИВОВ — до любых логов и ingest: в
                        # SpotFeed попадают только BTC/ETH, чужие символы
                        # отбрасываются (DEBUG один раз на символ).
                        asset = asset_for_symbol(str(symbol))
                        if asset is None:
                            norm = normalize_rtds_symbol(str(symbol))
                            if norm not in foreign_seen:
                                foreign_seen.add(norm)
                                log.debug(
                                    "Крипто-фид: чужой символ %r — пропускаю",
                                    symbol,
                                )
                        else:
                            if (asset, window) not in series_seen:
                                series_seen.add((asset, window))
                                log.info(
                                    "Фид резолюции: первый тик %s@%ds "
                                    "(symbol=%r value=%s)",
                                    asset, window, symbol, value,
                                )
                            self.ingest(asset, float(value), window=window)

                        # Фид «жив», но нужного ряда нет — это не жизнь.
                        if (
                            not missing_warned
                            and connected_at is not None
                            and time.time() - connected_at > FEED_ASSETS_WARN_AFTER_S
                        ):
                            missing = [
                                f"{a}@{w}s"
                                for a in EXPECTED_ASSETS for w in TWAP_WINDOWS
                                if (a, w) not in series_seen
                            ]
                            missing_warned = True
                            if missing:
                                log.warning(
                                    "Фид резолюции жив (%d чужих символов), но за "
                                    "%.0fс НЕТ тиков по рядам: %s — рынки этих "
                                    "окон останутся без цены и торговаться не "
                                    "будут",
                                    len(foreign_seen), FEED_ASSETS_WARN_AFTER_S,
                                    ", ".join(missing),
                                )
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
