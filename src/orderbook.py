"""
Локальное зеркало стакана Polymarket, поддерживаемое через WebSocket.

Обрабатываем три типа событий market-канала:
  * book             — полный снапшот стакана (после подписки и после сбоев)
  * price_change     — дельта по уровням
  * best_bid_ask     — быстрый апдейт только лучших котировок

ВАЖНО про связность YES/NO. Матчинг-движок Polymarket трактует YES и NO как
дополняющие друг друга: покупка NO по p эквивалентна продаже YES по (1-p).
Поэтому стаканы зеркальны:  best_ask(NO) == 1 - best_bid(YES).
Никакого «тейкерского арбитража yes_ask + no_ask < 1» не существует.
Метод `implied_from_complement` позволяет восстановить сторону, если по одному
из токенов апдейт ещё не пришёл.
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal

from polymarket import AsyncSecureClient
from polymarket.streams import MarketSpec

from .models import Book, BookLevel

log = logging.getLogger("polybot.book")

ONE = Decimal("1")


class OrderBookManager:
    """Хранит книги всех подписанных токенов и держит WS-подписку."""

    def __init__(self) -> None:
        self._books: dict[str, Book] = {}
        self._token_ids: set[str] = set()
        self._resubscribe = asyncio.Event()
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ API

    def book(self, token_id: str) -> Book | None:
        return self._books.get(token_id)

    def get_or_create(self, token_id: str) -> Book:
        if token_id not in self._books:
            self._books[token_id] = Book(token_id=token_id)
        return self._books[token_id]

    def set_tokens(self, token_ids: set[str]) -> None:
        """Обновить набор подписок. Триггерит переподключение при изменении."""
        if token_ids != self._token_ids:
            added = token_ids - self._token_ids
            removed = self._token_ids - token_ids
            self._token_ids = set(token_ids)
            for t in removed:
                self._books.pop(t, None)
            log.info("Подписки: +%d / -%d, всего %d", len(added), len(removed), len(token_ids))
            self._resubscribe.set()

    def implied_from_complement(self, token_id: str, complement_id: str) -> Book | None:
        """
        Восстановить книгу токена из книги его дополнения.
        bid(X) = 1 - ask(Y), ask(X) = 1 - bid(Y), уровни разворачиваются.
        """
        other = self._books.get(complement_id)
        if other is None or (not other.bids and not other.asks):
            return None
        mirrored = Book(token_id=token_id, updated_at=other.updated_at)
        mirrored.bids = [BookLevel(ONE - lvl.price, lvl.size) for lvl in other.asks]
        mirrored.asks = [BookLevel(ONE - lvl.price, lvl.size) for lvl in other.bids]
        return mirrored

    def stop(self) -> None:
        self._stop.set()
        self._resubscribe.set()

    # --------------------------------------------------------- обработчики

    def _apply_snapshot(self, token_id: str, bids: list, asks: list, ts: float) -> None:
        book = self.get_or_create(token_id)
        book.bids = sorted(
            (BookLevel(Decimal(str(l.price)), Decimal(str(l.size))) for l in bids),
            key=lambda x: x.price,
            reverse=True,
        )
        book.asks = sorted(
            (BookLevel(Decimal(str(l.price)), Decimal(str(l.size))) for l in asks),
            key=lambda x: x.price,
        )
        book.updated_at = ts

    def _apply_delta(
        self, token_id: str, price: Decimal, size: Decimal, side: str, ts: float
    ) -> None:
        book = self.get_or_create(token_id)
        levels = book.bids if side.upper() in ("BUY", "BID") else book.asks
        reverse = side.upper() in ("BUY", "BID")

        for i, lvl in enumerate(levels):
            if lvl.price == price:
                if size <= 0:
                    levels.pop(i)
                else:
                    lvl.size = size
                break
        else:
            if size > 0:
                levels.append(BookLevel(price, size))
                levels.sort(key=lambda x: x.price, reverse=reverse)

        book.updated_at = ts

    def _handle_event(self, event) -> None:  # noqa: ANN001 - union типов SDK
        etype = getattr(event, "type", "")
        payload = getattr(event, "payload", None)
        if payload is None:
            return
        now = time.time()

        if etype == "book":
            token = getattr(payload, "token_id", None)
            if token:
                self._apply_snapshot(
                    token, payload.bids or [], payload.asks or [], now
                )

        elif etype == "price_change":
            for change in getattr(payload, "price_changes", []) or []:
                token = getattr(change, "token_id", None)
                if not token:
                    continue
                self._apply_delta(
                    token,
                    Decimal(str(change.price)),
                    Decimal(str(change.size)),
                    str(change.side),
                    now,
                )

        elif etype == "best_bid_ask":
            token = getattr(payload, "token_id", None)
            if not token:
                return
            book = self.get_or_create(token)
            bb, ba = getattr(payload, "best_bid", None), getattr(payload, "best_ask", None)
            # Если полного снапшота ещё нет — создаём синтетический топ уровня.
            if bb is not None and not book.bids:
                book.bids = [BookLevel(Decimal(str(bb)), Decimal("0"))]
            if ba is not None and not book.asks:
                book.asks = [BookLevel(Decimal(str(ba)), Decimal("0"))]
            book.updated_at = now

        elif etype in ("market_resolved", "tick_size_change"):
            log.info("Событие рынка: %s", etype)

    # ------------------------------------------------------------- runner

    async def run(self, client: AsyncSecureClient) -> None:
        """Держит подписку на market-канал, переподключаясь при смене токенов."""
        backoff = 1.0
        while not self._stop.is_set():
            if not self._token_ids:
                await asyncio.sleep(1.0)
                continue

            tokens = list(self._token_ids)
            self._resubscribe.clear()
            try:
                log.info("Подписка на стаканы: %d токенов", len(tokens))
                # subscribe() — корутина: await возвращает handle-CM.
                async with await client.subscribe(
                    MarketSpec(token_ids=tokens)
                ) as stream:
                    backoff = 1.0
                    resub = asyncio.create_task(self._resubscribe.wait())
                    stream_iter = stream.__aiter__()
                    while not self._stop.is_set():
                        nxt = asyncio.create_task(stream_iter.__anext__())
                        done, _ = await asyncio.wait(
                            {nxt, resub}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if resub in done:
                            nxt.cancel()
                            log.info("Набор токенов изменился — переподписка")
                            break
                        try:
                            self._handle_event(nxt.result())
                        except StopAsyncIteration:
                            break
                    resub.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("Market stream упал: %s. Retry через %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 20.0)
