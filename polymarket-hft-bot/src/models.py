"""Внутренние модели данных бота (не путать с моделями SDK)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

Side = Literal["BUY", "SELL"]
Outcome = Literal["YES", "NO"]

ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(slots=True)
class TargetMarket:
    """Краткосрочный Up/Down рынок, который бот котирует."""

    condition_id: str
    slug: str
    question: str
    yes_token_id: str
    no_token_id: str
    end_ts: float                      # unix-время экспирации
    tick_size: Decimal
    min_order_size: Decimal
    neg_risk: bool
    asset: str                         # 'BTC' | 'ETH'
    strike: Decimal | None = None      # цена базового актива на открытии окна
    fees_enabled: bool = False
    rewards_max_spread: Decimal | None = None
    rewards_min_size: Decimal | None = None

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.end_ts - time.time())

    def token_for(self, outcome: Outcome) -> str:
        return self.yes_token_id if outcome == "YES" else self.no_token_id

    def other(self, outcome: Outcome) -> Outcome:
        return "NO" if outcome == "YES" else "YES"


@dataclass(slots=True)
class BookLevel:
    price: Decimal
    size: Decimal


@dataclass(slots=True)
class Book:
    """Локальное состояние стакана одного токена."""

    token_id: str
    bids: list[BookLevel] = field(default_factory=list)  # по убыванию цены
    asks: list[BookLevel] = field(default_factory=list)  # по возрастанию цены
    updated_at: float = 0.0

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return (b + a) / 2

    @property
    def spread(self) -> Decimal | None:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return a - b

    def depth(self, side: Literal["bids", "asks"], levels: int = 3) -> Decimal:
        """Суммарный объём первых N уровней."""
        book = self.bids if side == "bids" else self.asks
        return sum((lvl.size for lvl in book[:levels]), ZERO)

    def is_stale(self, timeout_s: float) -> bool:
        return (time.time() - self.updated_at) > timeout_s

    def microprice(self) -> Decimal | None:
        """Взвешенный по объёму mid — лучше предсказывает следующий трейд."""
        if not self.bids or not self.asks:
            return None
        bp, bs = self.bids[0].price, self.bids[0].size
        ap, asz = self.asks[0].price, self.asks[0].size
        total = bs + asz
        if total <= 0:
            return self.mid
        return (bp * asz + ap * bs) / total


@dataclass(slots=True)
class Quote:
    """Желаемый лимитный ордер."""

    token_id: str
    outcome: Outcome
    side: Side
    price: Decimal
    size: Decimal

    def key(self) -> tuple[str, Side]:
        return (self.token_id, self.side)


@dataclass(slots=True)
class LiveOrder:
    """Ордер, реально стоящий в стакане."""

    order_id: str
    token_id: str
    side: Side
    price: Decimal
    original_size: Decimal
    size_matched: Decimal = ZERO
    created_at: float = field(default_factory=time.time)

    @property
    def remaining(self) -> Decimal:
        return max(ZERO, self.original_size - self.size_matched)


@dataclass(slots=True)
class MarketPosition:
    """Позиция бота по одному рынку."""

    condition_id: str
    yes_size: Decimal = ZERO
    no_size: Decimal = ZERO
    yes_cost: Decimal = ZERO   # суммарно потрачено USDC на YES
    no_cost: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    merged_pairs: Decimal = ZERO

    @property
    def complete_pairs(self) -> Decimal:
        """Полные пары YES+NO — гарантированно стоят $1 каждая."""
        return min(self.yes_size, self.no_size)

    @property
    def net(self) -> Decimal:
        """Направленная экспозиция: >0 — лонг YES, <0 — лонг NO."""
        return self.yes_size - self.no_size

    @property
    def total_cost(self) -> Decimal:
        return self.yes_cost + self.no_cost

    def pair_cost_basis(self) -> Decimal | None:
        """Средняя себестоимость одной полной пары."""
        pairs = self.complete_pairs
        if pairs <= 0 or self.yes_size <= 0 or self.no_size <= 0:
            return None
        avg_yes = self.yes_cost / self.yes_size
        avg_no = self.no_cost / self.no_size
        return avg_yes + avg_no

    def apply_fill(self, outcome: Outcome, side: Side, price: Decimal, size: Decimal) -> None:
        """Обновление позиции по факту исполнения."""
        signed = size if side == "BUY" else -size
        if outcome == "YES":
            if side == "BUY":
                self.yes_cost += price * size
            else:
                avg = self.yes_cost / self.yes_size if self.yes_size > 0 else ZERO
                self.yes_cost -= avg * size
                self.realized_pnl += (price - avg) * size
            self.yes_size += signed
        else:
            if side == "BUY":
                self.no_cost += price * size
            else:
                avg = self.no_cost / self.no_size if self.no_size > 0 else ZERO
                self.no_cost -= avg * size
                self.realized_pnl += (price - avg) * size
            self.no_size += signed

    def apply_merge(self, size: Decimal) -> None:
        """Merge полной пары: YES+NO -> $1 USDC. Фиксирует прибыль пары."""
        size = min(size, self.complete_pairs)
        if size <= 0:
            return
        avg_yes = self.yes_cost / self.yes_size if self.yes_size > 0 else ZERO
        avg_no = self.no_cost / self.no_size if self.no_size > 0 else ZERO
        cost = (avg_yes + avg_no) * size
        self.realized_pnl += size * ONE - cost
        self.yes_cost -= avg_yes * size
        self.no_cost -= avg_no * size
        self.yes_size -= size
        self.no_size -= size
        self.merged_pairs += size


@dataclass(slots=True)
class FairValue:
    """Результат работы модели справедливой цены YES."""

    fair: Decimal              # итоговая справедливая вероятность YES
    model_prob: Decimal        # чистая оценка модели
    market_mid: Decimal        # рыночный mid
    edge: Decimal              # model - market
    sigma_annual: Decimal      # оценка волатильности
    confidence: Decimal        # 0..1, насколько доверяем модели
