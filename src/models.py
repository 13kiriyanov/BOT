"""Внутренние модели данных бота (не путать с моделями SDK)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Literal

Side = Literal["BUY", "SELL"]
Outcome = Literal["YES", "NO"]

ZERO = Decimal("0")
ONE = Decimal("1")

# CTF-позиции на Polygon хранятся как ERC-1155 с 6 знаками после запятой.
# merge_positions() в SDK ждёт amount ИМЕННО в этих базовых единицах и
# сверяет его с балансом токенов on-chain, а не в shares.
POSITION_DECIMALS = Decimal("1000000")


# Минимальный TTL GTD-ордера. SDK отклоняет expiration ближе, чем
# now + 180 секунд (_MIN_EXPIRATION_BUFFER_S в polymarket/_internal/actions/
# orders/limit.py), плюс 30 секунд запаса на латентность и рассинхрон часов.
# TTL короче этого минимума означает, что ни один GTD-ордер не подпишется.
MIN_GTD_TTL_S = 180 + 30


def shares_to_base_units(shares: Decimal) -> int:
    """Перевести shares в базовые единицы ERC-1155 (округление вниз)."""
    if shares <= 0:
        return 0
    return int((shares * POSITION_DECIMALS).to_integral_value(rounding=ROUND_DOWN))


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
    # Unix-время НАЧАЛА окна (из хвоста слага btc-updown-5m-<ts>). None —
    # неизвестно. Движок использует его, чтобы записать спот ровно в момент
    # открытия (самый надёжный источник страйка) и чтобы не калибровать
    # страйк по рыночному mid до старта окна.
    start_ts: float | None = None
    # Окно потока Chainlink TWAP, по которому резолвится ЭТОТ рынок (сек):
    # 5-минутные — 30, 15-минутные и 4-часовые — 60 (анонс Polymarket от
    # 7 августа 2026; точный ряд назван в description рынка). Модель, страйк
    # и реализованная часть TWAP обязаны браться из ряда ЭТОГО окна.
    twap_window_s: int = 60
    # Параметры программы ликвидити-наград (Gamma: clobRewards). None —
    # рынок не в программе или данные не отданы.
    rewards_daily_rate: Decimal | None = None
    rewards_end_date: str | None = None
    fees_enabled: bool = False
    # Ставка комиссии, которую платим МЫ, и её экспонента (см. fee_per_share).
    # fee_rate == 0 означает «для нас этот рынок бесплатный»: либо комиссий
    # нет вовсе, либо они taker-only, а мы всегда мейкер (post_only=True).
    fee_rate: Decimal = ZERO
    fee_exponent: Decimal = ZERO
    rewards_max_spread: Decimal | None = None
    rewards_min_size: Decimal | None = None

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.end_ts - time.time())

    def token_for(self, outcome: Outcome) -> str:
        return self.yes_token_id if outcome == "YES" else self.no_token_id

    def other(self, outcome: Outcome) -> Outcome:
        return "NO" if outcome == "YES" else "YES"

    # ------------------------------------------------------------ комиссии

    def fee_per_share(self, price: Decimal) -> Decimal:
        """
        Комиссия за одну share, купленную по цене `price`, в USDC.

        Форма взята из самого SDK (adjust_buy_amount_for_fees):

            effective_rate = rate * (p * (1 - p)) ** exponent
            fee_usdc       = shares * effective_rate

        Комиссия максимальна у 0.50 и падает к краям книги — то есть бьёт
        ровно по тем ценам, вокруг которых мы и котируем.
        """
        if self.fee_rate <= 0:
            return ZERO
        base = price * (ONE - price)
        if base <= 0:
            return ZERO
        if self.fee_exponent == 0:
            return self.fee_rate
        return self.fee_rate * (base ** self.fee_exponent)

    def fee_for(self, price: Decimal, size: Decimal) -> Decimal:
        """Комиссия за исполнение `size` shares по цене `price`."""
        return self.fee_per_share(price) * size

    def fee_per_pair(self, yes_price: Decimal, no_price: Decimal) -> Decimal:
        """
        Комиссия за сборку одной полной пары — обе ноги вместе.

        Именно её надо вычесть из маржи пары: пара приносит
        1 - (a + b) валовых, а чистыми — 1 - (a + b) - fee_per_pair.
        """
        return self.fee_per_share(yes_price) + self.fee_per_share(no_price)


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
class Fill:
    """
    Исполнение НАШЕГО ордера, приведённое к нашей перспективе.

    В событии trade user-канала верхнеуровневые side/price описывают
    тейкера; наша сторона сделки (мы всегда post_only-мейкер) лежит в
    maker_orders. execution.py разбирает событие и отдаёт движку уже
    готовый Fill — чтобы перспектива не перепутывалась дальше по коду.
    """

    trade_id: str
    condition_id: str
    token_id: str
    side: Side
    price: Decimal
    size: Decimal
    fee_rate_bps: Decimal | None = None
    ts: float = field(default_factory=time.time)


@dataclass(slots=True)
class Quote:
    """Желаемый лимитный ордер."""

    token_id: str
    outcome: Outcome
    side: Side
    price: Decimal
    size: Decimal
    # Уровень лестницы: 0 — ближайший к mid. Часть ключа дедупликации:
    # без него второй уровень той же стороны считался бы «тем же ордером»
    # и лестница схлопывалась бы в одну котировку.
    level: int = 0

    def key(self) -> tuple[str, Side, int]:
        return (self.token_id, self.side, self.level)


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
    level: int = 0                     # уровень лестницы (см. Quote.level)

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
    fees_paid: Decimal = ZERO          # комиссии рынка, уплаченные на филлах
    merge_costs: Decimal = ZERO        # газ, потраченный на merge пар

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

    def apply_fill(
        self,
        outcome: Outcome,
        side: Side,
        price: Decimal,
        size: Decimal,
        fee: Decimal = ZERO,
    ) -> None:
        """
        Обновление позиции по факту исполнения.

        Комиссия покупки капитализируется в себестоимость, а не списывается
        в PnL сразу: тогда pair_cost_basis() автоматически показывает полную
        цену пары, и прибыль merge считается уже чистой от комиссий.
        """
        signed = size if side == "BUY" else -size
        self.fees_paid += fee
        if outcome == "YES":
            if side == "BUY":
                self.yes_cost += price * size + fee
            else:
                avg = self.yes_cost / self.yes_size if self.yes_size > 0 else ZERO
                self.yes_cost -= avg * size
                self.realized_pnl += (price - avg) * size - fee
            self.yes_size += signed
        else:
            if side == "BUY":
                self.no_cost += price * size + fee
            else:
                avg = self.no_cost / self.no_size if self.no_size > 0 else ZERO
                self.no_cost -= avg * size
                self.realized_pnl += (price - avg) * size - fee
            self.no_size += signed

    def apply_merge(self, size: Decimal, gas_cost: Decimal = ZERO) -> None:
        """
        Merge полной пары: YES+NO -> $1 USDC. Фиксирует прибыль пары.

        `gas_cost` — издержки транзакции целиком за merge (не за пару).
        Merge на Polygon не бесплатен, и на мелких пачках газ съедает всю
        маржу; учитываем его здесь, чтобы PnL не был приукрашен.
        """
        size = min(size, self.complete_pairs)
        if size <= 0:
            return
        avg_yes = self.yes_cost / self.yes_size if self.yes_size > 0 else ZERO
        avg_no = self.no_cost / self.no_size if self.no_size > 0 else ZERO
        cost = (avg_yes + avg_no) * size
        self.realized_pnl += size * ONE - cost - gas_cost
        self.merge_costs += gas_cost
        self.yes_cost -= avg_yes * size
        self.no_cost -= avg_no * size
        self.yes_size -= size
        self.no_size -= size
        self.merged_pairs += size

    def correct_side(
        self, outcome: Outcome, exchange_size: Decimal, exchange_avg: Decimal | None
    ) -> None:
        """
        Скорректировать одну сторону к данным биржи (периодическая сверка).

        Коррекция — не сделка: realized_pnl не трогаем. Средняя цена
        сохраняется наша, если сторона была ненулевой; иначе берём среднюю
        биржи; иначе 1.0 за share — верхняя граница, завышающая нотионал,
        то есть ошибающаяся в сторону более строгих лимитов.
        """
        new_size = max(ZERO, exchange_size)
        cur_size = self.yes_size if outcome == "YES" else self.no_size
        cur_cost = self.yes_cost if outcome == "YES" else self.no_cost
        if cur_size > 0:
            avg = cur_cost / cur_size
        elif exchange_avg and exchange_avg > 0:
            avg = exchange_avg
        else:
            avg = ONE
        if outcome == "YES":
            self.yes_size = new_size
            self.yes_cost = avg * new_size
        else:
            self.no_size = new_size
            self.no_cost = avg * new_size

    def apply_recovered(
        self, outcome: Outcome, size: Decimal, avg_price: Decimal | None
    ) -> None:
        """
        Внести в учёт позицию, найденную на бирже при старте бота.

        realized_pnl намеренно не трогаем: PnL прошлой сессии — не наш
        результат, а дневной лимит убытка должен считаться от старта.
        Если биржа не отдала среднюю цену, берём 1.0 USDC за share — это
        верхняя граница себестоимости. Она завышает нотионал, поэтому
        риск-лимиты сработают раньше, а не позже.
        """
        if size <= 0:
            return
        avg = avg_price if avg_price and avg_price > 0 else ONE
        if outcome == "YES":
            self.yes_size += size
            self.yes_cost += avg * size
        else:
            self.no_size += size
            self.no_cost += avg * size


# Как биржа называет стороны бинарного рынка. Up/Down-серии подписывают
# исходы словами 'Up' и 'Down', а не 'Yes' и 'No'.
_YES_LABELS = frozenset({"YES", "UP"})
_NO_LABELS = frozenset({"NO", "DOWN"})


def outcome_label(position) -> Outcome | None:  # noqa: ANN001 - модель SDK
    """Сторона позиции по ответу API, если её вообще можно определить."""
    raw = getattr(position, "outcome", None)
    if raw is not None:
        text = str(raw).strip().upper()
        if text in _YES_LABELS:
            return "YES"
        if text in _NO_LABELS:
            return "NO"
    index = getattr(position, "outcome_index", None)
    if index is not None:
        try:
            return "YES" if int(index) == 0 else "NO"
        except (TypeError, ValueError):
            return None
    return None


@dataclass(slots=True)
class RecoveredPosition:
    """
    Позиция, найденная на бирже при старте бота.

    Бот не единственный источник правды: после рестарта (или падения) на
    кошельке остаются shares прошлой сессии. Пока они не заведены в учёт,
    все риск-лимиты считаются от нуля — то есть от неверной базы.
    """

    condition_id: str
    token_id: str
    size: Decimal
    avg_price: Decimal | None = None
    # Сторона, если её удалось определить по ответу API. None => неизвестна,
    # определим позже по token_id, когда discovery найдёт этот рынок.
    outcome: Outcome | None = None
    title: str = ""
    # Рынок резолвлен, позиция ждёт redeem: это уже требование к USDC,
    # а не рыночный риск.
    redeemable: bool = False

    @classmethod
    def from_api(cls, position) -> RecoveredPosition | None:  # noqa: ANN001
        """
        Разобрать позицию из ответа SDK. None — брать в учёт нечего.

        Читаем через getattr: формат ответа биржи меняется без нашего
        участия, и падать на незнакомом поле в момент старта — худшее,
        что может сделать риск-контур.
        """
        condition_id = str(getattr(position, "condition_id", "") or "")
        raw_size = getattr(position, "size", None)
        size = Decimal(str(raw_size)) if raw_size is not None else ZERO
        if not condition_id or size <= 0:
            return None
        raw_avg = getattr(position, "avg_price", None)
        return cls(
            condition_id=condition_id,
            token_id=str(getattr(position, "token_id", "") or ""),
            size=size,
            avg_price=Decimal(str(raw_avg)) if raw_avg is not None else None,
            outcome=outcome_label(position),
            title=str(getattr(position, "title", "") or ""),
            redeemable=bool(getattr(position, "redeemable", False)),
        )


@dataclass(slots=True)
class FairValue:
    """Результат работы модели справедливой цены YES."""

    fair: Decimal              # итоговая справедливая вероятность YES
    model_prob: Decimal        # чистая оценка модели
    market_mid: Decimal        # рыночный mid
    edge: Decimal              # model - market
    sigma_annual: Decimal      # оценка волатильности
    confidence: Decimal        # 0..1, насколько доверяем модели
