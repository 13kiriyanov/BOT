"""
Генератор котировок: two-sided market making == pair accumulation.

КЛЮЧЕВАЯ ИДЕЯ
-------------
Мы ставим BUY YES по цене `a` и BUY NO по цене `b`. Если обе ноги пассивно
исполнились — у нас полная пара, которая стоит ровно $1 при любом исходе.
Прибыль на паре = 1 - (a + b).

При этом покупка NO по `b` тождественна продаже YES по `(1 - b)`. То есть

    bid_YES  = a
    ask_YES  = 1 - b
    spread   = (1 - b) - a = 1 - (a + b)

«Накопление пар с суммой < 0.985» и «котирование с полуспредом 0.75 цента
вокруг fair value» — это буквально одно и то же действие. Поэтому один
модуль делает и то, и другое: мы задаём целевую стоимость пары, а из неё
выводится спред.

    half_spread = (1 - target_pair_cost) / 2
    bid_YES = r - half_spread
    bid_NO  = (1 - r) - half_spread

где r — резервная цена (fair value, сдвинутая инвентарём).

ИСТОЧНИКИ ПРИБЫЛИ (по убыванию значимости на практике)
 1. Захват спреда на обеих ногах.
 2. Мгновенный merge пары в USDC -> высокая оборачиваемость капитала.
 3. Liquidity rewards Polymarket за котировки внутри reward-спреда.
 4. Directional edge — самый маленький и самый рискованный компонент.
"""

from __future__ import annotations

import logging
from decimal import ROUND_DOWN, ROUND_UP, Decimal

from .models import ONE, ZERO, Book, FairValue, MarketPosition, Quote, TargetMarket

log = logging.getLogger("polybot.quote")


def round_to_tick(price: Decimal, tick: Decimal, mode: str = ROUND_DOWN) -> Decimal:
    """Округлить цену к сетке тиков."""
    if tick <= 0:
        return price
    return (price / tick).quantize(Decimal("1"), rounding=mode) * tick


def clamp_price(price: Decimal, tick: Decimal) -> Decimal:
    """Загнать цену в допустимый диапазон [tick, 1 - tick]."""
    return max(tick, min(ONE - tick, price))


class QuoteGenerator:
    """Считает желаемые котировки для одного рынка."""

    def __init__(self, strat_cfg, risk_cfg) -> None:  # noqa: ANN001
        self.s = strat_cfg
        self.r = risk_cfg

    # ------------------------------------------------- резервная цена

    def reservation_price(
        self, fair: Decimal, position: MarketPosition, fv: FairValue
    ) -> Decimal:
        """
        Сдвиг fair value от инвентаря (Avellaneda-Stoikov-lite).

        Лонг YES (net > 0) -> сдвигаем цены вниз: покупаем YES дешевле,
        а NO (== продажа YES) — дороже. Это естественно разгружает позицию.
        """
        net = position.net
        max_net = self.r.max_net_exposure
        if max_net <= 0:
            return fair

        skew = self.s.inventory_skew_coef * (net / max_net)

        # Directional tilt: если модель уверенно видит edge и это разрешено,
        # позволяем резервной цене немного сместиться В СТОРОНУ модели,
        # частично компенсируя inventory skew.
        if self.s.allow_directional and abs(fv.edge) >= self.s.directional_min_edge:
            # Разрешаем только пока net не упёрся в directional-лимит.
            room = self.s.directional_max_net - abs(net)
            if room > 0 and (fv.edge > 0) == (net <= 0):
                tilt = fv.edge * fv.confidence * Decimal("0.5")
                tilt = max(Decimal("-0.02"), min(Decimal("0.02"), tilt))
                skew -= tilt

        return fair - skew

    # ------------------------------------------------------- размеры

    def _size_for(
        self,
        market: TargetMarket,
        position: MarketPosition,
        outcome: str,
        fv: FairValue,
    ) -> Decimal:
        """Размер ордера с учётом инвентаря, времени и directional-наклона."""
        size = self.s.order_size

        # У экспирации сокращаем размер: adverse selection растёт нелинейно.
        left = market.seconds_left
        if left < 45:
            size *= Decimal("0.35")
        elif left < 120:
            size *= Decimal("0.7")

        # Если эта сторона уже перегружена — уменьшаем.
        side_size = position.yes_size if outcome == "YES" else position.no_size
        util = side_size / self.r.max_position_per_side if self.r.max_position_per_side > 0 else ZERO
        if util > Decimal("0.6"):
            size *= Decimal("0.5")

        # Directional: увеличиваем сторону, которую поддерживает модель.
        if self.s.allow_directional and abs(fv.edge) >= self.s.directional_min_edge:
            favours_yes = fv.edge > 0
            if (outcome == "YES") == favours_yes:
                size *= Decimal("1.25")
            else:
                size *= Decimal("0.8")

        # Reward-программа часто требует минимальный размер ордера.
        if market.rewards_min_size and size < market.rewards_min_size:
            size = market.rewards_min_size

        size = max(size, market.min_order_size)
        return size.quantize(Decimal("0.01"))

    # ------------------------------------------------- цены с учётом книги

    def _place_bid(
        self,
        raw_price: Decimal,
        book: Book | None,
        tick: Decimal,
        max_price: Decimal,
    ) -> Decimal | None:
        """
        Финальная цена бида.
        - Никогда не пересекаем ask (post_only отклонит такой ордер).
        - Если можно улучшить лучший бид на тик, не превысив нашу планку —
          делаем это: очередь в стакане важнее пары десятых цента.
        """
        price = round_to_tick(raw_price, tick, ROUND_DOWN)

        if book is not None:
            best_ask = book.best_ask
            if best_ask is not None:
                price = min(price, best_ask - tick)
            best_bid = book.best_bid
            if best_bid is not None:
                improved = best_bid + tick
                # Улучшаем книгу, только если это всё ещё внутри нашей планки.
                if best_bid < price and improved <= max_price:
                    price = min(price, max(price, improved)) if improved <= price else price
                # Если наша цена уже лучше лучшего бида — можно отступить
                # к best_bid + tick и сохранить edge, оставаясь первым в очереди.
                if price > best_bid + tick:
                    price = best_bid + tick

        price = clamp_price(price, tick)
        price = min(price, round_to_tick(max_price, tick, ROUND_DOWN))
        return price if price > 0 else None

    # ------------------------------------------------------------ главное

    def build_quotes(
        self,
        market: TargetMarket,
        fv: FairValue,
        position: MarketPosition,
        yes_book: Book | None,
        no_book: Book | None,
    ) -> list[Quote]:
        """Сформировать желаемые котировки (BUY YES и BUY NO)."""
        tick = market.tick_size

        r = self.reservation_price(fv.fair, position, fv)
        r = clamp_price(r, tick)

        # КОМИССИЯ. Пара приносит 1 - (a + b) валовых, но комиссия рынка
        # списывается с каждой ноги, поэтому чистая маржа меньше. Оцениваем
        # её по резервной цене, чтобы сразу раздвинуть спред; точная проверка
        # по финальным ценам — ниже, перед выдачей котировок.
        fee_pair = market.fee_per_pair(r, ONE - r)

        # Планка: полная стоимость пары (биды + комиссия) не должна
        # превысить max_pair_cost. Значит на сами биды остаётся меньше.
        budget = self.s.max_pair_cost - fee_pair
        if budget <= 0:
            log.debug(
                "[%s] Комиссия %s съедает всю планку %s — не котирую",
                market.slug, fee_pair, self.s.max_pair_cost,
            )
            return []

        half = (ONE - (self.s.target_pair_cost - fee_pair)) / 2

        # Полуспред не может быть меньше минимума в тиках.
        min_half = tick * self.s.min_half_spread_ticks
        half = max(half, min_half)

        # Если у рынка есть reward-программа, не выходим за её спред —
        # иначе теряем существенную часть реального PnL.
        if market.rewards_max_spread:
            # rewards_max_spread обычно задан в центах отклонения от mid.
            reward_half = market.rewards_max_spread / 100
            if reward_half > 0:
                half = min(half, reward_half)

        raw_yes_bid = r - half
        raw_no_bid = (ONE - r) - half

        # Распределяем допустимый бюджет пропорционально.
        max_yes = budget - raw_no_bid
        max_no = budget - raw_yes_bid

        yes_price = self._place_bid(raw_yes_bid, yes_book, tick, max_yes)
        no_price = self._place_bid(raw_no_bid, no_book, tick, max_no)

        quotes: list[Quote] = []
        if yes_price is None or no_price is None:
            return quotes

        # ФИНАЛЬНАЯ ЗАЩИТА: если после всех округлений, подгонок под книгу и
        # комиссий полная стоимость пары стала невыгодной — не котируем вообще.
        # Здесь комиссия считается уже по фактическим ценам, а не по оценке.
        cap = self.s.max_pair_cost
        pair_cost = yes_price + no_price + market.fee_per_pair(yes_price, no_price)
        if pair_cost >= cap:
            log.debug(
                "[%s] Пара невыгодна: %s + %s + комиссия = %s >= %s",
                market.slug, yes_price, no_price, pair_cost, cap,
            )
            # Пробуем ужать обе стороны на тик.
            yes_price -= tick
            no_price -= tick
            if yes_price <= 0 or no_price <= 0:
                return quotes
            pair_cost = yes_price + no_price + market.fee_per_pair(yes_price, no_price)
            if pair_cost >= cap:
                return quotes

        yes_size = self._size_for(market, position, "YES", fv)
        no_size = self._size_for(market, position, "NO", fv)

        quotes.append(
            Quote(market.yes_token_id, "YES", "BUY", yes_price, yes_size)
        )
        quotes.append(
            Quote(market.no_token_id, "NO", "BUY", no_price, no_size)
        )
        return quotes

    def build_unwind_quotes(
        self, market: TargetMarket, position: MarketPosition, book: Book | None
    ) -> list[Quote]:
        """
        Аварийная разгрузка: продать избыточную одностороннюю позицию.
        Используется, когда net превысил лимит или рынок скоро истечёт,
        а пары не сложились.
        """
        net = position.net
        if abs(net) < market.min_order_size:
            return []

        tick = market.tick_size
        # Продаём ту сторону, которой у нас больше.
        outcome = "YES" if net > 0 else "NO"
        token = market.token_for(outcome)  # type: ignore[arg-type]
        size = min(abs(net), self.s.order_size * 2)

        b = book
        if b is None or b.best_bid is None:
            return []
        # Агрессивно, но не в рынок: встаём на лучший бид.
        price = clamp_price(round_to_tick(b.best_bid, tick, ROUND_UP), tick)

        log.info("[%s] Разгрузка %s %s @ %s", market.slug, outcome, size, price)
        return [Quote(token, outcome, "SELL", price, size)]  # type: ignore[arg-type]
