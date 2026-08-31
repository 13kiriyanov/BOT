"""
Риск-менеджер. Единственный модуль, который имеет право сказать «стоп».

Философия: любая проверка, которая может провалиться, ДОЛЖНА провалиться
в сторону остановки торговли. Ошибочная остановка стоит упущенной прибыли;
ошибочное продолжение стоит депозита.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from .models import ZERO, MarketPosition, TargetMarket

log = logging.getLogger("polybot.risk")


class HaltReason(str, Enum):
    NONE = "none"
    DAILY_LOSS = "daily_loss_limit"
    NET_EXPOSURE = "max_net_exposure"
    NOTIONAL = "max_notional"
    HEARTBEAT = "heartbeat_timeout"
    REJECTS = "consecutive_rejects"
    MANUAL = "manual_stop"
    FATAL = "fatal_error"


@dataclass
class RiskState:
    halted: bool = False
    reason: HaltReason = HaltReason.NONE
    halted_at: float | None = None
    consecutive_rejects: int = 0
    session_start: float = field(default_factory=time.time)
    realized_pnl: Decimal = ZERO
    fills: int = 0
    volume: Decimal = ZERO


class RiskManager:
    """Проверки перед каждым котированием и глобальный kill switch."""

    def __init__(self, cfg) -> None:  # noqa: ANN001 - RiskSettings
        self.cfg = cfg
        self.state = RiskState()
        self._last_heartbeat = time.time()
        self._day_key = time.strftime("%Y-%m-%d")

    # ------------------------------------------------------- heartbeat

    def heartbeat(self) -> None:
        """Вызывается главным циклом на каждой итерации."""
        self._last_heartbeat = time.time()

    def heartbeat_expired(self) -> bool:
        return (time.time() - self._last_heartbeat) > self.cfg.heartbeat_timeout_s

    def seconds_since_heartbeat(self) -> float:
        return time.time() - self._last_heartbeat

    # ------------------------------------------------------------ halt

    def halt(self, reason: HaltReason, detail: str = "") -> None:
        if self.state.halted:
            return
        self.state.halted = True
        self.state.reason = reason
        self.state.halted_at = time.time()
        log.critical("ТОРГОВЛЯ ОСТАНОВЛЕНА: %s %s", reason.value, detail)

    def resume(self) -> None:
        """Ручное возобновление (только через рестарт или явную команду)."""
        log.warning("Торговля возобновлена после: %s", self.state.reason.value)
        self.state.halted = False
        self.state.reason = HaltReason.NONE
        self.state.consecutive_rejects = 0

    @property
    def is_halted(self) -> bool:
        return self.state.halted

    # ---------------------------------------------------------- учёт

    def record_fill(self, price: Decimal, size: Decimal) -> None:
        self.state.fills += 1
        self.state.volume += price * size
        self.state.consecutive_rejects = 0

    def record_realized(self, delta: Decimal) -> None:
        self.state.realized_pnl += delta
        self._check_daily_loss()

    def record_reject(self) -> None:
        self.state.consecutive_rejects += 1
        if self.state.consecutive_rejects >= self.cfg.max_consecutive_rejects:
            self.halt(
                HaltReason.REJECTS,
                f"{self.state.consecutive_rejects} отказов подряд",
            )

    def record_accept(self) -> None:
        self.state.consecutive_rejects = 0

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day_key:
            log.info(
                "Новый торговый день. PnL за %s: %.4f", self._day_key, self.state.realized_pnl
            )
            self._day_key = today
            self.state.realized_pnl = ZERO
            self.state.session_start = time.time()
            if self.state.reason == HaltReason.DAILY_LOSS:
                self.resume()

    def _check_daily_loss(self) -> None:
        self._roll_day()
        if self.state.realized_pnl <= -abs(self.cfg.daily_loss_limit):
            self.halt(
                HaltReason.DAILY_LOSS,
                f"PnL {self.state.realized_pnl} <= -{self.cfg.daily_loss_limit}",
            )

    # ------------------------------------------------- глобальные проверки

    def check_global(self, positions: dict[str, MarketPosition], open_orders: int) -> bool:
        """Проверки уровня портфеля. False => торговать нельзя."""
        self._roll_day()

        if self.state.halted:
            return False

        if self.heartbeat_expired():
            self.halt(
                HaltReason.HEARTBEAT,
                f"нет тика {self.seconds_since_heartbeat():.1f}s",
            )
            return False

        net = sum((p.net for p in positions.values()), ZERO)
        if abs(net) > self.cfg.max_net_exposure:
            self.halt(HaltReason.NET_EXPOSURE, f"net={net}")
            return False

        notional = sum((p.total_cost for p in positions.values()), ZERO)
        if notional > self.cfg.max_notional:
            self.halt(HaltReason.NOTIONAL, f"notional={notional}")
            return False

        if open_orders > self.cfg.max_open_orders:
            log.warning("Открытых ордеров %d > лимита %d", open_orders, self.cfg.max_open_orders)
            return False

        return True

    # ------------------------------------------------ проверки на рынок

    def can_quote_market(
        self,
        market: TargetMarket,
        book_stale: bool,
        price_stale: bool,
        spread: Decimal | None,
        depth: Decimal,
        max_spread: Decimal,
        min_depth: Decimal,
    ) -> tuple[bool, str]:
        """Можно ли котировать конкретный рынок. Возвращает (можно, причина)."""
        if price_stale:
            return False, "спот-фид протух"
        if book_stale:
            return False, "стакан протух"
        if market.seconds_left <= 0:
            return False, "рынок истёк"
        if spread is None:
            return False, "нет двусторонних котировок"
        if spread > max_spread:
            return False, f"спред {spread} > {max_spread}"
        if depth < min_depth:
            return False, f"глубина {depth} < {min_depth}"
        return True, ""

    def clamp_order_size(
        self,
        desired: Decimal,
        position_side_size: Decimal,
        current_net: Decimal,
        is_increasing_net: bool,
    ) -> Decimal:
        """
        Урезать размер ордера так, чтобы исполнение не пробило лимиты.
        Возвращает 0, если ордер ставить нельзя.
        """
        # Лимит на позицию по одной стороне.
        room_side = self.cfg.max_position_per_side - position_side_size
        if room_side <= 0:
            return ZERO
        size = min(desired, room_side)

        # Лимит на net exposure — только если ордер УВЕЛИЧИВАЕТ его модуль.
        if is_increasing_net:
            room_net = self.cfg.max_net_exposure - abs(current_net)
            if room_net <= 0:
                return ZERO
            size = min(size, room_net)

        return max(ZERO, size)

    def snapshot(self) -> dict:
        s = self.state
        return {
            "halted": s.halted,
            "reason": s.reason.value,
            "realized_pnl": float(s.realized_pnl),
            "fills": s.fills,
            "volume": float(s.volume),
            "uptime_s": round(time.time() - s.session_start, 1),
        }
