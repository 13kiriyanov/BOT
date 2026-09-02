"""
Замер окна опережения: насколько раньше стакана бот узнаёт о движении.

Directional-логика имеет смысл, только если фид резолюции (Chainlink TWAP)
показывает движение РАНЬШЕ, чем стакан YES соответствующего рынка его
отыгрывает. Иначе любое «модельное преимущество» уже в цене, и бот со своим
fair просто догоняет рынок. Без этого замера directional — вера, не факт.

Механика:
  * фид: по каждому ряду (актив, окно TWAP) держим опорную цену; когда
    накопленный ход |price/ref − 1| превышает порог, фиксируем событие
    движения (направление, момент t0) и переносим опору на текущую цену —
    так событие ловит и один резкий тик, и серию мелких в одну сторону;
  * стакан: по каждому рынку актива храним историю сдвигов mid лучших
    котировок YES (момент, направление) за lookback секунд;
  * задержка = момент первого сдвига стакана в ту же сторону − t0. Если
    стакан сдвинулся туда же ДО события фида (в пределах lookback и после
    предыдущего события), задержка отрицательная — стакан впереди; если за
    timeout секунд стакан так и не сдвинулся — тайм-аут: событие пишется с
    delay_ms=null и в статистику задержек не входит (считается отдельно).
    Новое событие фида, пришедшее пока прошлое ещё ждёт стакан, ничего не
    заводит: задержка меряется от ПЕРВОГО движения, не от последнего.

Событие lead_lag в trades.jsonl: актив, рынок, окно TWAP, размер хода в
базисных пунктах, направление, задержка в мс. В статусе — медиана и 10-й
процентиль задержки по активу: медиана > 0 — фид впереди на столько мс, это
и есть информационное окно; медиана ≤ 0 — стакан отыгрывает раньше, и
directional-логике не на чем стоять.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable

from .logging_setup import log_event

log = logging.getLogger("polybot.leadlag")

Sink = Callable[..., None]
"""Куда писать событие lead_lag; сигнатура log_event(event, **fields)."""

# Сколько последних задержек держим на актив для медианы и процентиля.
MAX_SAMPLES = 1000
# Сколько сдвигов стакана помним на рынок (за lookback их обычно единицы).
MAX_MOVES = 64


@dataclass(slots=True)
class _Pending:
    """Событие фида, ждущее ответного сдвига стакана."""

    t0: float
    direction: int
    move_bp: float
    deadline: float


@dataclass(slots=True)
class _MarketState:
    condition_id: str
    asset: str
    window_s: int
    slug: str
    last_mid: Decimal | None = None
    # История сдвигов mid: (момент, направление). Нужна, чтобы увидеть, что
    # стакан двинулся РАНЬШЕ фида.
    moves: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=MAX_MOVES))
    # Сдвиги не позже этого момента уже отнесены к прошлому событию фида.
    consumed_until: float = 0.0
    pending: _Pending | None = None


@dataclass(slots=True)
class _AssetStats:
    delays_ms: deque[float] = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))
    n_timeout: int = 0


class LeadLagTracker:
    """Сопоставляет движения фида с ответными сдвигами стакана YES."""

    def __init__(
        self,
        move_threshold: float,
        lookback_s: float,
        timeout_s: float,
        *,
        sink: Sink = log_event,
    ) -> None:
        if move_threshold <= 0:
            raise ValueError("порог хода фида должен быть > 0")
        if lookback_s < 0 or timeout_s <= 0:
            raise ValueError("lookback >= 0 и timeout > 0")
        self.move_threshold = float(move_threshold)
        self.lookback_s = float(lookback_s)
        self.timeout_s = float(timeout_s)
        self._sink = sink
        self._ref: dict[tuple[str, int | None], float] = {}
        self._markets: dict[str, _MarketState] = {}
        self._by_token: dict[str, str] = {}
        self._stats: dict[str, _AssetStats] = {}

    # ------------------------------------------------------------- рынки

    def register_market(
        self, condition_id: str, asset: str, window_s: int, slug: str, yes_token: str
    ) -> None:
        if condition_id in self._markets:
            return
        self._markets[condition_id] = _MarketState(condition_id, asset, window_s, slug)
        self._by_token[yes_token] = condition_id

    def forget_market(self, condition_id: str) -> None:
        state = self._markets.pop(condition_id, None)
        if state is None:
            return
        for token, cid in list(self._by_token.items()):
            if cid == condition_id:
                self._by_token.pop(token, None)

    # --------------------------------------------------------------- фид

    def on_spot_tick(self, asset: str, price: float, ts: float, window: int | None) -> None:
        """Тик фида: накопленный ход выше порога — событие движения."""
        if price <= 0:
            return
        key = (asset, window)
        ref = self._ref.get(key)
        if ref is None:
            self._ref[key] = price
            return
        move = price / ref - 1.0
        if abs(move) < self.move_threshold:
            return
        self._ref[key] = price
        direction = 1 if move > 0 else -1
        move_bp = move * 1e4

        for state in self._markets.values():
            if state.asset != asset:
                continue
            if window is not None and window != state.window_s:
                continue
            self._expire(state, ts)
            if state.pending is not None:
                # Стакан ещё не ответил на прошлое движение — меряем от него.
                continue
            prior = self._prior_move(state, direction, ts)
            if prior is not None:
                # Стакан двинулся туда же раньше фида: задержка отрицательная.
                state.consumed_until = ts
                self._resolve(state, direction, move_bp, (prior - ts) * 1000.0)
                continue
            state.pending = _Pending(ts, direction, move_bp, ts + self.timeout_s)

    def _prior_move(self, state: _MarketState, direction: int, ts: float) -> float | None:
        """Последний сдвиг стакана в направлении direction за lookback до ts."""
        since = max(ts - self.lookback_s, state.consumed_until)
        for t, d in reversed(state.moves):
            if t < since:
                break
            if d == direction and t <= ts:
                return t
        return None

    # ------------------------------------------------------------ стакан

    def on_book(
        self, token_id: str, best_bid: Decimal | None, best_ask: Decimal | None, ts: float
    ) -> None:
        """Обновление стакана YES: сдвиг mid лучших котировок — ответ рынка."""
        cid = self._by_token.get(token_id)
        if cid is None:
            return
        state = self._markets.get(cid)
        if state is None:
            return
        self._expire(state, ts)
        if best_bid is None or best_ask is None:
            return
        mid = (best_bid + best_ask) / 2
        prev, state.last_mid = state.last_mid, mid
        if prev is None or mid == prev:
            return
        direction = 1 if mid > prev else -1
        state.moves.append((ts, direction))

        pending = state.pending
        if pending is not None and direction == pending.direction:
            state.pending = None
            state.consumed_until = ts
            self._resolve(state, direction, pending.move_bp, (ts - pending.t0) * 1000.0)

    # ---------------------------------------------------------- события

    def _expire(self, state: _MarketState, now: float) -> None:
        pending = state.pending
        if pending is None or now <= pending.deadline:
            return
        state.pending = None
        state.consumed_until = pending.t0
        self._stats.setdefault(state.asset, _AssetStats()).n_timeout += 1
        self._sink(
            "lead_lag", asset=state.asset, market=state.slug, window_s=state.window_s,
            move_bp=round(pending.move_bp, 2), direction=pending.direction,
            delay_ms=None, timeout=True,
        )

    def _resolve(
        self, state: _MarketState, direction: int, move_bp: float, delay_ms: float
    ) -> None:
        self._stats.setdefault(state.asset, _AssetStats()).delays_ms.append(delay_ms)
        self._sink(
            "lead_lag", asset=state.asset, market=state.slug, window_s=state.window_s,
            move_bp=round(move_bp, 2), direction=direction,
            delay_ms=round(delay_ms, 1), timeout=False,
        )

    # ------------------------------------------------------------ сводка

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        out: dict[str, dict[str, float | int | None]] = {}
        for asset, stats in sorted(self._stats.items()):
            delays = list(stats.delays_ms)
            n = len(delays)
            median = statistics.median(delays) if n else None
            if n >= 10:
                p10: float | None = statistics.quantiles(delays, n=10)[0]
            elif n:
                p10 = min(delays)
            else:
                p10 = None
            out[asset] = {
                "n": n,
                "median_ms": round(median, 1) if median is not None else None,
                "p10_ms": round(p10, 1) if p10 is not None else None,
                "n_timeout": stats.n_timeout,
                "threshold_bp": round(self.move_threshold * 1e4, 2),
            }
        return out

    def summary_lines(self) -> list[str]:
        lines = []
        for asset, row in self.summary().items():
            if row["n"]:
                stat = (f"задержка стакана: медиана={row['median_ms']:+.0f} мс "
                        f"p10={row['p10_ms']:+.0f} мс (n={row['n']})")
            else:
                stat = "задержка стакана: замеров ещё нет"
            lines.append(
                f"LEAD-LAG {asset} | {stat} | тайм-аутов={row['n_timeout']} | "
                f"порог {row['threshold_bp']:.1f} bp"
            )
        return lines
