"""
Mark-out: измерение adverse selection по факту, а не по вере.

Симулятор показывает, что прибыльность стратегии целиком определяется долей
информированного потока, который нас исполняет. В проде это число ниоткуда
не берётся само — его надо мерить. Mark-out и есть эта мера: на каждый филл
через 5, 30 и 120 секунд читается mid соответствующего токена и считается

    mark-out = знак стороны * (mid(t + tau) - цена филла) * размер,

знак +1 для покупки, -1 для продажи. Систематически отрицательный mark-out
означает, что нас исполняют ровно перед движением против нас — та самая
toxicity из симулятора, измеренная на реальном потоке.

Филлы разделяются на корзины с РАЗНОЙ экономикой — смешивать их нельзя:

 * paired — часть филла, вошедшая в полную пару YES+NO. Пара стоит $1 при
   любом исходе, движение mid после входа на её PnL не влияет. Mark-out
   этой корзины — качество момента входа, а не прямой риск.
 * solo — часть, оставшаяся односторонней. Здесь mark-out бьёт прямо в
   PnL; это главная корзина: если она стабильно отрицательна и глубже
   маржи пары, стратегия убыточна независимо от числа собранных пар.
 * unwind — продажи (аварийная разгрузка перекоса): качество исполнения
   выхода, к сборке пар отношения не имеет.

Сопоставление филлов в пары — FIFO по каждому рынку. Классификация
фиксируется в момент записи события, то есть после последнего горизонта:
нога, нашедшая пару позже, чем через 120 секунд, останется в solo. Для окон
5-15 минут с циклом котирования 0.35 с это осознанное упрощение.

Ограничение: mid читается из локального зеркала стаканов. Если рынок к
горизонту истёк или выпал из подписки (например, филл случился за минуту до
экспирации), mid недоступен — в событии пишется null, в сводке такие точки
считаются отдельным счётчиком n_miss и в среднее не входят.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable

from .logging_setup import log_event
from .models import ZERO, Fill, Outcome

log = logging.getLogger("polybot.markout")

# Горизонты замера, секунды от момента филла.
DEFAULT_HORIZONS_S: tuple[float, ...] = (5.0, 30.0, 120.0)

# Максимум одновременно ожидающих замеров. Больше — что-то пошло не так
# (штормовое исполнение), и лучше потерять точки метрики, чем память.
MAX_PENDING = 2000

# Незапаренные остатки филлов старше этого возраста вычищаются из FIFO:
# их рынок давно истёк, парой они уже не станут.
PRUNE_AGE_S = 1800.0

MidSource = Callable[[str, str], "Decimal | None"]
"""(token_id, complement_token_id) -> mid токена или None, если книги нет."""

Sink = Callable[..., None]
"""Куда писать событие markout; сигнатура log_event(event, **fields)."""

BUCKET_PAIRED = "paired"
BUCKET_SOLO = "solo"
BUCKET_UNWIND = "unwind"

# Горизонт РЫНКА (длительность окна) из слага: btc-updown-5m-<ts> -> "5m".
# Разрез сводки по нему показывает на живых филлах, где экономика лучше —
# на 5-минутных или 15-минутных рынках; симулятор на этот вопрос отвечает
# только модельно (README, раздел про лестницу и наблюдаемого мейкера).
MARKET_HORIZON_RE = re.compile(r"-(\d+[mh])-\d{10}$")
MARKET_HORIZON_UNKNOWN = "?"


def horizon_label_from_slug(slug: str) -> str:
    """'btc-updown-5m-1788276000' -> '5m'; без распознаваемого хвоста — '?'."""
    m = MARKET_HORIZON_RE.search(slug or "")
    return m.group(1) if m else MARKET_HORIZON_UNKNOWN


def _horizon_label(horizon_s: float) -> str:
    return f"{horizon_s:g}s"


@dataclass(slots=True)
class _Record:
    """Один филл, ожидающий замеров."""

    fill: Fill
    outcome: Outcome
    complement_token: str
    paired: Decimal = ZERO     # сколько shares этого филла вошло в пары
    unpaired: Decimal = ZERO   # остаток в FIFO-очереди для будущего матчинга
    market_horizon: str = MARKET_HORIZON_UNKNOWN   # "5m" / "15m" / "4h" / "?"


@dataclass(slots=True)
class _Agg:
    """Накопитель по одной ячейке (горизонт x разрез)."""

    markout_usdc: Decimal = ZERO   # суммарный mark-out, USDC
    size: Decimal = ZERO           # суммарный размер, shares
    n: int = 0                     # замеров с доступным mid
    n_miss: int = 0                # замеров без mid (рынок истёк/книги нет)

    def add(self, markout_usdc: Decimal, size: Decimal) -> None:
        self.markout_usdc += markout_usdc
        self.size += size
        self.n += 1

    def miss(self) -> None:
        self.n_miss += 1

    @property
    def per_share(self) -> Decimal | None:
        """Средний mark-out на одну share (взвешен размером)."""
        if self.size <= 0:
            return None
        return self.markout_usdc / self.size


class MarkoutTracker:
    """
    Планирует замеры mark-out по филлам и копит сводку за сессию.

    Живёт внутри event loop движка: record_fill() зовётся из колбэка филла,
    замер — отдельная asyncio-задача со сном до каждого горизонта.
    """

    def __init__(
        self,
        mid_source: MidSource,
        *,
        horizons_s: tuple[float, ...] = DEFAULT_HORIZONS_S,
        sink: Sink = log_event,
        max_pending: int = MAX_PENDING,
    ) -> None:
        if not horizons_s or any(h <= 0 for h in horizons_s):
            raise ValueError("horizons_s должны быть положительными")
        self.horizons_s = tuple(sorted(horizons_s))
        self._mid_source = mid_source
        self._sink = sink
        self._max_pending = max_pending

        # FIFO незапаренных остатков: condition_id -> outcome -> записи.
        self._queues: dict[str, dict[Outcome, deque[_Record]]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._dropped = 0

        # (label, разрез) -> накопитель. Разрез — либо корзина экономики
        # (paired/solo/unwind), либо сторона рынка (YES/NO).
        self._by_bucket: dict[tuple[str, str], _Agg] = {}
        self._by_outcome: dict[tuple[str, str], _Agg] = {}
        # Разрез по горизонту рынка: ключ "<горизонт>/<корзина>", например
        # "5m/solo" — та же экономика корзин, но отдельно по длительности окна.
        self._by_market_horizon: dict[tuple[str, str], _Agg] = {}

    # ------------------------------------------------------------- приём

    def record_fill(
        self,
        fill: Fill,
        outcome: Outcome,
        complement_token: str,
        market_horizon: str = MARKET_HORIZON_UNKNOWN,
    ) -> None:
        """
        Принять филл: сматчить в FIFO пар и запланировать замеры.
        market_horizon — длительность окна рынка ("5m"/"15m"/...), см.
        horizon_label_from_slug; движок берёт её из слага рынка.
        """
        if fill.size <= 0:
            return

        rec = _Record(
            fill=fill, outcome=outcome, complement_token=complement_token,
            market_horizon=market_horizon or MARKET_HORIZON_UNKNOWN,
        )
        self._prune_old(fill.condition_id)
        if fill.side == "BUY":
            self._match_buy(rec)
        else:
            self._consume_sell(rec)

        if len(self._tasks) >= self._max_pending:
            # Терять точку метрики можно, терять память — нет.
            self._dropped += 1
            if self._dropped in (1, 100, 1000):
                log.warning(
                    "Очередь замеров mark-out переполнена (%d) — точки теряются "
                    "(потеряно %d)", self._max_pending, self._dropped,
                )
            return

        task = asyncio.create_task(self._measure(rec), name=f"markout-{fill.trade_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _queue(self, condition_id: str, outcome: Outcome) -> deque[_Record]:
        per_market = self._queues.setdefault(condition_id, {"YES": deque(), "NO": deque()})
        return per_market[outcome]

    def _match_buy(self, rec: _Record) -> None:
        """FIFO-матчинг покупки против незапаренных покупок другой стороны."""
        other: Outcome = "NO" if rec.outcome == "YES" else "YES"
        opposite = self._queue(rec.fill.condition_id, other)
        remaining = rec.fill.size
        while remaining > 0 and opposite:
            head = opposite[0]
            take = min(remaining, head.unpaired)
            head.unpaired -= take
            head.paired += take
            rec.paired += take
            remaining -= take
            if head.unpaired <= 0:
                opposite.popleft()
        if remaining > 0:
            rec.unpaired = remaining
            self._queue(rec.fill.condition_id, rec.outcome).append(rec)

    def _consume_sell(self, rec: _Record) -> None:
        """
        Продажа съедает незапаренный инвентарь своей стороны: эти shares
        уже не станут парой, и будущим покупкам другой стороны матчиться
        с ними нельзя. Сама продажа в пары не входит никогда.
        """
        same = self._queue(rec.fill.condition_id, rec.outcome)
        remaining = rec.fill.size
        while remaining > 0 and same:
            head = same[0]
            take = min(remaining, head.unpaired)
            head.unpaired -= take
            remaining -= take
            if head.unpaired <= 0:
                same.popleft()
        # Остаток продажи сверх известных односторонних филлов — инвентарь,
        # восстановленный при старте: в очередях его нет, трогать нечего.

    def _prune_old(self, condition_id: str) -> None:
        per_market = self._queues.get(condition_id)
        if not per_market:
            return
        deadline = time.time() - PRUNE_AGE_S
        for queue in per_market.values():
            while queue and queue[0].fill.ts < deadline:
                queue.popleft()

    # ------------------------------------------------------------- замер

    async def _measure(self, rec: _Record) -> None:
        """Дождаться каждого горизонта и снять mid. Дедлайны абсолютные."""
        samples: list[tuple[float, Decimal | None]] = []
        for horizon in self.horizons_s:
            delay = rec.fill.ts + horizon - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                mid = self._mid_source(rec.fill.token_id, rec.complement_token)
            except Exception as exc:  # noqa: BLE001 - метрика не роняет движок
                log.error("mid_source упал: %s", exc)
                mid = None
            samples.append((horizon, mid))
        self._finalize(rec, samples)

    def _finalize(self, rec: _Record, samples: list[tuple[float, Decimal | None]]) -> None:
        fill = rec.fill
        sign = 1 if fill.side == "BUY" else -1

        if fill.side == "SELL":
            portions = [(BUCKET_UNWIND, fill.size)]
        else:
            paired = min(rec.paired, fill.size)
            solo = fill.size - paired
            portions = [(BUCKET_PAIRED, paired), (BUCKET_SOLO, solo)]

        event: dict[str, Any] = {
            "trade_id": fill.trade_id,
            "condition_id": fill.condition_id,
            "token": fill.token_id,
            "outcome": rec.outcome,
            "side": fill.side,
            "sign": sign,
            "price": fill.price,
            "size": fill.size,
            "paired_size": min(rec.paired, fill.size) if fill.side == "BUY" else ZERO,
            "solo_size": (
                fill.size - min(rec.paired, fill.size) if fill.side == "BUY" else ZERO
            ),
            "bucket": BUCKET_UNWIND if fill.side == "SELL" else None,
            "market_horizon": rec.market_horizon,
        }

        for horizon, mid in samples:
            label = _horizon_label(horizon)
            if mid is None:
                event[f"mid_{label}"] = None
                event[f"markout_{label}"] = None
                for bucket, size in portions:
                    if size > 0:
                        self._agg(self._by_bucket, label, bucket).miss()
                        self._agg(
                            self._by_market_horizon, label,
                            f"{rec.market_horizon}/{bucket}",
                        ).miss()
                self._agg(self._by_outcome, label, rec.outcome).miss()
                continue

            per_share = sign * (mid - fill.price)
            markout_usdc = per_share * fill.size
            event[f"mid_{label}"] = mid
            event[f"markout_{label}"] = markout_usdc
            for bucket, size in portions:
                if size > 0:
                    self._agg(self._by_bucket, label, bucket).add(per_share * size, size)
                    self._agg(
                        self._by_market_horizon, label, f"{rec.market_horizon}/{bucket}",
                    ).add(per_share * size, size)
            self._agg(self._by_outcome, label, rec.outcome).add(markout_usdc, fill.size)

        try:
            self._sink("markout", **event)
        except Exception as exc:  # noqa: BLE001
            log.error("Не удалось записать событие markout: %s", exc)

    @staticmethod
    def _agg(store: dict[tuple[str, str], _Agg], label: str, key: str) -> _Agg:
        agg = store.get((label, key))
        if agg is None:
            agg = store[(label, key)] = _Agg()
        return agg

    # ------------------------------------------------------------- сводка

    def summary(self) -> dict[str, dict[str, dict[str, float | int | None]]]:
        """Машиночитаемая сводка за сессию: по корзинам и по сторонам."""

        def dump(store: dict[tuple[str, str], _Agg]) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for (label, key), agg in sorted(store.items()):
                per_share = agg.per_share
                out.setdefault(label, {})[key] = {
                    "per_share": float(per_share) if per_share is not None else None,
                    "usdc": float(agg.markout_usdc),
                    "size": float(agg.size),
                    "n": agg.n,
                    "n_miss": agg.n_miss,
                }
            return out

        return {
            "bucket": dump(self._by_bucket),
            "outcome": dump(self._by_outcome),
            "market_horizon": dump(self._by_market_horizon),
        }

    def summary_lines(self) -> list[str]:
        """Строки для status_loop: средний mark-out на share по горизонтам."""

        def cell(store: dict[tuple[str, str], _Agg], label: str, key: str) -> str:
            agg = store.get((label, key))
            if agg is None or (agg.n == 0 and agg.n_miss == 0):
                return "-"
            per_share = agg.per_share
            if per_share is None:
                return f"miss={agg.n_miss}"
            text = f"{per_share:+.4f} (n={agg.n}"
            if agg.n_miss:
                text += f" miss={agg.n_miss}"
            return text + ")"

        if not self._by_bucket and not self._by_outcome:
            return []

        # Ключи разреза по горизонту рынка — те, что реально встретились
        # (5m/solo, 15m/paired, ...), в устойчивом порядке.
        horizon_keys = tuple(sorted({key for _, key in self._by_market_horizon}))

        lines = []
        for store, keys, title in (
            (self._by_bucket, (BUCKET_PAIRED, BUCKET_SOLO, BUCKET_UNWIND), "MARKOUT"),
            (self._by_outcome, ("YES", "NO"), "MARKOUT по стороне"),
            (self._by_market_horizon, horizon_keys, "MARKOUT по горизонту рынка"),
        ):
            if not keys:
                continue
            parts = []
            for horizon in self.horizons_s:
                label = _horizon_label(horizon)
                inner = " ".join(f"{k}={cell(store, label, k)}" for k in keys)
                parts.append(f"{label}: {inner}")
            lines.append(f"{title} | " + " | ".join(parts))
        return lines

    # ------------------------------------------------------------- жизнь

    @property
    def pending(self) -> int:
        return len(self._tasks)

    def forget_market(self, condition_id: str) -> None:
        """Рынок ушёл из окна торговли — его FIFO больше не понадобится."""
        self._queues.pop(condition_id, None)

    async def aclose(self) -> None:
        """Отменить незавершённые замеры (их события уже не запишутся)."""
        tasks = list(self._tasks)
        if not tasks:
            return
        log.info("Отменяю %d незавершённых замеров mark-out", len(tasks))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
