"""Логирование: консоль + ротируемый файл + отдельный JSONL-аудит сделок."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from decimal import Decimal
from typing import Any

_TRADE_LOGGER = "polybot.trades"


class _SecretFilter(logging.Filter):
    """Страховка: вырезаем всё, что похоже на приватный ключ."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.msg)
        if "0x" in msg:
            import re

            record.msg = re.sub(r"0x[a-fA-F0-9]{64}", "0x<REDACTED>", msg)
        return True


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger("polybot")
    root.setLevel(getattr(logging, level))
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(_SecretFilter())
    root.addHandler(console)

    fileh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "bot.log"), maxBytes=20_000_000, backupCount=5
    )
    fileh.setFormatter(fmt)
    fileh.addFilter(_SecretFilter())
    root.addHandler(fileh)

    # Отдельный машиночитаемый лог сделок для последующего анализа PnL.
    trades = logging.getLogger(_TRADE_LOGGER)
    trades.setLevel(logging.INFO)
    trades.handlers.clear()
    trades.propagate = False
    th = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "trades.jsonl"), maxBytes=50_000_000, backupCount=10
    )
    th.setFormatter(logging.Formatter("%(message)s"))
    trades.addHandler(th)

    # SDK шумит на DEBUG — приглушаем.
    for noisy in ("httpx", "httpcore", "websockets", "polymarket"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def _default(o: Any) -> Any:
    if isinstance(o, Decimal):
        return float(o)
    return str(o)


def log_event(event: str, **fields: Any) -> None:
    """Записать структурированное событие в trades.jsonl."""
    payload = {"ts": time.time(), "event": event, **fields}
    logging.getLogger(_TRADE_LOGGER).info(json.dumps(payload, default=_default))
