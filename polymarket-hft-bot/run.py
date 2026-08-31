#!/usr/bin/env python3
"""
=============================================================================
                          ПРЕДУПРЕЖДЕНИЕ О РИСКАХ
=============================================================================
 ЭТО ВЫСОКОРИСКОВАННАЯ АЛГОРИТМИЧЕСКАЯ ТОРГОВАЯ СТРАТЕГИЯ.

 * НЕТ НИКАКИХ ГАРАНТИЙ ПРИБЫЛИ.
 * ВОЗМОЖНА ПОЛНАЯ И БЫСТРАЯ ПОТЕРЯ ВСЕХ СРЕДСТВ.
 * Маркет-мейкинг на 5-15-минутных крипторынках — среда с сильнейшим
   adverse selection. Против вас торгуют боты с колокацией и фидами,
   которые быстрее вашего на порядки.
 * Программные ошибки, разрывы сети, изменения API и проскальзывание
   могут привести к убыткам, превышающим любые расчёты.
 * Вы обязаны соблюдать geo-restrictions Polymarket, Terms of Service
   и законодательство своей юрисдикции. Проверьте это ДО запуска.
 * Это не финансовая рекомендация. Автор кода не несёт ответственности
   за ваши убытки.

 ЗАПУСКАЙТЕ С DRY_RUN=true. ПОТОМ С СУММОЙ, КОТОРУЮ НЕ ЖАЛКО ПОТЕРЯТЬ.
=============================================================================
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from src.config import load_settings
from src.engine import TradingEngine, install_signal_handlers
from src.logging_setup import setup_logging

BANNER = __doc__


async def main() -> int:
    load_dotenv()

    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}", file=sys.stderr)
        print("Скопируй .env.example в .env и заполни значения.", file=sys.stderr)
        return 2

    log = setup_logging(settings.runtime.log_dir, settings.runtime.log_level)
    print(BANNER)

    if not settings.runtime.dry_run:
        log.critical("DRY_RUN=false — БУДУТ ОТПРАВЛЕНЫ РЕАЛЬНЫЕ ОРДЕРА С РЕАЛЬНЫМИ ДЕНЬГАМИ")
        try:
            answer = input("Введите 'Я ПОНИМАЮ РИСКИ' для продолжения: ").strip()
        except EOFError:
            answer = ""
        if answer != "Я ПОНИМАЮ РИСКИ":
            log.info("Отменено пользователем.")
            return 0

    engine = TradingEngine(settings)
    install_signal_handlers(engine)

    try:
        await engine.run()
    except KeyboardInterrupt:
        engine.request_shutdown()
        await engine.shutdown()
    except Exception as exc:  # noqa: BLE001
        log.critical("ФАТАЛЬНАЯ ОШИБКА: %s", exc, exc_info=True)
        await engine.shutdown()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
