# CLAUDE.md

Контекст проекта для Claude Code. Читается автоматически при старте сессии.

## Что это

Маркет-мейкинг бот для краткосрочных (5–15 мин) рынков BTC/ETH Up-or-Down на
Polymarket CLOB. Python 3.11+, asyncio. Полное описание стратегии — в
`README.md`, прочитай его перед любыми изменениями логики.

## Критичный контекст — легко нарушить по незнанию

**SDK.** Пакет `polymarket-client`, импорт `polymarket`, основной класс
`AsyncSecureClient`. НЕ используй `py-clob-client` — он заархивирован
Polymarket в мае 2026, репозиторий read-only. Если видишь в интернете гайды с
`from py_clob_client.client import ClobClient` — это устаревший API, не
переписывай код под него.

**YES и NO зеркальны.** Матчинг-движок Polymarket трактует их как
дополняющие: покупка NO по `p` тождественна продаже YES по `1 − p`. Отсюда
`ask_NO ≡ 1 − bid_YES`. Тейкерского арбитража «yes_ask + no_ask < 1» внутри
одного рынка НЕ существует. Стратегия мейкерская: два бида с суммой < 1,
ждём пассивного исполнения обеих ног.

**Главный инвариант.** Сумма двух бидов ПЛЮС комиссия обеих ног ВСЕГДА
строго меньше `max_pair_cost` — после округления к тикам и после подгонки
под книгу. Если не получается — не котируем вовсе. Тесты
`test_pair_cost_invariant` и `test_pair_cost_invariant_with_fees` не должны
падать ни при каких изменениях.

**Комиссия считается по формуле из SDK**, а не по догадке:
`rate * (p * (1 - p)) ** exponent` на одну share — см.
`adjust_buy_amount_for_fees` в `polymarket/_internal/actions/orders/market.py`.
Ставка и экспонента приходят из `trading.fee_schedule` рынка. Если
расписание помечено `taker_only`, мы платим ноль: все наши ордера post_only,
то есть мы всегда мейкер.

**`merge_positions(amount=...)` принимает БАЗОВЫЕ ЕДИНИЦЫ, а не shares.**
CTF-позиции — ERC-1155 с 6 знаками, и SDK сверяет `amount` с балансом
токенов on-chain. Передать туда 25 вместо 25_000_000 значит смержить
0.000025 пары, заплатив полный газ, и разойтись с реальностью в учёте.
Перевод — `shares_to_base_units()` в `models.py`.

**Все ордера `post_only=True`.** Мы мейкер. Ордер, пересекающий спред, должен
быть отклонён биржей, а не исполнен как тейкер.

**Событие trade user-канала описывает ТЕЙКЕРА.** Верхнеуровневые
side/price/size — его; наша мейкерская нога лежит в `maker_orders` со
своими side/price/matched_amount. Читать верхнеуровневые поля как свой
филл значит перевернуть знак позиции. Разбор — только через
`_extract_own_fills` в `execution.py`.

**`merge_positions()` — не оптимизация, а суть стратегии.** Он превращает
пару YES+NO обратно в $1 USDC не дожидаясь резолюции. Без него капитал
заморожен до конца окна и бот делает 1–2 круга вместо десятков.

**Нет тестнета.** У Polymarket нет sandbox и paper-trading. Проверка логики —
только через `simulate.py` и `DRY_RUN=true`.

## Единицы на границе с SDK

Баг «не те единицы» не падает — он молча делает не то. Каждый вызов
`client.*` с числами закреплён канарейкой в `tests/test_units.py`; если
после обновления SDK канарейка упала (включая ImportError на приватном
пути) — перепроверь единицы руками, а не чини тест до зелёного.
Новый вызов `client.*` с числом = строка здесь + канарейка там. Форма
вызова — тоже контракт: корутину без await не поймает ни один тест на
заглушке, чья форма отличается от SDK (см. `tests/test_streams.py`).

| Вызов | Единицы | Канарейка |
|---|---|---|
| `create_limit_order(price, size)` | человеческие: USDC-цена 0..1, shares | `test_limit_order_price_and_size_are_human_units` |
| `create_limit_order(expiration)` | unix-секунды, не ближе now+180 (иначе UserInputError) | `test_limit_order_expiration_is_unix_seconds_with_exchange_minimum` |
| `merge_positions(amount)` | БАЗОВЫЕ единицы, 1e6 = 1 share | `test_merge_amount_is_base_units_and_scale_matches_sdk` |
| `list_positions()` → size, avg_price | человеческие | `test_position_model_returns_human_units` |
| `list_open_orders()` → price, sizes | человеческие | `test_open_order_model_returns_human_units` |
| `get_balance_allowance()` → balance | БАЗОВЫЕ единицы (int) | `test_balance_allowance_is_base_units_not_usdc` |
| user-stream trade → price/size | человеческие; верхний уровень — тейкер, наша нога в maker_orders | `test_user_trade_payload_units_and_maker_shape` |
| `subscribe(spec)` | КОРУТИНА: `async with await client.subscribe(...)`; без await все стримы мертвы | `test_subscribe_is_a_coroutine_returning_handle` |
| любой `list_*()` (пагинатор) | `async for` отдаёт ОБЪЕКТЫ Page; элементы — только через `.iter_items()`. Забытый iter_items не падает: getattr по Page молча даёт пустоту — «рынков/позиций/ордеров нет» | `test_paginator_iterates_pages_not_items` |

## Карта модулей

| Файл | Ответственность |
|---|---|
| `src/quoting.py` | ЯДРО. Генерация котировок, inventory skew, размеры |
| `src/fair_value.py` | GBM-модель, EWMA волатильность и momentum |
| `src/risk.py` | Лимиты, kill switch, dead-man switch |
| `src/execution.py` | Ордера: cancel/replace, атрибуция ног трейда, дедуп |
| `src/markout.py` | Mark-out: adverse selection по филлам (paired/solo) |
| `src/regime.py` | Детектор режима CALM/TRENDING/VOLATILE, гистерезис |
| `src/engine.py` | Оркестрация 10 asyncio-задач |
| `src/discovery.py` | Поиск рынков + калибровка страйка (3 стратегии) |
| `src/orderbook.py` | Локальное зеркало стаканов через WS |
| `src/price_feed.py` | Спот BTC/ETH: RTDS Polymarket или Binance |
| `src/config.py` | Pydantic-конфиг, кросс-валидация лимитов |
| `src/models.py` | Дата-классы, комиссии рынка, учёт позиций и пар |

## Проверка после изменений

```bash
pytest tests/ -v
python simulate.py --runs 150 --toxicity 0.5
```

Тесты разложены по файлам: `tests/test_strategy.py` — чистая логика
(модель, котирование, риск, учёт); `tests/test_markout.py`,
`tests/test_regime.py` и `tests/test_simulate.py` — тоже без SDK. `tests/test_engine.py` (движок на
заглушке клиента), `tests/test_execution.py` (user-stream),
`tests/test_discovery.py` (поиск рынков и diag.py на настоящем пагинаторе и
gamma-моделях) и `tests/test_units.py` (канарейки единиц) требуют
установленного пакета `polymarket`; без него они пропускаются.

Диагностика «рынков=0» в проде: `python diag.py` — читает публичный Gamma
API без ключей и ордеров, печатает актуальные слаги серий, найденные рынки
с token_id обеих ног и воронку фильтров бота.

Если добавляешь логику стратегии — добавь тест на её инвариант. Если меняешь
риск-лимиты — проверь, что кросс-валидация в `Settings._validate_cross()`
всё ещё осмысленна.

## Стиль

- Type hints обязательны.
- `Decimal` для всех цен и размеров, никогда `float`. Исключение —
  внутренняя математика в `fair_value.py`, где нужен `math`.
- Комментарии по-русски, идентификаторы по-английски.
- Любая проверка, которая может провалиться, должна проваливаться в сторону
  ОСТАНОВКИ торговли, а не продолжения.

## Незакрытые задачи

1. Цена газа merge — константа `STRAT_MERGE_GAS_COST` из конфига, а не факт
   из чека транзакции. При скачке цены POL реальные издержки разойдутся с
   учтёнными. Точное значение можно взять, дождавшись `handle.wait()`.
2. `merge_loop` не ждёт подтверждения транзакции: локальный учёт считает
   пары смерженными сразу после вызова. Если merge не прошёл on-chain,
   учёт разойдётся с реальностью до ближайшей сверки.
3. Позиции по рынкам вне окна торговли бот видит при старте, но закрыть не
   может — котирует он только своё окно. Redeem резолвленных позиций
   не автоматизирован.
4. `_place_bid` содержит мёртвую ветку: `min(price, max(price, improved))`
   тождественно равно `price`. Поведение задаётся следующей строкой.
5. Сверка позиций пропускает рынки с активностью моложе 90 с (data-api
   отстаёт от CLOB). На непрерывно торгуемом рынке она срабатывает только
   в паузах; дрейф в часы плотного потока ловится позже, а не сразу.
6. Реакция на TRENDING по умолчанию выключена: вердикт «эффект
   статистически неразличим от нуля» перемерян после моделирования
   разгрузки и только укрепился (см. README). Детектор по собственным
   филлам структурно опаздывает, а разгрузка вдобавок съедает его сигнал:
   продажи балансируют поток в окне, и в TRENDING он проводит 4-5 секунд
   из 600 трендовых. Быстрее его сделал бы только рыночный поток целиком
   (публичные трейды) — его в боте нет.
7. RISK_MAX_NET_EXPOSURE выше порога разгрузки
   min(STRAT_DIRECTIONAL_MAX_NET, лимит) ни на что не влияет: разгрузка
   не даёт |net| дойти до жёсткого лимита (свип в README). Реальный
   рычаг защиты от тренда — STRAT_DIRECTIONAL_MAX_NET, и его ужесточение
   оплачено оборотом спокойных окон почти один в один.
8. Дефолтные слаги серий (`bitcoin-up-or-down`, `ethereum-up-or-down`) не
   сверены с живым API: из среды разработки сеть к Polymarket закрыта, а на
   сайте фигурируют «Up/Down TWAP» рынки. Прогнать `python diag.py` с
   машины с доступом и, если скан покажет другие слаги, обновить
   STRAT_SERIES_SLUGS/STRAT_TITLE_KEYWORDS (раздел 3-4 вывода diag).
