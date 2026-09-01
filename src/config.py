"""
=============================================================================
 ПРЕДУПРЕЖДЕНИЕ О РИСКАХ
=============================================================================
 ЭТО ВЫСОКОРИСКОВАННАЯ АЛГОРИТМИЧЕСКАЯ ТОРГОВАЯ СТРАТЕГИЯ.
 НЕТ НИКАКИХ ГАРАНТИЙ ПРИБЫЛИ. ВОЗМОЖНА ПОЛНАЯ ПОТЕРЯ СРЕДСТВ.
 Вы обязаны соблюдать geo-restrictions Polymarket, Terms of Service
 и законодательство своей юрисдикции. Автор кода не несёт
 ответственности за ваши убытки. Это не финансовая рекомендация.
=============================================================================

Конфигурация бота. Всё читается из переменных окружения (.env).
Секреты НИКОГДА не хардкодятся и не логируются.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import MIN_GTD_TTL_S


class WalletSettings(BaseSettings):
    """Реквизиты кошелька и API-доступа Polymarket."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Приватный ключ EOA-кошелька (Polygon). ТОЛЬКО из окружения.
    private_key: SecretStr = Field(alias="POLYMARKET_PRIVATE_KEY")

    # Адрес proxy-кошелька Polymarket (если торгуешь через Magic/Safe-прокси).
    # Оставь пустым, если торгуешь напрямую с EOA.
    wallet_address: str | None = Field(default=None, alias="POLYMARKET_WALLET_ADDRESS")

    # L2 API-креды. Если не заданы — бот выведет их из приватного ключа
    # (create_or_derive) при старте.
    api_key: SecretStr | None = Field(default=None, alias="POLYMARKET_API_KEY")
    api_secret: SecretStr | None = Field(default=None, alias="POLYMARKET_API_SECRET")
    api_passphrase: SecretStr | None = Field(
        default=None, alias="POLYMARKET_API_PASSPHRASE"
    )

    @field_validator("private_key")
    @classmethod
    def _check_pk(cls, v: SecretStr) -> SecretStr:
        raw = v.get_secret_value().strip()
        if not raw.startswith("0x") or len(raw) != 66:
            raise ValueError("POLYMARKET_PRIVATE_KEY должен быть hex-строкой 0x + 64 символа")
        return SecretStr(raw)


class StrategySettings(BaseSettings):
    """Параметры торговой логики."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="STRAT_"
    )

    # --- Выбор рынков -------------------------------------------------------
    # ОСНОВНОЙ путь поиска — полнотекстовый title_search по ключевым словам:
    # на живом API он находит updown-рынки (btc-updown-5m-*, *-15m-*), а
    # запросы по слагам серий стабильно возвращают ноль. Слаги серий — только
    # РЕЗЕРВ на случай, если поиск по заголовку не дал ничего; пустой список
    # (дефолт) отключает резерв без предупреждений в логе.
    series_slugs: list[str] = Field(default_factory=list)
    title_keywords: list[str] = Field(
        default_factory=lambda: ["Bitcoin Up or Down", "Ethereum Up or Down"]
    )
    # Торгуем рынок, только если до экспирации от MIN до MAX секунд.
    min_seconds_to_expiry: int = 90
    max_seconds_to_expiry: int = 15 * 60
    # Максимум одновременно котируемых рынков.
    max_concurrent_markets: int = 4
    # Как часто пересканировать список активных рынков (сек).
    discovery_interval_s: float = 20.0

    # --- Котирование (two-sided MM / pair accumulation) ---------------------
    # Целевая сумма двух бидов (a + b). Всё, что < 1.0, — валовая маржа пары.
    # 0.985 => 1.5 цента на пару. Агрессивнее = больше филлов, меньше маржа.
    target_pair_cost: Decimal = Decimal("0.985")
    # Никогда не котируем так, чтобы a + b превысило это значение.
    max_pair_cost: Decimal = Decimal("0.995")
    # Минимальный полуспред от fair value, в тиках.
    min_half_spread_ticks: int = 1
    # Размер одного ордера (в shares; 1 share = $1 при выигрыше).
    order_size: Decimal = Decimal("20")
    # Минимальный размер ордера, принимаемый рынком (переопределится из API).
    fallback_min_order_size: Decimal = Decimal("5")
    # Не котируем, если рыночный спред шире (защита от неликвида).
    max_market_spread: Decimal = Decimal("0.06")
    # Не котируем, если суммарная глубина топ-3 уровней меньше (shares).
    min_book_depth: Decimal = Decimal("50")

    # --- Комиссии рынка -----------------------------------------------------
    # У части рынков fees_enabled=true. Ставку и экспоненту бот читает из
    # расписания комиссий рынка; комиссия вычитается из маржи пары, то есть
    # спред автоматически раздвигается. Эта ставка используется ТОЛЬКО как
    # запасная: рынок объявил комиссии, но расписание не отдал. Занижать её
    # опасно — котирование с отрицательной чистой маржой по логам не видно.
    # При 0.02 комиссия пары у 0.50 равна ~1 центу, то есть съедает почти всю
    # валовую маржу дефолтного target_pair_cost: такие рынки бот пропустит.
    fallback_fee_rate: Decimal = Decimal("0.02")

    # --- Fair value модель --------------------------------------------------
    # Вес модели против рыночного mid. 0 = чистый MM по рынку, 1 = чистая модель.
    model_weight: Decimal = Decimal("0.35")
    # Окно EWMA для реализованной волатильности (полупериод, сек).
    vol_halflife_s: float = 45.0
    # Окно EWMA для momentum-дрейфа (полупериод, сек).
    momentum_halflife_s: float = 8.0
    # Коэффициент, с которым momentum превращается в дрейф. 0 = без directional.
    momentum_drift_coef: Decimal = Decimal("0.30")
    # Максимальный сдвиг fair value от модели (защита от взрыва модели).
    max_model_deviation: Decimal = Decimal("0.15")
    # Минимальная стартовая волатильность, годовая (пока не накопили данных).
    vol_floor_annual: Decimal = Decimal("0.30")

    # --- Инвентарь / directional bias --------------------------------------
    # Целевой net inventory в shares (0 = дельта-нейтрально).
    target_net_inventory: Decimal = Decimal("0")
    # Коэффициент skew котировок от инвентаря (Avellaneda-Stoikov-lite).
    # Сдвиг резервной цены = -inventory_skew_coef * net / max_net.
    inventory_skew_coef: Decimal = Decimal("0.010")
    # Разрешить оставлять направленную позицию, если модель видит edge.
    allow_directional: bool = True
    # Минимальный edge (модель - рынок), чтобы вообще держать направление.
    directional_min_edge: Decimal = Decimal("0.025")
    # Максимальный net exposure ради directional (shares).
    directional_max_net: Decimal = Decimal("60")

    # --- Merge полных пар ---------------------------------------------------
    # Автоматически мержить полные пары обратно в USDC (освобождает капитал).
    auto_merge: bool = True
    # Минимальный размер пары для merge (газ/latency не окупаются на мелочи).
    min_merge_size: Decimal = Decimal("25")
    # Не чаще, чем раз в N секунд.
    merge_interval_s: float = 20.0
    # Стоимость одной транзакции merge, USDC. У proxy-кошелька Polymarket
    # merge идёт через relayer, газ платит он — тогда ставь 0. При торговле
    # с EOA транзакция своя, и газ реальный.
    merge_gas_cost: Decimal = Decimal("0.01")
    # Мержим, только если ожидаемая прибыль пачки больше газа во столько раз.
    # Merge ради прибыли, равной газу, — это просто сжигание газа.
    merge_min_profit_ratio: Decimal = Decimal("3")

    # --- Режим рынка ---------------------------------------------------------
    # Детектор (src/regime.py) работает всегда — это только метрики. Флаги
    # ниже включают РЕАКЦИЮ котирования на его состояние.
    # В TRENDING полуспред асимметричен: сторона, которую засыпает поток,
    # отодвигается или снимается, противоположная подтягивается к рынку —
    # она достраивает пары к уже накопленному инвентарю.
    # ПО УМОЛЧАНИЮ ВЫКЛЮЧЕНО: на 2000 окон с трендами (см. README) эффект
    # реакции статистически неразличим от нуля в обеих группах окон —
    # детектор по собственным филлам опаздывает, урон трендового окна
    # ограничивает риск-лимит net, а не котировки. Включай только после
    # проверки на своих данных (python simulate.py --regime-compare).
    regime_trending_response: bool = False
    # В VOLATILE не котировать вовсе: модель и котировки не успевают.
    regime_volatile_no_quote: bool = True
    # Насколько тиков отодвигается сторона, которую выносит поток...
    trending_crowded_extra_ticks: int = 3
    # ...или снять её полностью (строже, чем отодвигание).
    trending_remove_crowded: bool = False
    # Насколько тиков подтягивается к рынку противоположная сторона.
    # Не может превышать trending_crowded_extra_ticks: иначе асимметрия
    # УХУДШАЛА бы сумму пары, а не улучшала.
    trending_tighten_ticks: int = 1
    # Пороги детектора. Входные строже выходных — это гистерезис.
    regime_window_s: float = 120.0
    # Калибр — собственный темп филлов бота (~4-6 за 120 с): порог выше
    # делает детектор слепым, вход случается к середине тренда.
    regime_min_fills: int = 4
    regime_imbalance_enter: float = 0.70
    regime_imbalance_soft: float = 0.45
    regime_imbalance_exit: float = 0.40
    regime_autocorr_enter: float = 0.25
    regime_vol_ratio_enter: float = 1.8
    regime_vol_ratio_exit: float = 1.35
    # Минимальное удержание TRENDING: реакция сама душит поток филлов,
    # по которому тренд обнаружен, и без удержания детектор осциллирует.
    regime_min_hold_s: float = 45.0
    # Прогрев вол-сигнала ПО ВРЕМЕНИ (сек), в дополнение к минимуму сэмплов.
    # Живой фид даёт 5-20 тиков/с: счётчик сэмплов набирается за полминуты,
    # когда медленная EWMA ещё прибита к первым тикам, — отношение fast/slow
    # завышено и бот глохнет в ложном VOLATILE прямо на старте. До истечения
    # прогрева вол-сигнал НЕДОСТУПЕН (режим по нему не назначается вовсе).
    # Дефолт = полупериод медленной EWMA (300 с): остаточный вес первых
    # сэмплов <= 50%, завышение ограничено sqrt(2) < порога входа 1.8.
    regime_vol_min_elapsed_s: float = 300.0

    # --- Валидность страйка --------------------------------------------------
    # Сторож расхождения модели с рынком. Если |model - mid| держится выше
    # порога дольше окна, страйк признаётся невалидным: модель для рынка
    # отключается (чистый MM по рынку), а не клипается вокруг смещённого
    # центра — клип оставляет fair сдвинутым на max_model_deviation и
    # разъезжает котировки так, что обе ноги никогда не исполняются.
    strike_divergence_threshold: Decimal = Decimal("0.25")
    strike_divergence_hold_s: float = 8.0
    # Диагностика сломанного fair value: если сумма двух бидов ушла ниже
    # target_pair_cost больше, чем на этот зазор, — WARNING в лог.
    pair_sum_warn_gap: Decimal = Decimal("0.05")

    # --- Восстановление после рестарта --------------------------------------
    # Читать открытые позиции с биржи при старте. Выключать это значит
    # считать риск-лимиты от нуля, имея на кошельке позицию прошлой сессии.
    recover_positions: bool = True

    # --- Цикл ---------------------------------------------------------------
    # Период пересчёта котировок (сек). Ниже 0.15 упрёшься в rate limit.
    quote_interval_s: float = 0.35
    # Перевыставляем ордер, только если цена сдвинулась больше чем на N тиков.
    requote_threshold_ticks: int = 1
    # TTL для GTD-ордеров (сек). 0 = использовать GTC. Биржа не принимает
    # GTD с expiration ближе, чем now + 180 секунд, поэтому минимум — 210
    # (см. MIN_GTD_TTL_S в models.py). Значение ниже валидация отвергает.
    order_ttl_s: int = 240


class RiskSettings(BaseSettings):
    """Жёсткие лимиты риска. Нарушение => отмена всех ордеров."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="RISK_"
    )

    # Максимальная позиция по одной стороне одного рынка (shares).
    max_position_per_side: Decimal = Decimal("250")
    # Максимальный net directional exposure по всем рынкам (shares).
    max_net_exposure: Decimal = Decimal("120")
    # Максимальный нотионал в работе (USDC).
    max_notional: Decimal = Decimal("500")
    # Дневной лимит убытка (USDC, положительное число). Достигнут => стоп.
    daily_loss_limit: Decimal = Decimal("50")
    # Максимум открытых ордеров одновременно.
    max_open_orders: int = 16
    # Dead-man switch: если главный цикл не тикал N сек — отменить всё.
    heartbeat_timeout_s: float = 6.0
    # Если цена спота не обновлялась N сек — считаем фид мёртвым, не котируем.
    stale_price_timeout_s: float = 4.0
    # Если стакан не обновлялся N сек — рынок считается протухшим.
    stale_book_timeout_s: float = 15.0
    # Максимум отклонённых ордеров подряд до аварийной остановки.
    max_consecutive_rejects: int = 8


class RuntimeSettings(BaseSettings):
    """Режим работы и инфраструктура."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # dry_run=True — бот считает и логирует котировки, но НЕ отправляет ордера.
    # ВСЕГДА начинай с dry_run=True.
    dry_run: bool = Field(default=True, alias="DRY_RUN")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_dir: str = Field(default="./logs", alias="LOG_DIR")
    # Источник спот-цены: 'polymarket' (встроенный RTDS-стрим) или 'binance'.
    price_source: Literal["polymarket", "binance"] = Field(
        default="polymarket", alias="PRICE_SOURCE"
    )
    binance_ws_url: str = Field(
        default="wss://stream.binance.com:9443/stream", alias="BINANCE_WS_URL"
    )


class Settings:
    """Агрегатор всех конфигов."""

    def __init__(self) -> None:
        self.wallet = WalletSettings()  # type: ignore[call-arg]
        self.strategy = StrategySettings()
        self.risk = RiskSettings()
        self.runtime = RuntimeSettings()
        self._validate_cross()

    def _validate_cross(self) -> None:
        s, r = self.strategy, self.risk
        if s.target_pair_cost >= s.max_pair_cost:
            raise ValueError("STRAT_TARGET_PAIR_COST должен быть < STRAT_MAX_PAIR_COST")
        if s.max_pair_cost >= Decimal("1"):
            raise ValueError("STRAT_MAX_PAIR_COST >= 1.0 — стратегия убыточна by design")
        if s.order_size > r.max_position_per_side:
            raise ValueError("STRAT_ORDER_SIZE больше, чем RISK_MAX_POSITION_PER_SIDE")
        if s.directional_max_net > r.max_net_exposure:
            raise ValueError("STRAT_DIRECTIONAL_MAX_NET больше RISK_MAX_NET_EXPOSURE")
        if s.min_seconds_to_expiry >= s.max_seconds_to_expiry:
            raise ValueError("min_seconds_to_expiry должен быть < max_seconds_to_expiry")
        if not (Decimal("0") <= s.fallback_fee_rate <= Decimal("0.10")):
            raise ValueError("STRAT_FALLBACK_FEE_RATE вне разумного диапазона 0..0.10")
        if s.merge_gas_cost < 0:
            raise ValueError("STRAT_MERGE_GAS_COST не может быть отрицательным")
        if s.merge_min_profit_ratio < Decimal("1"):
            raise ValueError(
                "STRAT_MERGE_MIN_PROFIT_RATIO < 1 — merge будет стоить дороже прибыли"
            )
        if 0 < s.order_ttl_s < MIN_GTD_TTL_S:
            raise ValueError(
                f"STRAT_ORDER_TTL_S={s.order_ttl_s} ниже биржевого минимума GTD "
                f"({MIN_GTD_TTL_S} сек): ни один ордер не подпишется. "
                "Поставь >= минимума или 0 (GTC)."
            )
        if s.trending_tighten_ticks < 0 or s.trending_crowded_extra_ticks < 0:
            raise ValueError("Сдвиги котировок в TRENDING не могут быть отрицательными")
        if s.trending_tighten_ticks > s.trending_crowded_extra_ticks:
            raise ValueError(
                "STRAT_TRENDING_TIGHTEN_TICKS > STRAT_TRENDING_CROWDED_EXTRA_TICKS: "
                "асимметрия увеличивала бы сумму пары вместо того, чтобы уменьшать"
            )
        if not (s.regime_imbalance_exit <= s.regime_imbalance_soft <= s.regime_imbalance_enter):
            raise ValueError("Пороги односторонности: exit <= soft <= enter (гистерезис)")
        if s.regime_vol_ratio_exit >= s.regime_vol_ratio_enter:
            raise ValueError("REGIME_VOL_RATIO: порог выхода должен быть ниже порога входа")
        if s.regime_vol_min_elapsed_s < 0:
            raise ValueError("STRAT_REGIME_VOL_MIN_ELAPSED_S не может быть отрицательным")
        if not (Decimal("0") < s.strike_divergence_threshold <= Decimal("0.9")):
            raise ValueError("STRAT_STRIKE_DIVERGENCE_THRESHOLD вне диапазона (0, 0.9]")
        if s.strike_divergence_threshold <= s.max_model_deviation:
            raise ValueError(
                "STRAT_STRIKE_DIVERGENCE_THRESHOLD должен быть выше "
                "STRAT_MAX_MODEL_DEVIATION: иначе сторож срабатывает на "
                "расхождении, которое клип и так считает рабочим"
            )
        if s.strike_divergence_hold_s <= 0:
            raise ValueError("STRAT_STRIKE_DIVERGENCE_HOLD_S должен быть > 0")
        if not (Decimal("0") <= s.pair_sum_warn_gap < s.target_pair_cost):
            raise ValueError("STRAT_PAIR_SUM_WARN_GAP вне разумного диапазона")


def load_settings() -> Settings:
    """Единая точка загрузки конфигурации."""
    return Settings()
