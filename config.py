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
    # Слаги серий Polymarket для краткосрочных Up/Down рынков.
    # Проверь актуальные на сайте: обычно это часовые/15-минутные серии.
    series_slugs: list[str] = Field(
        default_factory=lambda: ["bitcoin-up-or-down", "ethereum-up-or-down"]
    )
    # Резервный поиск по заголовку, если серия не найдена.
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

    # --- Цикл ---------------------------------------------------------------
    # Период пересчёта котировок (сек). Ниже 0.15 упрёшься в rate limit.
    quote_interval_s: float = 0.35
    # Перевыставляем ордер, только если цена сдвинулась больше чем на N тиков.
    requote_threshold_ticks: int = 1
    # TTL для GTD-ордеров (сек). 0 = использовать GTC.
    order_ttl_s: int = 60


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


def load_settings() -> Settings:
    """Единая точка загрузки конфигурации."""
    return Settings()
