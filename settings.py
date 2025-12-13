"""
GLOBAL SETTINGS CONFIG
Institutional CoinDCX Trading Bot
- 45 Unique Logics
- No ML dependency (ML optional later)
- Target: 15–25 signals/day
"""

# ==================== GENERAL ====================
APP_NAME = "Arundhoni Institutional Bot"
ENV = "production"   # development | production
TIMEZONE = "Asia/Kolkata"

# ==================== SCANNER CONFIG ====================
SCANNER_CONFIG = {
    "symbols": [
        "B-BTC_USDT",
        "B-ETH_USDT",
        "B-SOL_USDT",
        "B-BNB_USDT",
        "B-XRP_USDT",
        "B-DOGE_USDT",
        "B-ADA_USDT",
        "B-AVAX_USDT",
        "B-MATIC_USDT",
        "B-DOT_USDT",
        "B-LINK_USDT",
        "B-UNI_USDT",
        "B-ATOM_USDT",
        "B-LTC_USDT",
        "B-TRX_USDT",
    ],

    # Multi-timeframe confirmation
    "timeframes": ["5m", "15m", "1h"],

    # Scan every 5 minutes
    "scan_interval_seconds": 300,

    # Signal filtering
    "min_signal_score": 65,
    "high_quality_score": 75,

    # Hard limits
    "max_signals_per_day": 25,
    "cooldown_minutes_per_symbol": 30,
}

# ==================== TRADING MODE ====================
TRADING_CONFIG = {
    "mode": "paper",          # paper | live
    "exchange": "coindcx",
    "allow_long": True,
    "allow_short": False,     # CoinDCX spot = no short
    "max_open_positions": 5,
    "max_trades_per_scan": 3,
}

# ==================== CAPITAL & POSITION SIZING ====================
CAPITAL_CONFIG = {
    "initial_capital": 10000,     # INR
    "risk_per_trade_percent": 1.5,  # % of capital
    "max_capital_per_trade": 0.20,  # 20%
}

# ==================== RISK MANAGEMENT ====================
RISK_CONFIG = {
    "min_risk_reward": 1.5,

    # Daily protection
    "max_daily_loss_percent": 4.0,
    "max_daily_trades": 10,

    # Volatility protection
    "atr_multiplier_sl": 1.5,
    "atr_multiplier_tp": 2.5,

    # Drawdown protection
    "max_drawdown_percent": 20,

    # Cooldown
    "consecutive_loss_limit": 2,
}

# ==================== MARKET HEALTH FILTERS ====================
MARKET_HEALTH = {
    "btc_calm_threshold": 0.015,        # 1.5%
    "funding_rate_extreme": 0.0015,
    "fear_greed_low": 20,
    "fear_greed_high": 80,
    "news_skip_minutes": 30,
    "spread_max_percent": 0.001,        # 0.1%
    "low_liquidity_hours_utc": [2, 3, 4, 5],
}

# ==================== ORDER FLOW ====================
ORDERFLOW_CONFIG = {
    "orderbook_imbalance_threshold": 1.2,
    "large_order_threshold": 100_000,
    "spoofing_wall_threshold": 500_000,
    "vwap_deviation_percent": 0.005,
}

# ==================== DERIVATIVES / FUTURES PLACEHOLDERS ====================
# (Spot mode e inactive, future-proof)
DERIVATIVES_CONFIG = {
    "oi_trend_lookback": 20,
    "liquidation_proximity_percent": 0.02,
    "gamma_exposure_threshold": 0.5,
}

# ==================== ANTI-TRAP LOGICS ====================
ANTI_TRAP_CONFIG = {
    "round_number_avoid_distance": 0.001,
    "sl_hunting_wick_ratio": 2.0,
    "avoid_round_hours_utc": [9, 10, 17],
}

# ==================== TELEGRAM ====================
TELEGRAM_CONFIG = {
    "enabled": True,
    "send_only_high_quality": True,
    "max_signals_per_message": 5,
}

# ==================== LOGGING ====================
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_dir": "logs",
    "log_file_prefix": "coindcx_bot",
}

# ==================== 45 UNIQUE LOGICS MASTER SWITCH ====================
UNIQUE_LOGICS = {
    # Market Health (8)
    "btc_calm": True,
    "market_regime": True,
    "funding_filter": True,
    "fear_greed": True,
    "fragile_market": True,
    "news_filter": True,
    "spread_filter": True,
    "liquidity_window": True,

    # Price Action (7)
    "breakout_confirmation": True,
    "market_structure": True,
    "orderblock_retest": True,
    "fair_value_gap": True,
    "ema_alignment": True,
    "atr_filter": True,
    "bollinger_squeeze": True,

    # Momentum (6)
    "rsi_logic": True,
    "macd_logic": True,
    "stochastic_logic": True,
    "obv_divergence": True,
    "mfi_logic": True,
    "roc_logic": True,

    # Order Flow (10)
    "orderbook_imbalance": True,
    "vwap_deviation": True,
    "vwap_structure": True,
    "cvd_logic": True,
    "large_orders": True,
    "spoofing_detection": True,
    "true_liquidity": True,
    "aggression_ratio": True,
    "spread_velocity": True,
    "volume_spike": True,

    # Derivatives (6)
    "oi_trend": False,
    "oi_divergence": False,
    "liquidation_clusters": False,
    "funding_arbitrage": False,
    "gamma_exposure": False,
    "gamma_sizing": False,

    # Anti-Trap (8)
    "avoid_round_numbers": True,
    "avoid_sr": True,
    "sl_hunting_zone": True,
    "odd_time_filter": True,
    "sudden_wick_filter": True,
    "bot_rush_filter": True,
    "manipulation_candle": True,
    "consecutive_loss_cooldown": True,
}
