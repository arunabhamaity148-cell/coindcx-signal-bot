"""
Configuration file for Crypto Trading Bot (FINAL - quick fix)
- Multi-symbol quick mode (15 symbols)
- TA-only fallback enabled via ml_required flag
- Check interval set to 5 minutes (300 seconds)
"""
import os
from dotenv import load_dotenv

# load API keys from config/api_keys.env
load_dotenv('config/api_keys.env')

# ==================== EXCHANGE CONFIG ====================
EXCHANGES = {
    'bybit': {
        'api_key': os.getenv('BYBIT_API_KEY'),
        'secret': os.getenv('BYBIT_SECRET'),
        'testnet': False,
        'capital_allocation': 0.70  # 70%
    },
    'okx': {
        'api_key': os.getenv('OKX_API_KEY'),
        'secret': os.getenv('OKX_SECRET'),
        'password': os.getenv('OKX_PASSWORD'),
        'testnet': False,
        'capital_allocation': 0.20  # 20%
    },
    'binance': {  # Data only
        'api_key': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET'),
        'testnet': False,
        'use_for_trading': False  # Only for data
    }
}

# ==================== TRADING CONFIG ====================
TRADING_CONFIG = {
    # Multi-symbol Quick mode (15 symbols)
    'symbols': [
        'BTC/USDT:USDT',
        'ETH/USDT:USDT',
        'SOL/USDT:USDT',
        'BNB/USDT:USDT',
        'XRP/USDT:USDT',
        'DOGE/USDT:USDT',
        'ADA/USDT:USDT',
        'AVAX/USDT:USDT',
        'MATIC/USDT:USDT',
        'DOT/USDT:USDT',
        'LINK/USDT:USDT',
        'UNI/USDT:USDT',
        'ATOM/USDT:USDT',
        'LTC/USDT:USDT',
        'BCH/USDT:USDT'
    ],
    'primary_symbol': 'BTC/USDT:USDT',
    # interval for checks (in seconds) - 300s = 5 minutes
    'check_interval': 300,
    'timeframe': '15m',
    'initial_capital': 10000,  # INR
    'max_leverage': 7,
    'min_leverage': 3,
    'position_size_percent': 0.15,  # 15% per trade
    'max_positions': 5,              # increase for multi-symbol scanning
    'max_daily_trades': 25,          # allow more trades per day in quick mode
    'max_daily_loss_percent': 0.20,  # 20%
    'max_trade_duration_hours': 4,

    # Quick fix ML toggle: if False -> TA-only (no ML required)
    'ml_required': False,
    # minimum ML confidence required (ignored if ml_required=False)
    'confidence_threshold': 0.0,
}

# ==================== RISK MANAGEMENT ====================
RISK_CONFIG = {
    'stop_loss_percent': 0.02,  # 2%
    'take_profit_percent': 0.03,  # 3%
    'trailing_stop_percent': 0.015,  # 1.5%
    'liquidation_buffer_percent': 0.05,  # 5%
    'max_drawdown_percent': 0.25,  # 25%
    'risk_reward_ratio': 1.5,
    'consecutive_loss_limit': 3,
    'consecutive_loss_position_reduce': 0.50,  # 50%
}

# ==================== ML CONFIG ====================
ML_CONFIG = {
    'lstm_weight': 0.50,
    'xgboost_weight': 0.30,
    'rf_weight': 0.20,
    # Note: confidence thresholds here are defaults; TRADING_CONFIG['ml_required'] controls usage
    'confidence_threshold': 0.70,
    'high_confidence_threshold': 0.75,
    'lookback_candles': 60,
    'features_count': 50,
    'retrain_frequency_days': 30,
    'model_path': 'models/',
}

# ==================== DATA CONFIG ====================
DATA_CONFIG = {
    'historical_years': 5,
    # data update interval in seconds when polling (keep small if using websockets)
    'data_update_interval': 15,
    'use_websocket': True,
    'database_type': 'postgresql',
    'db_host': os.getenv('DB_HOST', 'localhost'),
    'db_port': os.getenv('DB_PORT', '5432'),
    'db_name': os.getenv('DB_NAME', 'trading_bot'),
    'db_user': os.getenv('DB_USER', 'postgres'),
    'db_password': os.getenv('DB_PASSWORD'),
}

# ==================== UNIQUE LOGICS CONFIG ====================
UNIQUE_LOGICS = {
    # Market Health Filters
    'btc_calm_threshold': 0.015,  # 1.5% max volatility
    'funding_rate_extreme': 0.0015,  # 0.15% (decimal)
    'fear_greed_extreme_low': 20,
    'fear_greed_extreme_high': 80,
    'news_skip_minutes': 30,
    'spread_max_percent': 0.001,  # 0.1% expressed as decimal
    'low_liquidity_hours': [2, 3, 4, 5],  # UTC

    # Order Flow
    'orderbook_imbalance_threshold': 1.2,
    'vwap_deviation_percent': 0.005,  # 0.5%
    'large_order_threshold': 100000,  # $100k
    'spoofing_wall_threshold': 500000,  # $500k
    'cross_exchange_divergence': 0.003,  # 0.3%

    # Liquidation & Gamma
    'liquidation_proximity_percent': 0.02,  # 2%
    'gamma_exposure_threshold': 0.5,

    # Anti-trap
    'round_number_avoid_distance': 0.001,  # 0.1%
    'sl_hunting_zone_percent': 0.005,  # 0.5%
    'avoid_round_hours': [9, 10, 17],  # UTC
}

# ==================== TELEGRAM CONFIG ====================
TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'enable_notifications': True,
    'notify_on_trade': True,
    'notify_on_error': True,
    'daily_summary_time': '18:00',  # IST
}

# ==================== BACKTEST CONFIG ====================
BACKTEST_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'train_split': 0.70,
    'validation_split': 0.15,
    'test_split': 0.15,
    'include_fees': True,
    'maker_fee': 0.0002,  # 0.02%
    'taker_fee': 0.0005,  # 0.05%
    'slippage': 0.0005,  # 0.05%
}

# ==================== LOGGING CONFIG ====================
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'logs/trading_bot.log',
    'max_log_size_mb': 100,
    'backup_count': 5,
}