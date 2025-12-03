import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET = os.getenv('BINANCE_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
USE_TESTNET = os.getenv('USE_TESTNET', 'false').lower() == 'true'

# Trading Configuration
TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
QUICK_MIN_SCORE = 5
MID_MIN_SCORE = 5
TREND_MIN_SCORE = 5

# TP/SL Configuration (percentage)
TP_SL_CONFIG = {
    'QUICK': {
        'tp': 0.4,  # 0.4%
        'sl': 0.25  # 0.25%
    },
    'MID': {
        'tp': 2.5,  # 2.5%
        'sl': 0.8   # 0.8%
    },
    'TREND': {
        'tp': 6.0,  # 6%
        'sl': 1.5   # 1.5%
    }
}

# Leverage Configuration
SUGGESTED_LEVERAGE = {
    'QUICK': 50,
    'MID': 30,
    'TREND': 20
}

# Safety Configuration
MIN_SAFE_LIQ_DISTANCE_PCT = 5.0  # Minimum 5% distance from liquidation
MAX_SPREAD_PCT = 0.05  # Max 0.05% spread
MIN_ORDERBOOK_DEPTH = 10000  # Min orderbook depth in USD

# Cooldown Configuration (seconds)
COOLDOWN_SECONDS = {
    'QUICK': 1800,   # 30 minutes
    'MID': 3600,     # 60 minutes
    'TREND': 7200    # 120 minutes
}

# BTC Volatility Check
BTC_VOLATILITY_THRESHOLDS = {
    '1m': 0.5,  # Max 0.5% move in 1 minute
    '5m': 1.5   # Max 1.5% move in 5 minutes
}

# Logic Weights (for weighted scoring)
LOGIC_WEIGHTS = {
    # QUICK
    'RSI_oversold_breakout': 1.2,
    'MACD_bullish_cross': 1.5,
    'Volume_spike_support': 1.0,
    'VWAP_reclaim_long': 1.0,
    'EMA_9_21_bull_cross': 1.3,
    'Orderblock_retest_long': 1.1,
    'Liquidity_sweep_long': 1.2,
    'Bollinger_band_squeeze_break': 1.4,
    'Spread_tight_low_latency': 0.8,
    'Market_structure_HH_HL': 1.0,
    
    # MID
    'RSI_overbought_reversal': 1.2,
    'MACD_hidden_bullish': 1.3,
    'MACD_divergence_support': 1.4,
    'ADX_trend_strength_up': 1.5,
    'Volume_delta_buy_pressure': 1.0,
    'EMA_50_bounce': 1.2,
    'EMA_200_bounce': 1.3,
    'FVG_immediate_fill': 1.1,
    'Keltner_breakout_up': 1.2,
    'Trendline_break_retest': 1.0,
    
    # TREND
    'Breaker_block_retest': 1.3,
    'Chop_zone_exit_long': 1.1,
    'Bollinger_midband_reject_flip': 1.0,
    'Supertrend_flip_bull': 1.5,
    'ATR_volatility_drop_entry': 1.2,
    'Pullback_0_382_fib_entry': 1.3,
    'Pullback_0_5_fib_entry': 1.4,
    'Pullback_0_618_fib_entry': 1.5,
    'Support_demand_zone_reaction': 1.2,
    'Imbalance_fill_continuation': 1.0
}

# Retry Configuration
FETCH_RETRY_COUNT = 3
FETCH_RETRY_BACKOFF = 2  # exponential backoff multiplier