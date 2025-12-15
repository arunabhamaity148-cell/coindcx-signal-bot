"""
🔐 CoinDCX Futures Trading Bot - Configuration
Bangla Version for Best Friend 🇧🇩
"""

import os
import logging
from datetime import time

# ========================
# 🔧 DEBUG MODE
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========================
# 🔑 API CREDENTIALS
# ========================
COINDCX_API_KEY = os.getenv("COINDCX_API_KEY", "your_api_key_here")
COINDCX_SECRET = os.getenv("COINDCX_SECRET", "your_secret_here")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_telegram_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your_chat_id")

# ========================
# 💰 TRADING PARAMETERS
# ========================
MARGIN_PER_TRADE = 3000  # ₹3000 per trade
LEVERAGE = 5  # 5x leverage
TARGET_DAILY_PROFIT = 2000  # ₹2000/day
MAX_DAILY_SIGNALS = 15
MIN_DAILY_SIGNALS = 10

# ========================
# 🎯 SIGNAL MODES
# ========================
SIGNAL_MODES = {
    "quick": {
        "timeframe": "5m",
        "hold_time": 15,  # minutes
        "tp1": 0.6,  # 0.6% TP1
        "tp2": 1.2,  # 1.2% TP2
        "sl": 0.4,   # 0.4% SL
    },
    "mid": {
        "timeframe": "15m",
        "hold_time": 60,
        "tp1": 1.2,
        "tp2": 2.0,
        "sl": 0.8,
    },
    "trend": {
        "timeframe": "1h",
        "hold_time": 240,
        "tp1": 2.0,
        "tp2": 3.5,
        "sl": 1.2,
    }
}

# ========================
# 📊 WATCHLIST (50 PAIRS)
# ========================
WATCHLIST = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "AVAX", "DOGE", "MATIC", "DOT",
    "LINK", "LTC", "UNI", "ATOM", "FIL",
    "NEAR", "APT", "ARB", "OP", "INJ",
    "SUI", "TIA", "WLD", "PEPE", "SHIB",
    "RNDR", "FET", "TAO", "JUP", "PYTH",
    "ALGO", "SAND", "MANA", "AXS", "ICP",
    "AAVE", "MKR", "GRT", "LDO", "STX",
    "FTM", "WIF", "BONK", "FLOKI", "ORDI",
    "SEI", "RUNE", "CFX", "PENDLE", "ZETA"
]

# ========================
# ⚙️ RISK MANAGEMENT
# ========================
MIN_RR_RATIO = 1.5  # Lowered from 1.8 for more signals
MAX_POSITION_SIZE = 0.15  # 15% of total capital
LIQUIDATION_BUFFER = 0.25  # 25% buffer from liq price
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_AFTER_LOSS = 30  # minutes

# ========================
# 🛡️ MARKET HEALTH FILTERS
# ========================
BTC_VOLATILITY_THRESHOLD = 3.0  # Increased from 2.5 for more flexibility
FUNDING_RATE_EXTREME = 0.05  # ±0.05% threshold
FEAR_GREED_EXTREME = [10, 90]  # Avoid <10 or >90
MIN_VOLUME_24H = 500000  # Lowered from 1M
MAX_SPREAD_PERCENT = 0.2  # Increased from 0.15

# ========================
# ⏰ TIME FILTERS
# ========================
AVOID_NEWS_HOURS = [
    (time(17, 30), time(18, 30)),  # US Market Open
    (time(13, 0), time(13, 30)),   # Asian Lunch
]
AVOID_WEEKENDS = False  # CoinDCX trades 24/7

# ========================
# 📈 TECHNICAL PARAMETERS
# ========================
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
RSI_PERIOD = 14
RSI_OVERSOLD = 35  # Loosened from 30
RSI_OVERBOUGHT = 65  # Loosened from 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
VOLUME_MA_PERIOD = 20

# ========================
# 📱 TELEGRAM SETTINGS
# ========================
SEND_CHART_WITH_SIGNAL = True
PERFORMANCE_REPORT_INTERVAL = 3600  # Every hour
EMOJI_CONFIG = {
    "long": "🟢",
    "short": "🔴",
    "tp1": "🎯",
    "tp2": "💰",
    "sl": "🛑",
    "win": "✅",
    "loss": "❌",
    "alert": "⚡",
}

# ========================
# 🚀 SYSTEM SETTINGS
# ========================
SCAN_INTERVAL = 90  # Increased to 90 seconds
LOG_LEVEL = "INFO"
ENABLE_BACKTEST_MODE = False
DRY_RUN = False  # Set True for paper trading

# ========================
# 🔧 DEBUG SETTINGS
# ========================
DEBUG_MODE = True  # Show detailed logs
TEST_PAIRS_LIMIT = 30  # Test first 30 pairs
MIN_SIGNAL_SCORE = 3  # Minimum score to generate signal (lowered!)