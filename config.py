"""
🔐 CoinDCX Futures Trading Bot - CONFIG (FINAL STABLE)
Bangla Version 🇧🇩 | Railway + CoinDCX Ready
"""

import os
import logging
from datetime import time

# ========================
# 🔧 LOGGING / DEBUG
# ========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(message)s"
)

DEBUG_MODE = True   # 🔥 block reasons log e dekhabe

# ========================
# 🔑 API CREDENTIALS
# ========================
COINDCX_API_KEY = os.getenv("COINDCX_API_KEY", "")
COINDCX_SECRET = os.getenv("COINDCX_SECRET", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ========================
# 💰 TRADING PARAMETERS
# ========================
MARGIN_PER_TRADE = 3000      # ₹3000
LEVERAGE = 5                # ONLY 5x (safe)
TARGET_DAILY_PROFIT = 2000
MAX_DAILY_SIGNALS = 15
MIN_DAILY_SIGNALS = 5       # 🔥 loosened (10 → 5)

# ========================
# 🎯 SIGNAL MODES
# ========================
SIGNAL_MODES = {
    "quick": {
        "timeframe": "5m",
        "hold_time": 10,
        "tp1": 0.5,
        "tp2": 1.0,
        "sl": 0.4,
    },
    "mid": {
        "timeframe": "15m",
        "hold_time": 45,
        "tp1": 1.0,
        "tp2": 1.8,
        "sl": 0.7,
    },
    "trend": {
        "timeframe": "1h",
        "hold_time": 180,
        "tp1": 1.8,
        "tp2": 3.0,
        "sl": 1.2,
    }
}

# ========================
# 📊 WATCHLIST (SYMBOL ONLY)
# ========================
WATCHLIST = [
    "BTC","ETH","BNB","SOL","XRP","ADA","AVAX","DOGE","MATIC","DOT",
    "LINK","LTC","UNI","ATOM","FIL","NEAR","APT","ARB","OP","INJ",
    "SUI","TIA","WLD","PEPE","SHIB","RNDR","FET","TAO","JUP","PYTH"
]

TEST_PAIRS_LIMIT = 30  # Railway load control

# ========================
# ⚙️ RISK MANAGEMENT
# ========================
MIN_RR_RATIO = 1.5          # 🔥 relaxed (signals asbe)
LIQUIDATION_BUFFER = 0.30   # 30% distance from liq
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_AFTER_LOSS = 30    # minutes

# ========================
# 🛡️ MARKET HEALTH FILTERS
# ========================
BTC_VOLATILITY_THRESHOLD = 3.5   # 🔥 relaxed
FEAR_GREED_EXTREME = (10, 90)
MIN_VOLUME_24H = 300_000         # 🔥 relaxed
MAX_SPREAD_PERCENT = 0.25

# ========================
# ⏰ TIME FILTERS
# ========================
AVOID_NEWS_HOURS = [
    (time(17, 30), time(18, 30)),  # US open
]

AVOID_WEEKENDS = False  # Crypto 24/7

# ========================
# 📈 TECHNICAL PARAMETERS
# ========================
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
RSI_OVERSOLD = 35        # 🔥 loosened
RSI_OVERBOUGHT = 65

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STD = 2

VOLUME_MA_PERIOD = 20

# ========================
# 📱 TELEGRAM
# ========================
SEND_CHART_WITH_SIGNAL = True
PERFORMANCE_REPORT_INTERVAL = 3600

EMOJI_CONFIG = {
    "long": "🟢 LONG",
    "short": "🔴 SHORT",
    "tp1": "🎯 TP1",
    "tp2": "💰 TP2",
    "sl": "🛑 SL",
    "alert": "⚡ SIGNAL",
    "win": "✅",
    "loss": "❌",
}

# ========================
# 🚀 SYSTEM
# ========================
SCAN_INTERVAL = 90        # seconds
ENABLE_BACKTEST_MODE = False
DRY_RUN = False

# ========================
# 🔥 SIGNAL ENGINE
# ========================
MIN_SIGNAL_SCORE = 2      # 🔥 VERY IMPORTANT