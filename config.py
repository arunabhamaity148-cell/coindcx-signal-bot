"""
🔐 CoinDCX Futures Trading Bot - Configuration
Bangla Version for Best Friend 🇧🇩
FINAL FIXED & STABLE VERSION ✅
"""

import os
from datetime import time   # ✅ REQUIRED

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
MARGIN_PER_TRADE = 3000
LEVERAGE = 5
TARGET_DAILY_PROFIT = 2000
MAX_DAILY_SIGNALS = 15
MIN_DAILY_SIGNALS = 10

# ========================
# 🎯 SIGNAL MODES
# ========================
SIGNAL_MODES = {
    "quick": {
        "timeframe": "5m",
        "hold_time": 15,
        "tp1": 0.6,
        "tp2": 1.2,
        "sl": 0.4,
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
# 📊 COINDCX WATCHLIST (20 PAIRS)
# ========================
WATCHLIST = [
    "B-BTC_USDT", "B-ETH_USDT", "B-BNB_USDT", "B-SOL_USDT",
    "B-XRP_USDT", "B-ADA_USDT", "B-DOGE_USDT", "B-AVAX_USDT",
    "B-MATIC_USDT", "B-DOT_USDT", "B-LINK_USDT", "B-LTC_USDT",
    "B-UNI_USDT", "B-ATOM_USDT", "B-NEAR_USDT", "B-TRX_USDT",
    "B-SHIB_USDT", "B-FTM_USDT", "B-ALGO_USDT", "B-VET_USDT",
]

# ========================
# ⚙️ RISK MANAGEMENT
# ========================
MIN_RR_RATIO = 1.8
MAX_POSITION_SIZE = 0.15
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_AFTER_LOSS = 30  # minutes

# ========================
# 🛡️ MARKET HEALTH FILTERS
# ========================
BTC_VOLATILITY_THRESHOLD = 2.5
FEAR_GREED_EXTREME = (10, 90)
MIN_VOLUME_24H = 1_000_000
MAX_SPREAD_PERCENT = 0.15

# ========================
# ⏰ NEWS / TIME FILTER (🔥 FIX)
# ========================
AVOID_NEWS_HOURS = [
    (time(17, 30), time(18, 30)),  # US Market Open
    (time(13, 0), time(13, 30)),   # Asia Lunch
]

# ========================
# 📈 TECHNICAL PARAMETERS
# ========================
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

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
PERFORMANCE_REPORT_INTERVAL = 3600

EMOJI_CONFIG = {
    "long": "🟢 LONG",
    "short": "🔴 SHORT",
    "tp1": "🎯 TP1",
    "tp2": "💰 TP2",
    "sl": "🛑 SL",
    "win": "✅ WIN",
    "loss": "❌ LOSS",
    "alert": "⚡ SIGNAL",
}

# ========================
# 🚀 SYSTEM SETTINGS
# ========================
SCAN_INTERVAL = 60
LOG_LEVEL = "INFO"
DRY_RUN = False
ENABLE_BACKTEST_MODE = False