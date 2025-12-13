"""
🔐 CoinDCX Futures Trading Bot - Configuration
Bangla Version for Best Friend 🇧🇩
"""

import os
from datetime import time

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
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT",
    "LINKUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "FILUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "SUIUSDT", "TIAUSDT", "WLDUSDT", "PEPEUSDT", "SHIBUSDT",
    "RNDRUSDT", "FETUSDT", "TAOUSDT", "JUPUSDT", "PYTHUSDT",
    "ALGOUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", "ICPUSDT",
    "AAVEUSDT", "MKRUSDT", "GRTUSDT", "LDOUSDT", "STXUSDT",
    "FTMUSDT", "WIFUSDT", "BONKUSDT", "FLOKIUSDT", "ORDIUSDT",
    "SEIUSDT", "RUNEUSDT", "CFXUSDT", "PENDLEUSDT", "ZETAUSDT"
]

# ========================
# ⚙️ RISK MANAGEMENT
# ========================
MIN_RR_RATIO = 1.8  # Minimum 1:1.8 Risk-Reward
MAX_POSITION_SIZE = 0.15  # 15% of total capital
LIQUIDATION_BUFFER = 0.25  # 25% buffer from liq price
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_AFTER_LOSS = 30  # minutes

# ========================
# 🛡️ MARKET HEALTH FILTERS
# ========================
BTC_VOLATILITY_THRESHOLD = 2.5  # Max 2.5% hourly movement
FUNDING_RATE_EXTREME = 0.05  # ±0.05% threshold
FEAR_GREED_EXTREME = [10, 90]  # Avoid <10 or >90
MIN_VOLUME_24H = 1000000  # $1M minimum volume
MAX_SPREAD_PERCENT = 0.15  # 0.15% max spread

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
SCAN_INTERVAL = 60  # Scan every 60 seconds
LOG_LEVEL = "INFO"
ENABLE_BACKTEST_MODE = False
DRY_RUN = False  # Set True for paper trading