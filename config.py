import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------
# 🔐 API Keys
# ---------------------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

USE_TESTNET = os.getenv("USE_TESTNET", "false").lower() == "true"


# ---------------------------------------
# 🔥 Trading Pairs (Stable 50 Binance Futures Pairs)
# ---------------------------------------
TRADING_PAIRS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","DOTUSDT",
    "AVAXUSDT","LINKUSDT","LTCUSDT","ATOMUSDT","ETCUSDT","FILUSDT","APTUSDT","ARBUSDT",
    "SUIUSDT","OPUSDT","APEUSDT","NEARUSDT","FLOWUSDT","ALGOUSDT","HBARUSDT","INJUSDT",
    "IMXUSDT","STXUSDT","RUNEUSDT","AAVEUSDT","SNXUSDT","GRTUSDT","MKRUSDT",
    "TIAUSDT","SEIUSDT","ONDOUSDT","WIFUSDT","ORDIUSDT","ARKMUSDT","DYDXUSDT",
    "JTOUSDT","MINAUSDT","CRVUSDT","GALAUSDT","SKLUSDT","SANDUSDT","1000FLOKIUSDT"
]


# ---------------------------------------
# 🎯 Minimum Score (AI + Weighted Logic)
# ---------------------------------------
QUICK_MIN_SCORE = 9
MID_MIN_SCORE = 9
TREND_MIN_SCORE = 9


# ---------------------------------------
# 🎯 TP/SL Model (%)
# ---------------------------------------
TP_SL_CONFIG = {
    "QUICK": {"tp": 0.4, "sl": 0.25},
    "MID":   {"tp": 2.5, "sl": 0.8},
    "TREND": {"tp": 6.0, "sl": 1.5}
}


# ---------------------------------------
# 📌 Leverage Suggestion (AI overrides later)
# ---------------------------------------
SUGGESTED_LEVERAGE = {
    "QUICK": 20,
    "MID": 25,
    "TREND": 20
}

SCORE_FOR_50X = 9.0   # score 9+ → 50x leverage override


# ---------------------------------------
# 🔰 Safety System
# ---------------------------------------
MIN_SAFE_LIQ_DISTANCE_PCT = 5.0
MAX_SPREAD_PCT = 0.10
MIN_ORDERBOOK_DEPTH = 3000

BTC_VOLATILITY_THRESHOLDS = {
    "1m": 0.50,
    "5m": 1.50
}


# ---------------------------------------
# 🕒 Cooldown Per Mode
# ---------------------------------------
COOLDOWN_SECONDS = {
    "QUICK": 1800,   # 30 mins
    "MID":   3600,   # 1 hour  
    "TREND": 7200    # 2 hours
}

# Avoid multi-mode spam (per symbol cooldown)
GLOBAL_SYMBOL_COOLDOWN = 1800


# ---------------------------------------
# 🤖 AI Control & Auto-Publish
# ---------------------------------------
ASSISTANT_CONTROLLED_PUBLISH = False     # False → AI allowed → bot auto publish
AUTO_PUBLISH = True                     # Start safe-mode first 24 hours
AI_MIN_CONFIDENCE_TO_SEND = 90           # Only high confidence signals go to Telegram


# ---------------------------------------
# 🧠 Logic Weights (Default)
# ---------------------------------------
LOGIC_WEIGHTS = {
    k: 1.0 for k in [
        "RSI_long","RSI_short","MACD_bull","MACD_bear",
        "VWAP_long","VWAP_short","EMA_bull","EMA_bear",
        "RSI_os","RSI_ob","ST_long","ST_short"
    ]
}


# ---------------------------------------
# 📁 Storage
# ---------------------------------------
PENDING_SIGNALS_FILE = "pending_signals.json"

MAX_TELEGRAM_PER_MINUTE = 6
MAX_TELEGRAM_PER_HOUR = 100