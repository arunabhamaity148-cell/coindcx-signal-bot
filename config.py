# ==========================================
# config.py — PRO VERSION
# ==========================================

import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# API CONFIG
# ==========================================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Testnet Toggle
USE_TESTNET = os.getenv("USE_TESTNET", "false").lower() == "true"


# ==========================================
# TRADING PAIRS (ONLY BINANCE — AS YOU WANT)
# ==========================================
TRADING_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT",
    "MATICUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT",
    "FILUSDT", "APTUSDT", "ARBUSDT", "SUIUSDT", "OPUSDT",
    "APEUSDT", "NEARUSDT", "FLOWUSDT", "ALGOUSDT", "HBARUSDT",
    "INJUSDT", "IMXUSDT", "STXUSDT", "RUNEUSDT", "MKRUSDT",
    "CRVUSDT", "AAVEUSDT", "SNXUSDT", "GRTUSDT", "ENSUSDT",
    "MINAUSDT", "PEPEUSDT", "JTOUSDT", "WIFUSDT", "SEIUSDT",
    "ONDOUSDT", "TIAUSDT", "1000FLOKIUSDT", "ORDIUSDT", "BONKUSDT",
    "ARKMUSDT", "DYDXUSDT", "SKLUSDT", "GALAUSDT", "SANDUSDT"
]


# ==========================================
# SIGNAL MINIMUM SCORE (Balanced Mode)
# ==========================================
QUICK_MIN_SCORE = 7.0       # Balanced mode
MID_MIN_SCORE   = 7.0       # Balanced mode
TREND_MIN_SCORE = 7.0       # Balanced mode


# ==========================================
# TP/SL CONFIG (Single TP/SL — আপনার পছন্দ মতো)
# ==========================================

# QUICK = scalping → ছোট Target / ছোট Stop
# MID   = medium swing → মাঝারি TP/SL
# TREND = higher timeframe → বড় TP/SL

TP_SL_CONFIG = {
    "QUICK": {
        "tp": 0.40,   # +0.40% (Fast scalping)
        "sl": 0.20    # -0.20%
    },
    "MID": {
        "tp": 2.00,   # +2%
        "sl": 1.00    # -1%
    },
    "TREND": {
        "tp": 6.00,   # +6%
        "sl": 2.00    # -2%
    }
}


# ==========================================
# LEVERAGE CONFIG (DEFAULT = 50× for QUICK/MID)
# ==========================================
SUGGESTED_LEVERAGE = {
    "QUICK": 30,   # Your preference
    "MID":   30,   # Your preference
    "TREND": 20    # Trend trades are safer with lower leverage
}


# ==========================================
# SAFETY CONFIG
# ==========================================
MIN_SAFE_LIQ_DISTANCE_PCT = 5.0  # SL must be at least 5% away from liquidation

MAX_SPREAD_PCT = 0.05            # Spread must be < 0.05%
MIN_ORDERBOOK_DEPTH = 15000      # Depth requirement (in USD)


# ==========================================
# COOLDOWN CONFIG (PER MODE)
# ==========================================
COOLDOWN_SECONDS = {
    "QUICK": 1800,   # 30 min per coin
    "MID":   3600,   # 60 min per coin
    "TREND": 7200    # 120 min per coin
}


# ==========================================
# BTC VOLATILITY CHECK
# ==========================================
BTC_VOLATILITY_THRESHOLDS = {
    "1m": 0.50,   # Max 0.5% move in 1 min
    "5m": 1.50    # Max 1.5% move in 5 min
}


# ==========================================
# RETRY / BACKOFF CONFIG
# ==========================================
FETCH_RETRY_COUNT = 3
FETCH_RETRY_BACKOFF = 2  # 2s, 4s, 8s (exponential)


# ==========================================
# LOGIC WEIGHTS (Final balanced version)
# ==========================================
LOGIC_WEIGHTS = {

    # -------------------------
    # QUICK MODE (10 Logics)
    # -------------------------
    "RSI_long": 1.3,
    "RSI_short": 1.3,
    "MACD_bull": 1.4,
    "MACD_bear": 1.4,
    "VWAP_long": 1.2,
    "VWAP_short": 1.2,
    "EMA_bull": 1.3,
    "EMA_bear": 1.3,
    "Spread_tight": 1.0,
    "Structure_HH": 1.1,

    # -------------------------
    # MID MODE (10 Logics)
    # -------------------------
    "RSI_os": 1.3,
    "RSI_ob": 1.3,
    "MACD_bull_mid": 1.4,
    "Volume_mid": 1.1,
    "EMA50_touch": 1.3,
    "EMA200_touch": 1.2,
    "FVG_fill": 1.2,
    "Keltner_up": 1.2,
    "Trendline_break": 1.1,
    "ADX_trend": 1.4,

    # -------------------------
    # TREND MODE (10 Logics)
    # -------------------------
    "ST_long": 1.5,
    "ST_short": 1.5,
    "ATR_ok": 1.1,
    "BB_midband": 1.2,
    "Fib_382": 1.3,
    "Fib_50": 1.3,
    "Fib_618": 1.4,
    "Demand_zone": 1.2,
    "Imbalance_cont": 1.2,
    "Chop_exit": 1.1,
}