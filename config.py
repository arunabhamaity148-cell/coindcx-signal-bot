import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

USE_TESTNET = os.getenv("USE_TESTNET", "false").lower() == "true"

# 50 Futures pairs — Binance-supported only
TRADING_PAIRS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","DOTUSDT",
    "AVAXUSDT","LINKUSDT","LTCUSDT","ATOMUSDT","ETCUSDT","FILUSDT","APTUSDT","ARBUSDT",
    "SUIUSDT","OPUSDT","APEUSDT","NEARUSDT","FLOWUSDT","ALGOUSDT","HBARUSDT","INJUSDT",
    "IMXUSDT","STXUSDT","RUNEUSDT","AAVEUSDT","SNXUSDT","GRTUSDT","MKRUSDT",
    "TIAUSDT","SEIUSDT","ONDOUSDT","WIFUSDT","ORDIUSDT","ARKMUSDT","DYDXUSDT",
    "JTOUSDT","MINAUSDT","CRVUSDT","GALAUSDT","SKLUSDT","SANDUSDT","1000FLOKIUSDT"
]

# Score thresholds
QUICK_MIN_SCORE = 5
MID_MIN_SCORE = 5
TREND_MIN_SCORE = 5

# TP/SL % model
TP_SL_CONFIG = {
    "QUICK": {"tp": 0.4, "sl": 0.25},
    "MID":   {"tp": 2.5, "sl": 0.8},
    "TREND": {"tp": 6.0, "sl": 1.5}
}

SUGGESTED_LEVERAGE = {
    "QUICK": 20,
    "MID": 25,
    "TREND": 20
}

MIN_SAFE_LIQ_DISTANCE_PCT = 5.0
MAX_SPREAD_PCT = 0.10
MIN_ORDERBOOK_DEPTH = 3000

COOLDOWN_SECONDS = {
    "QUICK": 1800,
    "MID": 3600,
    "TREND": 7200
}

BTC_VOLATILITY_THRESHOLDS = {
    "1m": 0.50,
    "5m": 1.50
}

# Default logic weights
LOGIC_WEIGHTS = {k: 1.0 for k in [
    "RSI_long","RSI_short","MACD_bull","MACD_bear","VWAP_long","VWAP_short",
    "EMA_bull","EMA_bear","RSI_os","RSI_ob","ST_long","ST_short"
]}