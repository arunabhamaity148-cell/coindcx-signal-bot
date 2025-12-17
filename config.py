"""
Configuration file for CoinDCX Smart Trading Bot
All settings controlled via environment variables
"""
import os

# ============ TELEGRAM CONFIG ============
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ============ TRADING MARKETS ============
MARKETS_STR = os.getenv("COINDCX_MARKETS", "B-BTC_USDT,B-ETH_USDT,B-SOL_USDT,B-XRP_USDT")
MARKETS = [m.strip() for m in MARKETS_STR.split(",") if m.strip()]

# ============ CANDLE SETTINGS ============
CANDLE_INTERVAL = os.getenv("CANDLE_INTERVAL", "15m")
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", "120"))

# ============ SIGNAL SETTINGS ============
MIN_SIGNAL_SCORE = int(os.getenv("MIN_SIGNAL_SCORE", "75"))
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "15"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "30"))
MAX_SIGNALS_PER_DAY = int(os.getenv("MAX_SIGNALS_PER_DAY", "10"))

# ============ FEATURE FLAGS ============
ENABLE_SMART_LOGIC = os.getenv("ENABLE_SMART_LOGIC", "true").lower() == "true"
ENABLE_VOLUME_FILTER = os.getenv("ENABLE_VOLUME_FILTER", "true").lower() == "true"
ENABLE_PATTERN_DETECTION = os.getenv("ENABLE_PATTERN_DETECTION", "true").lower() == "true"

# ============ RISK MANAGEMENT ============
ATR_MULTIPLIER_SL = float(os.getenv("ATR_MULTIPLIER_SL", "1.5"))
ATR_MULTIPLIER_TP1 = float(os.getenv("ATR_MULTIPLIER_TP1", "2.0"))
ATR_MULTIPLIER_TP2 = float(os.getenv("ATR_MULTIPLIER_TP2", "3.5"))
RISK_REWARD_MIN = float(os.getenv("RISK_REWARD_MIN", "1.5"))

# ============ INDICATOR PERIODS ============
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
EMA_TREND = int(os.getenv("EMA_TREND", "200"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))

# ============ SMART LOGIC WEIGHTS ============
WEIGHT_REGULAR_SCORE = float(os.getenv("WEIGHT_REGULAR_SCORE", "0.6"))
WEIGHT_SMART_SCORE = float(os.getenv("WEIGHT_SMART_SCORE", "0.4"))

# Validation
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("⚠️ TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set!")

if not MARKETS:
    raise ValueError("⚠️ COINDCX_MARKETS must be set!")