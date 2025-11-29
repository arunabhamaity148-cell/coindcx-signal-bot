# config.py
# 75 % win-rate bot settings – tune here only

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
    "FTMUSDT", "AVAXUSDT", "MATICUSDT", "ATOMUSDT", "NEARUSDT"
]

MODE       = "quick"   # quick / mid / trend
TIMEFRAME  = "15m"     # Binance kline interval
PASS_SCORE = 85        # min score for signal

# Risk-Reward ( % )
SL_PCT  = 0.20   # Stop-loss distance
TP1_PCT = 0.20   # 1st target (50 % size)
TP2_PCT = 0.40   # 2nd target (50 % size)

# Guards on/off (True = enable)
USE_NEWS_GUARD   = True
USE_FUNDING_GUARD= True
USE_SPREAD_GUARD = True
USE_MARKET_AWAKE = True
