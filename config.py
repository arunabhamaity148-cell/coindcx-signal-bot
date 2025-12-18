import os

class Config:
    # Telegram
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
    
    # CoinDCX
    COINDCX_BASE_URL = "https://public.coindcx.com"
    MARKETS = os.environ.get('COINDCX_MARKETS', 'B-BTC_USDT,B-ETH_USDT,B-SOL_USDT,B-XRP_USDT,B-MATIC_USDT,B-ADA_USDT,B-DOGE_USDT,B-DOT_USDT,B-LTC_USDT,B-LINK_USDT,B-UNI_USDT,B-AVAX_USDT,B-ATOM_USDT,B-TRX_USDT,B-SHIB_USDT').split(',')
    
    # Trading (BALANCED)
    CANDLE_INTERVAL = os.environ.get('CANDLE_INTERVAL', '15m')
    MIN_SIGNAL_SCORE = int(os.environ.get('MIN_SIGNAL_SCORE', '60'))
    CHECK_INTERVAL_MINUTES = int(os.environ.get('CHECK_INTERVAL_MINUTES', '5'))
    COOLDOWN_MINUTES = int(os.environ.get('COOLDOWN_MINUTES', '20'))
    MAX_SIGNALS_PER_DAY = int(os.environ.get('MAX_SIGNALS_PER_DAY', '999'))
    
    # Risk Management
    ATR_SL_MULTIPLIER = 1.5
    ATR_TP1_MULTIPLIER = 2.0
    ATR_TP2_MULTIPLIER = 3.5
    MIN_RR_RATIO = 1.3
    
    # Technical
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    EMA_FAST = 9
    EMA_SLOW = 21
    ADX_PERIOD = 14
    ADX_STRONG = 20
    ATR_PERIOD = 14
    
    # Regime Filters (RELAXED)
    BLOCK_RANGING_MARKETS = False
    BLOCK_HIGH_VOLATILITY = False
    MAX_VOLATILITY_PERCENT = 8.0