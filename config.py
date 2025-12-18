import os

class Config:
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

    COINDCX_BASE_URL = "https://public.coindcx.com"
    
    # INR FUTURES MARKETS
    MARKETS = os.environ.get('COINDCX_MARKETS', 
        'F-BTC_INR,F-ETH_INR,F-SOL_INR,F-MATIC_INR,F-XRP_INR,'
        'F-ADA_INR,F-DOGE_INR,F-DOT_INR,F-LTC_INR,F-LINK_INR,'
        'F-UNI_INR,F-AVAX_INR,F-ATOM_INR,F-TRX_INR,F-SHIB_INR,'
        'F-ARB_INR,F-OP_INR,F-APT_INR,F-SUI_INR,F-INJ_INR'
    ).split(',')

    # Leverage
    LEVERAGE = int(os.environ.get('LEVERAGE', '12'))
    MIN_LEVERAGE = 10
    MAX_LEVERAGE = 15

    # Timeframes
    SIGNAL_TIMEFRAME = '5m'
    TREND_TIMEFRAME = '15m'
    BIAS_TIMEFRAME = '1h'
    
    # Scanning
    CHECK_INTERVAL_MINUTES = int(os.environ.get('CHECK_INTERVAL_MINUTES', '3'))
    COOLDOWN_MINUTES = int(os.environ.get('COOLDOWN_MINUTES', '30'))
    MAX_SIGNALS_PER_DAY = int(os.environ.get('MAX_SIGNALS_PER_DAY', '999'))

    # Scoring
    MIN_SIGNAL_SCORE = int(os.environ.get('MIN_SIGNAL_SCORE', '70'))
    HIGH_QUALITY_THRESHOLD = 80
    MEDIUM_QUALITY_THRESHOLD = 70

    # Risk Management
    ATR_SL_MULTIPLIER = 1.8
    ATR_TP1_MULTIPLIER = 2.5
    ATR_TP2_MULTIPLIER = 4.0
    MIN_RR_RATIO = 1.5

    # Technical Indicators
    RSI_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21
    ADX_PERIOD = 14
    ATR_PERIOD = 14

    # Smart Filters
    MIN_ADX_THRESHOLD = 15
    BLOCK_RANGING_SCORE = 75
    BLOCK_VOLATILE_SCORE = 80
    
    # BTC Stability
    BTC_PAIR = 'F-BTC_INR'
    BTC_MIN_ADX = 15
    BTC_MAX_VOLATILITY = 3.5
    BTC_CHECK_CANDLES = 10