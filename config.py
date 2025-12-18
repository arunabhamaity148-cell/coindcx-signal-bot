import os

class Config:
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

    COINDCX_BASE_URL = "https://public.coindcx.com"

    # INR FUTURES MARKETS (Execution pairs - analysis will be on SPOT)
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
    COOLDOWN_MINUTES = int(os.environ.get('COOLDOWN_MINUTES', '20'))
    MAX_SIGNALS_PER_DAY = int(os.environ.get('MAX_SIGNALS_PER_DAY', '999'))

    # Scoring (RELAXED for INR futures)
    MIN_SIGNAL_SCORE = int(os.environ.get('MIN_SIGNAL_SCORE', '55'))
    HIGH_QUALITY_THRESHOLD = 70
    MEDIUM_QUALITY_THRESHOLD = 55

    # Risk Management
    ATR_SL_MULTIPLIER = 2.0
    ATR_TP1_MULTIPLIER = 3.0
    ATR_TP2_MULTIPLIER = 5.0
    MIN_RR_RATIO = 1.3

    # Technical Indicators
    RSI_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21
    ADX_PERIOD = 14
    ATR_PERIOD = 14

    # Relaxed Filters for INR Futures
    MIN_ADX_THRESHOLD = 12        # Lowered from 15
    MIN_ATR_THRESHOLD = 0.00001   # Works with INR altcoins

    # Regime scoring thresholds (relaxed)
    BLOCK_RANGING_SCORE = 65
    BLOCK_VOLATILE_SCORE = 60

    # BTC Check (DISABLED by default)
    ENABLE_BTC_CHECK = os.environ.get('ENABLE_BTC_CHECK', 'false').lower() == 'true'
    BTC_PAIR = 'B-BTC_USDT'  # Spot BTC for safety
    BTC_CHECK_INTERVAL_MINUTES = 10

    # Candle data requirement (RELAXED)
    MIN_CANDLES_REQUIRED = 50

    # MTF Configuration (RELAXED mode)
    MTF_STRICT_MODE = False