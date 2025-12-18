import os

class Config:
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

    COINDCX_BASE_URL = "https://public.coindcx.com"

    # INR SPOT MARKETS (FULLY WORKING)
    MARKETS = os.environ.get('COINDCX_MARKETS',
        'B-BTC_INR,B-ETH_INR,B-SOL_INR,B-BNB_INR,B-XRP_INR,'
        'B-ADA_INR,B-DOGE_INR,B-MATIC_INR,B-DOT_INR,B-TRX_INR,'
        'B-LINK_INR,B-LTC_INR,B-AVAX_INR,B-OP_INR,B-ARB_INR,'
        'B-APT_INR,B-SUI_INR,B-INJ_INR,B-ATOM_INR,B-UNI_INR'
    ).split(',')

    # Leverage (virtual suggestion only)
    LEVERAGE = int(os.environ.get('LEVERAGE', '12'))
    MIN_LEVERAGE = 10
    MAX_LEVERAGE = 15

    # Timeframes
    SIGNAL_TIMEFRAME = '5m'
    TREND_TIMEFRAME = '15m'
    BIAS_TIMEFRAME = '1h'

    # Scan frequency
    CHECK_INTERVAL_MINUTES = int(os.environ.get('CHECK_INTERVAL_MINUTES', '3'))
    COOLDOWN_MINUTES = int(os.environ.get('COOLDOWN_MINUTES', '20'))
    MAX_SIGNALS_PER_DAY = int(os.environ.get('MAX_SIGNALS_PER_DAY', '999'))

    # RELAXED scoring
    MIN_SIGNAL_SCORE = int(os.environ.get('MIN_SIGNAL_SCORE', '55'))
    HIGH_QUALITY_THRESHOLD = 70
    MEDIUM_QUALITY_THRESHOLD = 55

    # Risk management
    ATR_SL_MULTIPLIER = 2.0
    ATR_TP1_MULTIPLIER = 3.0
    ATR_TP2_MULTIPLIER = 5.0
    MIN_RR_RATIO = 1.3

    # Indicators
    RSI_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21
    ADX_PERIOD = 14
    ATR_PERIOD = 14

    # Relaxed filters
    MIN_ADX_THRESHOLD = 12
    MIN_ATR_THRESHOLD = 0.00001

    # Regime thresholds
    BLOCK_RANGING_SCORE = 65
    BLOCK_VOLATILE_SCORE = 60

    # BTC check (optional)
    ENABLE_BTC_CHECK = os.environ.get('ENABLE_BTC_CHECK', 'false').lower() == 'true'
    BTC_PAIR = 'B-BTC_USDT'
    BTC_CHECK_INTERVAL_MINUTES = 10

    # Candle requirement
    MIN_CANDLES_REQUIRED = 50

    # MTF relaxed
    MTF_STRICT_MODE = False