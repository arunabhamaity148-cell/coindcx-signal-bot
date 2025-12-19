import os

class Config:
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

    # CoinDCX API Configuration
    COINDCX_BASE_URL = "https://public.coindcx.com"
    COINDCX_API_KEY = os.environ.get('COINDCX_API_KEY', '')
    COINDCX_API_SECRET = os.environ.get('COINDCX_API_SECRET', '')
    
    # Trading Mode
    USE_AUTHENTICATED_API = os.environ.get('USE_AUTHENTICATED_API', 'true').lower() == 'true'
    
    # FUTURES MARKETS (Primary)
    FUTURES_MARKETS = os.environ.get('FUTURES_MARKETS', 
        'F-BTC_INR,F-ETH_INR,F-SOL_INR,F-MATIC_INR,F-XRP_INR,'
        'F-ADA_INR,F-DOGE_INR,F-DOT_INR,F-LTC_INR,F-LINK_INR,'
        'F-UNI_INR,F-AVAX_INR,F-ATOM_INR,F-TRX_INR,F-SHIB_INR,'
        'F-ARB_INR,F-OP_INR,F-APT_INR,F-SUI_INR,F-INJ_INR'
    ).split(',')
    
    # SPOT to FUTURES mapping (fallback)
    SPOT_TO_FUTURES_MAP = {
        'B-BTC_USDT': 'F-BTC_INR',
        'B-ETH_USDT': 'F-ETH_INR',
        'B-SOL_USDT': 'F-SOL_INR',
        'B-MATIC_USDT': 'F-MATIC_INR',
        'B-XRP_USDT': 'F-XRP_INR',
        'B-ADA_USDT': 'F-ADA_INR',
        'B-DOGE_USDT': 'F-DOGE_INR',
        'B-DOT_USDT': 'F-DOT_INR',
        'B-LTC_USDT': 'F-LTC_INR',
        'B-LINK_USDT': 'F-LINK_INR',
        'B-UNI_USDT': 'F-UNI_INR',
        'B-AVAX_USDT': 'F-AVAX_INR',
        'B-ATOM_USDT': 'F-ATOM_INR',
        'B-TRX_USDT': 'F-TRX_INR',
        'B-SHIB_USDT': 'F-SHIB_INR',
        'B-ARB_USDT': 'F-ARB_INR',
        'B-OP_USDT': 'F-OP_INR',
        'B-APT_USDT': 'F-APT_INR',
        'B-SUI_USDT': 'F-SUI_INR',
        'B-INJ_USDT': 'F-INJ_INR'
    }
    
    # Get working markets based on mode
    @property
    def MARKETS(self):
        if self.USE_AUTHENTICATED_API and self.COINDCX_API_KEY:
            return self.FUTURES_MARKETS
        else:
            return list(self.SPOT_TO_FUTURES_MAP.keys())

    # Leverage
    LEVERAGE = int(os.environ.get('LEVERAGE', '12'))
    MIN_LEVERAGE = 10
    MAX_LEVERAGE = 15

    # Timeframes
    SIGNAL_TIMEFRAME = '5m'
    TREND_TIMEFRAME = '15m'
    BIAS_TIMEFRAME = '1h'
    
    # Scanning
    CHECK_INTERVAL_MINUTES = int(os.environ.get('CHECK_INTERVAL_MINUTES', '10'))
    COOLDOWN_MINUTES = int(os.environ.get('COOLDOWN_MINUTES', '30'))
    MAX_SIGNALS_PER_DAY = int(os.environ.get('MAX_SIGNALS_PER_DAY', '15'))
    MAX_SIGNALS_PER_SCAN = int(os.environ.get('MAX_SIGNALS_PER_SCAN', '3'))

    # BALANCED SCORING - Quality signals
    MIN_SIGNAL_SCORE = int(os.environ.get('MIN_SIGNAL_SCORE', '60'))
    HIGH_QUALITY_THRESHOLD = 75
    MEDIUM_QUALITY_THRESHOLD = 60
    
    # PRIORITY scoring - send best signals first
    PRIORITY_HIGH_SCORE = 80  # Instant send
    PRIORITY_MEDIUM_SCORE = 70  # Send if slots available

    # Risk Management
    ATR_SL_MULTIPLIER = 2.0
    ATR_TP1_MULTIPLIER = 3.0
    ATR_TP2_MULTIPLIER = 5.0
    MIN_RR_RATIO = 1.5  # Stricter R:R

    # Technical Indicators
    RSI_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21
    ADX_PERIOD = 14
    ATR_PERIOD = 14

    # STRICTER Filters
    MIN_ADX_THRESHOLD = 15  # Back to 15
    MIN_ATR_THRESHOLD = 0.00001
    
    # Regime scoring - STRICTER
    BLOCK_RANGING_SCORE = 70  # Up from 65
    BLOCK_VOLATILE_SCORE = 75  # Up from 60
    
    # BTC Check - ENABLED for quality
    ENABLE_BTC_CHECK = os.environ.get('ENABLE_BTC_CHECK', 'true').lower() == 'true'
    BTC_PAIR = 'B-BTC_USDT'
    BTC_CHECK_INTERVAL_MINUTES = 10
    BTC_VOLATILITY_THRESHOLD = 5.0  # Stricter
    BTC_DUMP_THRESHOLD = 3.0  # Stricter
    
    # Data requirements
    MIN_CANDLES_REQUIRED = 50
    
    # MTF - BALANCED (allow neutral)
    MTF_STRICT_MODE = False  # Allow neutral MTF
    REQUIRE_MTF_ALIGNMENT = True  # Must have MTF alignment