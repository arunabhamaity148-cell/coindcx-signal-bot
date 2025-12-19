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
    
    # FUTURES MARKETS - Only working pairs
    FUTURES_MARKETS = os.environ.get('FUTURES_MARKETS', 
        'F-BTC_INR,F-ETH_INR,F-SOL_INR,F-MATIC_INR,F-ADA_INR,F-DOGE_INR'
    ).split(',')
    
    # SPOT to FUTURES mapping - Only working pairs
    SPOT_TO_FUTURES_MAP = {
        'B-BTC_USDT': 'F-BTC_INR',
        'B-ETH_USDT': 'F-ETH_INR',
        'B-SOL_USDT': 'F-SOL_INR',
        'B-MATIC_USDT': 'F-MATIC_INR',
        'B-ADA_USDT': 'F-ADA_INR',
        'B-DOGE_USDT': 'F-DOGE_INR',
        'B-DOT_USDT': 'F-DOT_INR',
        'B-TRX_USDT': 'F-TRX_INR',
        'B-SHIB_USDT': 'F-SHIB_INR',
        'B-ARB_USDT': 'F-ARB_INR',
        'B-SUI_USDT': 'F-SUI_INR'
    }
    
    @property
    def MARKETS(self):
        if self.USE_AUTHENTICATED_API and self.COINDCX_API_KEY:
            return self.FUTURES_MARKETS
        else:
            return list(self.SPOT_TO_FUTURES_MAP.keys())

    # Leverage
    LEVERAGE = int(os.environ.get('LEVERAGE', '12'))
    
    # Timeframes
    SIGNAL_TIMEFRAME = '5m'
    TREND_TIMEFRAME = '15m'
    BIAS_TIMEFRAME = '1h'
    
    # SMART SCANNING
    CHECK_INTERVAL_MINUTES = int(os.environ.get('CHECK_INTERVAL_MINUTES', '10'))
    COOLDOWN_MINUTES = int(os.environ.get('COOLDOWN_MINUTES', '30'))
    MAX_SIGNALS_PER_DAY = int(os.environ.get('MAX_SIGNALS_PER_DAY', '12'))
    MAX_SIGNALS_PER_4_HOURS = 3  # Quality control
    
    # DYNAMIC SCORING
    BASE_MIN_SCORE = int(os.environ.get('MIN_SIGNAL_SCORE', '55'))
    HIGH_QUALITY_THRESHOLD = 75
    PERFECT_SETUP_THRESHOLD = 80
    
    # ADVANCED FILTERS
    VOLUME_SURGE_THRESHOLD = 3.0  # 3x average
    WHALE_MOVE_THRESHOLD = 2.0  # 2% single candle
    MIN_ORDER_FLOW_STRENGTH = 0.3  # Strong directional flow
    
    # Risk Management
    ATR_SL_MULTIPLIER = 2.0
    ATR_TP1_MULTIPLIER = 3.0
    ATR_TP2_MULTIPLIER = 5.0
    MIN_RR_RATIO = 1.5
    
    # Technical Indicators
    RSI_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21
    ADX_PERIOD = 14
    ATR_PERIOD = 14
    
    # SMART Filters
    MIN_ADX_THRESHOLD = 12
    MIN_ATR_THRESHOLD = 0.00001
    BLOCK_RANGING_SCORE = 65
    BLOCK_VOLATILE_SCORE = 70
    
    # BTC Check
    ENABLE_BTC_CHECK = os.environ.get('ENABLE_BTC_CHECK', 'true').lower() == 'true'
    BTC_PAIR = 'B-BTC_USDT'
    BTC_CHECK_INTERVAL_MINUTES = 10
    BTC_VOLATILITY_THRESHOLD = 5.0
    BTC_DUMP_THRESHOLD = 3.0
    
    # Data requirements
    MIN_CANDLES_REQUIRED = 50
    
    # MTF
    MTF_STRICT_MODE = False
    REQUIRE_MTF_ALIGNMENT = True
    
    # BONUS SCORING
    VOLUME_SURGE_BONUS = 15
    WHALE_CANDLE_BONUS = 10
    LIQUIDITY_SWEEP_BONUS = 12
    TIME_OF_DAY_MULTIPLIER = True
    
    # CONFIRMATIONS REQUIRED (multi-layer)
    REQUIRE_VOLUME_OR_WHALE = True  # At least one
    PERFECT_SETUP_INSTANT_SEND = True  # Score 80+ bypasses some checks