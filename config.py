import os

class Config:
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

    COINDCX_BASE_URL = "https://public.coindcx.com"
    COINDCX_API_KEY = os.environ.get('COINDCX_API_KEY', '')
    COINDCX_API_SECRET = os.environ.get('COINDCX_API_SECRET', '')
    
    USE_AUTHENTICATED_API = os.environ.get('USE_AUTHENTICATED_API', 'true').lower() == 'true'
    
    # Working pairs only
    FUTURES_MARKETS = os.environ.get('FUTURES_MARKETS', 
        'F-BTC_INR,F-ETH_INR,F-SOL_INR,F-MATIC_INR,F-ADA_INR,F-DOGE_INR'
    ).split(',')
    
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

    LEVERAGE = int(os.environ.get('LEVERAGE', '12'))
    
    # Timeframes
    SIGNAL_TIMEFRAME = '5m'
    TREND_TIMEFRAME = '15m'
    BIAS_TIMEFRAME = '1h'
    
    # COINDCX OPTIMIZED - More signals, realistic thresholds
    CHECK_INTERVAL_MINUTES = int(os.environ.get('CHECK_INTERVAL_MINUTES', '15'))
    COOLDOWN_MINUTES = int(os.environ.get('COOLDOWN_MINUTES', '45'))
    MAX_SIGNALS_PER_DAY = 20
    
    # RELAXED SCORING - CoinDCX style
    MIN_SIGNAL_SCORE = int(os.environ.get('MIN_SIGNAL_SCORE', '45'))
    HIGH_QUALITY_THRESHOLD = 60
    PERFECT_SETUP_THRESHOLD = 70
    
    # Risk Management - INR specific
    ATR_SL_MULTIPLIER = 1.8
    ATR_TP1_MULTIPLIER = 2.5
    ATR_TP2_MULTIPLIER = 4.0
    MIN_RR_RATIO = 1.3
    
    # Technical Indicators
    RSI_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21
    ADX_PERIOD = 14
    ATR_PERIOD = 14
    
    # COINDCX FRIENDLY FILTERS - Much more relaxed
    MIN_ADX_THRESHOLD = 8  # INR markets often range
    MIN_ATR_THRESHOLD = 0.000001  # Very low for INR
    BLOCK_RANGING_SCORE = 40  # Allow ranging trades
    BLOCK_VOLATILE_SCORE = 50  # Allow volatile trades
    
    # BTC Check - Relaxed
    ENABLE_BTC_CHECK = os.environ.get('ENABLE_BTC_CHECK', 'false').lower() == 'true'
    BTC_PAIR = 'B-BTC_USDT'
    BTC_CHECK_INTERVAL_MINUTES = 15
    BTC_VOLATILITY_THRESHOLD = 8.0  # Higher tolerance
    BTC_DUMP_THRESHOLD = 5.0  # Only extreme dumps
    
    MIN_CANDLES_REQUIRED = 40  # Lower requirement
    
    # MTF - VERY RELAXED for CoinDCX
    MTF_STRICT_MODE = False
    REQUIRE_MTF_ALIGNMENT = False  # DISABLED - price action priority
    
    # BONUSES - Optional, not required
    VOLUME_SURGE_BONUS = 10
    WHALE_CANDLE_BONUS = 8
    LIQUIDITY_SWEEP_BONUS = 8
    TIME_OF_DAY_MULTIPLIER = False  # DISABLED
    
    # NO FORCED CONFIRMATIONS - Let price action speak
    REQUIRE_VOLUME_OR_WHALE = False
    PERFECT_SETUP_INSTANT_SEND = False

