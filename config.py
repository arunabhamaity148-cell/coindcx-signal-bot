import os
from dataclasses import dataclass

@dataclass
class Config:
    """Multi-Mode Bot Configuration: QUICK + MID + TREND Together"""

    # API Keys (Railway environment variables)
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET')
    CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY')
    CHATGPT_MODEL = os.getenv('CHATGPT_MODEL', 'gpt-4o-mini')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    # Bot Settings
    AUTO_TRADE = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
    
    # ✅ MULTI-MODE: All 3 timeframes active
    MULTI_MODE_ENABLED = True
    ACTIVE_MODES = ['QUICK', 'MID', 'TREND']
    
    MODE = 'QUICK'  # Default (for backward compatibility)
    MAX_SIGNALS_PER_DAY = 18  # 6 per mode (3 modes × 6 = 18 total)
    MAX_LEVERAGE = 15

    # Trading Pairs - CoinDCX format
    PAIRS = [
        'BTCUSDT',
        'ETHUSDT',
        'SOLUSDT',
        'XRPUSDT',
        'ADAUSDT',
        'DOGEUSDT',
        'BNBUSDT',
        'BCHUSDT',
        'SUIUSDT'
    ]

    # ✅ FIXED: Bigger ATR multipliers = Wider SL = Pass liquidation check
    MODE_CONFIG = {
        'QUICK': {
            'timeframe': '5m',
            'ema_fast': 8,
            'ema_slow': 21,
            'leverage': 5,
            'atr_sl_multiplier': 4.0,   # ✅ Was 1.5 → Now 4.0 (wider SL)
            'atr_tp1_multiplier': 8.0,   # 2:1 Risk:Reward
            'atr_tp2_multiplier': 12.0,  # 3:1 Risk:Reward
            'min_score': 35,             # ✅ Lowered threshold
            'min_adx': 12                # ✅ Very low for quick signals
        },
        'MID': {
            'timeframe': '15m',
            'ema_fast': 12,
            'ema_slow': 26,
            'leverage': 7,
            'atr_sl_multiplier': 4.5,   # ✅ Was 2.0 → Now 4.5
            'atr_tp1_multiplier': 9.0,
            'atr_tp2_multiplier': 13.5,
            'min_score': 40,
            'min_adx': 15
        },
        'TREND': {
            'timeframe': '1h',
            'ema_fast': 20,
            'ema_slow': 50,
            'leverage': 10,
            'atr_sl_multiplier': 5.0,   # ✅ Was 2.5 → Now 5.0
            'atr_tp1_multiplier': 10.0,
            'atr_tp2_multiplier': 15.0,
            'min_score': 45,
            'min_adx': 18
        }
    }

    # Risk Management - VERY RELAXED
    LIQUIDATION_BUFFER = 0.008  # 0.8% minimum (was 1.5%)
    MIN_ADX_STRENGTH = 12       # Very low (was 20)
    COOLDOWN_MINUTES = 15       # Short cooldown (was 30)

    # Position sizing (% of capital per trade)
    POSITION_SIZE_PERCENT = 2.0
    MAX_CONCURRENT_TRADES = 6   # Can have 2 per mode

    # Trap Detection Thresholds - RELAXED
    LIQUIDITY_WICK_RATIO = 2.8   # More lenient (was 2.0)
    NEWS_SPIKE_THRESHOLD = 0.06  # 6% (was 3%)
    MAX_SPREAD_PERCENT = 1.0     # 1% (was 0.5%)

    # ✅ RELAXED RSI Thresholds
    RSI_OVERBOUGHT = 78  # Was 70
    RSI_OVERSOLD = 22    # Was 30

    # Score requirements
    MIN_SIGNAL_SCORE = 35

    # Power Hours (IST) - Extended
    POWER_HOURS = [
        (9, 13),   # 9 AM - 1 PM
        (14, 19)   # 2 PM - 7 PM
    ]
    USE_POWER_HOURS = False  # Set to False to trade 24/7

    # WebSocket
    WS_URL = "wss://stream.coindcx.com"
    WS_RECONNECT_DELAY = 5

    # CoinDCX REST API
    COINDCX_BASE_URL = "https://api.coindcx.com"

    # Performance tracking
    TRACK_PERFORMANCE = True
    PERFORMANCE_LOG_FILE = "signal_performance.csv"

    # ChatGPT settings
    CHATGPT_MAX_RETRIES = 2
    CHATGPT_TIMEOUT = 10
    CHATGPT_FALLBACK_ENABLED = True

    @classmethod
    def validate(cls):
        """Validate all required environment variables"""
        required = [
            'COINDCX_API_KEY',
            'COINDCX_SECRET',
            'CHATGPT_API_KEY',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]

        missing = []
        for var in required:
            if not getattr(cls, var):
                missing.append(var)

        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

        print("✅ Configuration validated")
        if cls.MULTI_MODE_ENABLED:
            print(f"🎯 Multi-Mode: {', '.join(cls.ACTIVE_MODES)}")
        else:
            print(f"📊 Mode: {cls.MODE}")
        print(f"📈 Max signals: {cls.MAX_SIGNALS_PER_DAY}/day")
        print(f"⚡ Min ADX: {cls.MIN_ADX_STRENGTH}")
        print(f"🛡️ Liquidation buffer: {cls.LIQUIDATION_BUFFER*100}%")
        print(f"⏰ Cooldown: {cls.COOLDOWN_MINUTES} minutes")
        return True

    @classmethod
    def get_min_score(cls, mode=None):
        """Get minimum score based on mode"""
        if mode and mode in cls.MODE_CONFIG:
            return cls.MODE_CONFIG[mode].get('min_score', cls.MIN_SIGNAL_SCORE)
        return cls.MIN_SIGNAL_SCORE

config = Config()