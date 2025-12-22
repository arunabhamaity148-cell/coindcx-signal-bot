import os
from dataclasses import dataclass

@dataclass
class Config:
    """Bot Configuration - Optimized for Moderate Signals (5-8/day)"""

    # API Keys (Railway environment variables)
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET')
    CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY')
    CHATGPT_MODEL = os.getenv('CHATGPT_MODEL', 'gpt-4o-mini')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    # Bot Settings
    AUTO_TRADE = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
    MODE = os.getenv('MODE', 'QUICK')  # QUICK for more signals
    MAX_SIGNALS_PER_DAY = 8  # Reduced from 10 for quality over quantity
    MAX_LEVERAGE = 15

    # Trading Pairs - CoinDCX format
    # 9 pairs = Good variety for catching opportunities
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

    # Mode Configurations - OPTIMIZED FOR MODERATE SIGNALS
    MODE_CONFIG = {
        'QUICK': {
            'timeframe': '5m',
            'ema_fast': 8,        # Changed from 9 (faster response)
            'ema_slow': 21,
            'leverage': 5,        # Reduced from 12 (safer)
            'atr_sl_multiplier': 1.5,
            'atr_tp1_multiplier': 2.5,  # Increased from 2.0 (better R:R)
            'atr_tp2_multiplier': 4.0   # Increased from 3.0
        },
        'MID': {
            'timeframe': '15m',
            'ema_fast': 12,
            'ema_slow': 26,
            'leverage': 7,        # Reduced from 10
            'atr_sl_multiplier': 2.0,
            'atr_tp1_multiplier': 3.0,  # Increased from 2.5
            'atr_tp2_multiplier': 5.0   # Increased from 4.0
        },
        'TREND': {
            'timeframe': '1h',
            'ema_fast': 20,
            'ema_slow': 50,
            'leverage': 10,       # Reduced from 15
            'atr_sl_multiplier': 2.5,
            'atr_tp1_multiplier': 3.5,  # Increased from 3.0
            'atr_tp2_multiplier': 6.0   # Increased from 5.0
        }
    }

    # Risk Management - OPTIMIZED FOR MORE SIGNALS
    LIQUIDATION_BUFFER = 0.015   # Changed from 0.005 (1.5% instead of 0.5%)
    MIN_ADX_STRENGTH = 20        # Changed from 25 (allow weaker trends)
    COOLDOWN_MINUTES = 30        # Kept at 30 (good balance)
    
    # New: Position sizing (% of capital per trade)
    POSITION_SIZE_PERCENT = 2.0  # Risk 2% per trade (conservative)
    MAX_CONCURRENT_TRADES = 3    # Maximum open positions at once

    # Trap Detection Thresholds - SLIGHTLY RELAXED
    LIQUIDITY_WICK_RATIO = 2.2   # Changed from 2.0 (less strict)
    NEWS_SPIKE_THRESHOLD = 0.035 # Changed from 0.03 (3.5% instead of 3%)
    MAX_SPREAD_PERCENT = 0.6     # Changed from 0.5 (slightly higher tolerance)
    
    # New: Score requirements (for signal quality)
    MIN_SIGNAL_SCORE = 50        # Minimum score to generate signal
    MIN_SCORE_QUICK = 45         # Lower threshold for QUICK mode
    MIN_SCORE_MID = 50           # Medium threshold for MID mode
    MIN_SCORE_TREND = 55         # Higher threshold for TREND mode

    # Power Hours (IST) - EXTENDED FOR MORE OPPORTUNITIES
    POWER_HOURS = [
        (9, 13),   # Changed from (10, 13) - Start 1 hour earlier
        (14, 19)   # Changed from (15, 18) - Extended by 1 hour
    ]
    
    # New: Enable/disable power hours check
    USE_POWER_HOURS = False  # Set to False to trade 24/7

    # WebSocket
    WS_URL = "wss://stream.coindcx.com"
    WS_RECONNECT_DELAY = 5  # seconds

    # CoinDCX REST API
    COINDCX_BASE_URL = "https://api.coindcx.com"
    
    # New: Performance tracking
    TRACK_PERFORMANCE = True     # Log all signals for analysis
    PERFORMANCE_LOG_FILE = "signal_performance.csv"
    
    # New: ChatGPT settings
    CHATGPT_MAX_RETRIES = 2      # Retry if ChatGPT fails
    CHATGPT_TIMEOUT = 10         # Seconds to wait for response
    CHATGPT_FALLBACK_ENABLED = True  # Auto-approve if ChatGPT fails

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
        print(f"📊 Mode: {cls.MODE}")
        print(f"🎯 Max signals: {cls.MAX_SIGNALS_PER_DAY}/day")
        print(f"⚡ Min ADX: {cls.MIN_ADX_STRENGTH}")
        print(f"🛡️ Liquidation buffer: {cls.LIQUIDATION_BUFFER*100}%")
        print(f"⏰ Cooldown: {cls.COOLDOWN_MINUTES} minutes")
        return True
    
    @classmethod
    def get_min_score(cls):
        """Get minimum score based on current mode"""
        mode_scores = {
            'QUICK': cls.MIN_SCORE_QUICK,
            'MID': cls.MIN_SCORE_MID,
            'TREND': cls.MIN_SCORE_TREND
        }
        return mode_scores.get(cls.MODE, cls.MIN_SIGNAL_SCORE)

config = Config()