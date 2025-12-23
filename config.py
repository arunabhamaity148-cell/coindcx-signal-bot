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

    # ✅ FINAL TUNED: Compact SL-TP + High Quality
    MODE_CONFIG = {
        'QUICK': {
            'timeframe': '5m',
            'ema_fast': 8,
            'ema_slow': 21,
            'leverage': 5,
            'atr_sl_multiplier': 1.8,    # ✅ 4.0 → 1.8
            'atr_tp1_multiplier': 3.6,   # 2:1 R:R
            'atr_tp2_multiplier': 5.4,   # 3:1 R:R
            'min_score': 55,             # ✅ 35 → 55
            'min_adx': 20                # ✅ 12 → 20
        },
        'MID': {
            'timeframe': '15m',
            'ema_fast': 12,
            'ema_slow': 26,
            'leverage': 7,
            'atr_sl_multiplier': 2.0,    # ✅ 4.5 → 2.0
            'atr_tp1_multiplier': 4.0,
            'atr_tp2_multiplier': 6.0,
            'min_score': 60,             # ✅ 40 → 60
            'min_adx': 22                # ✅ 15 → 22
        },
        'TREND': {
            'timeframe': '1h',
            'ema_fast': 20,
            'ema_slow': 50,
            'leverage': 10,
            'atr_sl_multiplier': 2.2,    # ✅ 5.0 → 2.2
            'atr_tp1_multiplier': 4.4,
            'atr_tp2_multiplier': 6.6,
            'min_score': 65,             # ✅ 45 → 65
            'min_adx': 25                # ✅ 18 → 25
        }
    }

    # Risk Management - TIGHTENED
    LIQUIDATION_BUFFER = 0.012   # ✅ 0.8 % → 1.2 %
    MIN_ADX_STRENGTH = 20        # ✅ 12 → 20
    COOLDOWN_MINUTES = 8         # ✅ 15 → 8

    # Position sizing
    POSITION_SIZE_PERCENT = 1.5  # ✅ 2.0 → 1.5 (safer)
    MAX_CONCURRENT_TRADES = 6

    # Trap Detection - Balanced
    LIQUIDITY_WICK_RATIO = 2.5   # ✅ 2.8 → 2.5
    NEWS_SPIKE_THRESHOLD = 0.05  # ✅ 6 % → 5 %
    MAX_SPREAD_PERCENT = 0.8     # ✅ 1.0 → 0.8 %

    # ✅ TIGHTENED RSI Thresholds
    RSI_OVERBOUGHT = 72  # ✅ 78 → 72
    RSI_OVERSOLD = 28    # ✅ 22 → 28

    # Score
    MIN_SIGNAL_SCORE = 35

    # Power Hours
    POWER_HOURS = [(9, 13), (14, 19)]
    USE_POWER_HOURS = False

    # WebSocket & API
    WS_URL = "wss://stream.coindcx.com"
    WS_RECONNECT_DELAY = 5
    COINDCX_BASE_URL = "https://api.coindcx.com"

    # Performance
    TRACK_PERFORMANCE = True
    PERFORMANCE_LOG_FILE = "signal_performance.csv"

    # ChatGPT
    CHATGPT_MAX_RETRIES = 2
    CHATGPT_TIMEOUT = 10
    CHATGPT_FALLBACK_ENABLED = True

    @classmethod
    def validate(cls):
        required = [
            'COINDCX_API_KEY',
            'COINDCX_SECRET',
            'CHATGPT_API_KEY',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]
        missing = [v for v in required if not getattr(cls, v)]
        if missing:
            raise ValueError(f"Missing env vars: {', '.join(missing)}")
        print("✅ Configuration validated")
        print(f"🎯 Multi-Mode: {', '.join(cls.ACTIVE_MODES)}")
        print(f"📈 Max signals: {cls.MAX_SIGNALS_PER_DAY}/day")
        print(f"⚡ Min ADX: {cls.MIN_ADX_STRENGTH}")
        print(f"🛡️ Liquidation buffer: {cls.LIQUIDATION_BUFFER*100}%")
        print(f"⏰ Cooldown: {cls.COOLDOWN_MINUTES} minutes")
        return True

    @classmethod
    def get_min_score(cls, mode=None):
        if mode and mode in cls.MODE_CONFIG:
            return cls.MODE_CONFIG[mode].get('min_score', cls.MIN_SIGNAL_SCORE)
        return cls.MIN_SIGNAL_SCORE

config = Config()
