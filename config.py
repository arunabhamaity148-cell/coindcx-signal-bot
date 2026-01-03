import os
from dataclasses import dataclass

@dataclass
class Config:
    """
    Multi-Mode Bot Configuration with Enhanced Risk Management
    """

    # API Keys
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET')
    CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY')
    CHATGPT_MODEL = os.getenv('CHATGPT_MODEL', 'gpt-4o-mini')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    # Bot Settings
    AUTO_TRADE = os.getenv('AUTO_TRADE', 'false').lower() == 'true'

    # Multi-Mode
    MULTI_MODE_ENABLED = True
    ACTIVE_MODES = ['QUICK', 'MID', 'TREND']

    MODE = 'QUICK'
    MAX_SIGNALS_PER_DAY = 24
    MAX_LEVERAGE = 15

    # Trading Pairs
    PAIRS = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
        'DOGEUSDT', 'BNBUSDT', 'BCHUSDT', 'SUIUSDT', 'MATICUSDT',
        'LINKUSDT', 'ATOMUSDT', 'LTCUSDT', 'TRXUSDT'
    ]

    SCAN_INTERVAL_SECONDS = 90

    # ✅ UPDATED: Realistic TP Multipliers
    MODE_CONFIG = {
        'QUICK': {
            'timeframe': '5m',
            'ema_fast': 8,
            'ema_slow': 21,
            'leverage': 5,
            'atr_sl_multiplier': 1.8,
            'atr_tp1_multiplier': 2.7,   # Reduced from 3.6
            'atr_tp2_multiplier': 3.6,   # Reduced from 5.4
            'min_score': 55,
            'min_adx': 25
        },
        'MID': {
            'timeframe': '15m',
            'ema_fast': 12,
            'ema_slow': 26,
            'leverage': 7,
            'atr_sl_multiplier': 2.0,
            'atr_tp1_multiplier': 3.0,   # Reduced from 4.0
            'atr_tp2_multiplier': 4.5,   # Reduced from 6.0
            'min_score': 60,
            'min_adx': 22
        },
        'TREND': {
            'timeframe': '1h',
            'ema_fast': 20,
            'ema_slow': 50,
            'leverage': 10,
            'atr_sl_multiplier': 2.2,
            'atr_tp1_multiplier': 3.5,   # Reduced from 4.4
            'atr_tp2_multiplier': 5.5,   # Reduced from 6.6
            'min_score': 65,
            'min_adx': 25
        }
    }

    # ✅ NEW: Partial Profit Booking Configuration
    PARTIAL_BOOKING = {
        'TREND': {
            'tp1': (0.8, 50),   # 0.8% profit = book 50%
            'tp2': (1.5, 30),   # 1.5% profit = book 30%
            'tp3': (2.5, 20)    # 2.5% profit = book 20%
        },
        'MID': {
            'tp1': (0.5, 50),
            'tp2': (1.0, 30),
            'tp3': (1.8, 20)
        },
        'QUICK': {
            'tp1': (0.3, 70),
            'tp2': (0.6, 20),
            'tp3': (1.0, 10)
        },
        'SCALP': {
            'tp1': (0.2, 80),
            'tp2': (0.4, 15),
            'tp3': (0.7, 5)
        }
    }

    # ✅ NEW: Trailing Stop Configuration
    TRAILING_CONFIG = {
        'breakeven_profit': 0.4,        # Move to breakeven at 0.4%
        'lock_50_profit': 0.8,          # Lock 50% profit at 0.8%
        'lock_70_profit': 2.0,          # Lock 70% profit at 2.0%
        'trail_atr_multiplier': {
            'TREND': 2.0,
            'MID': 1.5,
            'QUICK': 1.2,
            'SCALP': 1.0
        }
    }

    # ✅ NEW: Time-Based Exit Limits (hours)
    TIME_BASED_EXIT = {
        'TREND': 16,
        'MID': 8,
        'QUICK': 4,
        'SCALP': 2
    }

    # ✅ NEW: Risk Limits
    RISK_LIMITS = {
        'daily_loss_percent': 3.0,      # Kill switch at 3% daily loss
        'max_positions': 5,             # Max positions per day
        'capital_per_trade': 2.0,       # 2% risk per trade
        'max_position_size_pct': 20.0   # Never use > 20% capital
    }

    # Risk Management
    LIQUIDATION_BUFFER = 0.012
    MIN_ADX_STRENGTH = 20
    COOLDOWN_MINUTES = 8
    SAME_PAIR_COOLDOWN_MINUTES = 30

    # Position sizing
    POSITION_SIZE_PERCENT = 1.5
    MAX_CONCURRENT_TRADES = 6

    # Trap Detection
    LIQUIDITY_WICK_RATIO = 2.5
    NEWS_SPIKE_THRESHOLD = 0.05
    MAX_SPREAD_PERCENT = 0.8

    # RSI Thresholds
    RSI_OVERBOUGHT = 72
    RSI_OVERSOLD = 28

    # Score
    MIN_SIGNAL_SCORE = 35
    MIN_TP_DISTANCE_PERCENT = 0.8

    # Dynamic decimal precision
    PRICE_DECIMAL_RULES = {
        'low': (1.0, 4),
        'mid': (100.0, 2),
        'high': (float('inf'), 2)
    }

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

    # ChatGPT Final Judge
    CHATGPT_FINAL_JUDGE_ENABLED = True
    CHATGPT_TIMEOUT = 8
    CHATGPT_MAX_RETRIES = 2
    CHATGPT_FALLBACK_APPROVE = False
    CHATGPT_TEMPERATURE = 0.2
    CHATGPT_MAX_TOKENS = 100
    CHATGPT_MIN_CONFIDENCE = 70
    CHATGPT_FALLBACK_ENABLED = True

    @classmethod
    def validate(cls):
        """Validate configuration"""
        required = [
            'COINDCX_API_KEY',
            'COINDCX_SECRET',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]

        if cls.CHATGPT_FINAL_JUDGE_ENABLED:
            required.append('CHATGPT_API_KEY')

        missing = [v for v in required if not getattr(cls, v)]
        if missing:
            raise ValueError(f"❌ Missing env vars: {', '.join(missing)}")

        print("✅ Configuration validated")

        if cls.MULTI_MODE_ENABLED:
            print(f"🎯 Multi-Mode: {', '.join(cls.ACTIVE_MODES)}")
        else:
            print(f"📊 Mode: {cls.MODE}")

        print(f"📊 Tracking: {len(cls.PAIRS)} pairs")
        print(f"⏱️  Scan interval: {cls.SCAN_INTERVAL_SECONDS}s")
        print(f"📈 Max signals: {cls.MAX_SIGNALS_PER_DAY}/day")
        
        # NEW: Display risk limits
        print(f"\n🛡️ RISK MANAGEMENT:")
        print(f"   Daily loss limit: {cls.RISK_LIMITS['daily_loss_percent']}%")
        print(f"   Max positions/day: {cls.RISK_LIMITS['max_positions']}")
        print(f"   Risk per trade: {cls.RISK_LIMITS['capital_per_trade']}%")

        print(f"\n{'='*50}")
        if cls.CHATGPT_FINAL_JUDGE_ENABLED:
            print(f"🤖 ChatGPT Final Judge: ENABLED")
            print(f"   Model: {cls.CHATGPT_MODEL}")
            print(f"   Timeout: {cls.CHATGPT_TIMEOUT}s")
        else:
            print(f"⚠️  ChatGPT Final Judge: DISABLED")
        print(f"{'='*50}\n")

        return True

    @classmethod
    def get_min_score(cls, mode=None):
        """Get minimum score based on mode"""
        if mode and mode in cls.MODE_CONFIG:
            return cls.MODE_CONFIG[mode].get('min_score', cls.MIN_SIGNAL_SCORE)
        return cls.MIN_SIGNAL_SCORE

    @classmethod
    def get_decimal_places(cls, price: float) -> int:
        """Get appropriate decimal places based on price"""
        for rule_name, (threshold, decimals) in cls.PRICE_DECIMAL_RULES.items():
            if price < threshold:
                return decimals
        return 2

config = Config()