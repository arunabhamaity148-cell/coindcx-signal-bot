import os
from dataclasses import dataclass

@dataclass
class Config:
    """
    Multi-Mode Bot Configuration: QUICK + MID + TREND Together
    
    ✅ NEW: ChatGPT as Final Trade Judge
    - Every signal reviewed by ChatGPT before sending
    - Timeout/error = signal rejected (safety first)
    - Strict JSON-only responses
    """

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

    # ✅ FINAL TUNED: Compact SL-TP + High Quality + FIXED ADX
    MODE_CONFIG = {
        'QUICK': {
            'timeframe': '5m',
            'ema_fast': 8,
            'ema_slow': 21,
            'leverage': 5,
            'atr_sl_multiplier': 1.8,
            'atr_tp1_multiplier': 3.6,   # 2:1 R:R
            'atr_tp2_multiplier': 5.4,   # 3:1 R:R
            'min_score': 55,
            'min_adx': 25  # ✅ FIX #3: Changed from 20 to 25 (stricter)
        },
        'MID': {
            'timeframe': '15m',
            'ema_fast': 12,
            'ema_slow': 26,
            'leverage': 7,
            'atr_sl_multiplier': 2.0,
            'atr_tp1_multiplier': 4.0,
            'atr_tp2_multiplier': 6.0,
            'min_score': 60,
            'min_adx': 22
        },
        'TREND': {
            'timeframe': '1h',
            'ema_fast': 20,
            'ema_slow': 50,
            'leverage': 10,
            'atr_sl_multiplier': 2.2,
            'atr_tp1_multiplier': 4.4,
            'atr_tp2_multiplier': 6.6,
            'min_score': 65,
            'min_adx': 25
        }
    }

    # Risk Management - TIGHTENED
    LIQUIDATION_BUFFER = 0.012   # 1.2%
    MIN_ADX_STRENGTH = 20
    COOLDOWN_MINUTES = 8

    # Position sizing
    POSITION_SIZE_PERCENT = 1.5
    MAX_CONCURRENT_TRADES = 6

    # Trap Detection - Balanced
    LIQUIDITY_WICK_RATIO = 2.5
    NEWS_SPIKE_THRESHOLD = 0.05  # 5%
    MAX_SPREAD_PERCENT = 0.8     # 0.8%

    # ✅ TIGHTENED RSI Thresholds
    RSI_OVERBOUGHT = 72
    RSI_OVERSOLD = 28

    # Score
    MIN_SIGNAL_SCORE = 35

    # ✅ FIX #1 & #4: Minimum TP distance protection for low-price coins
    MIN_TP_DISTANCE_PERCENT = 0.8  # 0.8% minimum TP distance from entry

    # ✅ FIX #1: Dynamic decimal precision based on price
    # For coins < $1: use more decimals to preserve TP accuracy
    PRICE_DECIMAL_RULES = {
        'low': (1.0, 4),      # Price < $1: 4 decimals (e.g., 0.1234)
        'mid': (100.0, 2),    # $1-$100: 2 decimals (e.g., 12.34)
        'high': (float('inf'), 2)  # > $100: 2 decimals (e.g., 12345.67)
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

    # ================================
    # 🤖 CHATGPT FINAL JUDGE SETTINGS
    # ================================
    
    # ChatGPT as Final Trade Judge (MANDATORY)
    CHATGPT_FINAL_JUDGE_ENABLED = True  # Set to False to disable ChatGPT review
    
    # Timeout settings (safety first)
    CHATGPT_TIMEOUT = 8  # seconds per API call
    CHATGPT_MAX_RETRIES = 2  # retry attempts before giving up
    
    # Fallback behavior on ChatGPT failure
    # If True: timeout/error = approve signal (risky)
    # If False: timeout/error = reject signal (safe) ✅ RECOMMENDED
    CHATGPT_FALLBACK_APPROVE = False
    
    # Temperature for consistency (0.0-1.0)
    CHATGPT_TEMPERATURE = 0.2  # Low = more consistent decisions
    
    # Max tokens for response (keep low for JSON-only)
    CHATGPT_MAX_TOKENS = 100
    
    # ChatGPT Evaluation Criteria (informational only)
    CHATGPT_REJECTION_CRITERIA = [
        "Late entries (momentum exhausted)",
        "Low volume trend continuation",
        "Exhausted RSI + high ADX",
        "Poor risk/reward ratio (< 1.5)",
        "No pullback confirmation",
        "Chasing price action"
    ]
    
    # Minimum approval confidence (if ChatGPT provides it)
    CHATGPT_MIN_CONFIDENCE = 70  # 0-100 scale
    
    # ================================
    # LEGACY CHATGPT SETTINGS (kept for trap validation)
    # ================================
    CHATGPT_FALLBACK_ENABLED = True  # For trap detection fallback


    @classmethod
    def validate(cls):
        """Validate all required environment variables"""
        required = [
            'COINDCX_API_KEY',
            'COINDCX_SECRET',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]
        
        # ChatGPT API key required if final judge is enabled
        if cls.CHATGPT_FINAL_JUDGE_ENABLED:
            required.append('CHATGPT_API_KEY')
        
        missing = [v for v in required if not getattr(cls, v)]
        if missing:
            raise ValueError(f"❌ Missing env vars: {', '.join(missing)}")

        print("✅ Configuration validated")
        
        # Multi-mode info
        if cls.MULTI_MODE_ENABLED:
            print(f"🎯 Multi-Mode: {', '.join(cls.ACTIVE_MODES)}")
        else:
            print(f"📊 Mode: {cls.MODE}")
        
        # Signal limits
        print(f"📈 Max signals: {cls.MAX_SIGNALS_PER_DAY}/day")
        
        # Risk parameters
        print(f"⚡ Min ADX: {cls.MIN_ADX_STRENGTH}")
        print(f"🛡️ Liquidation buffer: {cls.LIQUIDATION_BUFFER*100}%")
        print(f"⏰ Cooldown: {cls.COOLDOWN_MINUTES} minutes")
        print(f"🎯 Min TP distance: {cls.MIN_TP_DISTANCE_PERCENT}%")
        
        # 🤖 ChatGPT Final Judge status
        print(f"\n{'='*50}")
        if cls.CHATGPT_FINAL_JUDGE_ENABLED:
            print(f"🤖 ChatGPT Final Judge: ENABLED")
            print(f"   Model: {cls.CHATGPT_MODEL}")
            print(f"   Timeout: {cls.CHATGPT_TIMEOUT}s")
            print(f"   Max Retries: {cls.CHATGPT_MAX_RETRIES}")
            print(f"   Fallback on error: {'APPROVE' if cls.CHATGPT_FALLBACK_APPROVE else 'REJECT ✅'}")
            print(f"\n   ⚠️  EVERY signal will be reviewed by ChatGPT")
            print(f"   ⚠️  Rejected signals will be SILENTLY dropped")
        else:
            print(f"⚠️  ChatGPT Final Judge: DISABLED")
            print(f"   Signals will be sent based on rules only")
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
        """
        Get appropriate decimal places based on price
        ✅ FIX #1: Prevents rounding errors for low-price coins
        """
        for rule_name, (threshold, decimals) in cls.PRICE_DECIMAL_RULES.items():
            if price < threshold:
                return decimals
        return 2  # Default


    @classmethod
    def get_chatgpt_stats_display(cls) -> str:
        """Get formatted ChatGPT configuration for display"""
        if not cls.CHATGPT_FINAL_JUDGE_ENABLED:
            return "ChatGPT Final Judge: DISABLED"
        
        return f"""
🤖 ChatGPT Final Judge Configuration:
   ├─ Model: {cls.CHATGPT_MODEL}
   ├─ Timeout: {cls.CHATGPT_TIMEOUT}s (with {cls.CHATGPT_MAX_RETRIES} retries)
   ├─ Temperature: {cls.CHATGPT_TEMPERATURE}
   ├─ Error Handling: {'Approve' if cls.CHATGPT_FALLBACK_APPROVE else 'Reject (Safe)'}
   └─ Status: ACTIVE (Every signal reviewed)
        """.strip()


config = Config()