import os
from dataclasses import dataclass

@dataclass
class Config:
    """Bot Configuration - All settings from Railway environment"""
    
    # API Keys (Railway environment variables)
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET')
    CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY')
    CHATGPT_MODEL = os.getenv('CHATGPT_MODEL', 'gpt-4o-mini')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Bot Settings
    AUTO_TRADE = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
    MODE = os.getenv('MODE', 'QUICK')  # QUICK/MID/TREND
    MAX_SIGNALS_PER_DAY = 10
    MAX_LEVERAGE = 15
    
    # Trading Pairs - CoinDCX format
    # Use exact market names from CoinDCX
   
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
    
    # Mode Configurations
    MODE_CONFIG = {
        'QUICK': {
            'timeframe': '5m',
            'ema_fast': 9,
            'ema_slow': 21,
            'leverage': 12,
            'atr_sl_multiplier': 1.5,
            'atr_tp1_multiplier': 2.0,
            'atr_tp2_multiplier': 3.0
        },
        'MID': {
            'timeframe': '15m',
            'ema_fast': 12,
            'ema_slow': 26,
            'leverage': 10,
            'atr_sl_multiplier': 2.0,
            'atr_tp1_multiplier': 2.5,
            'atr_tp2_multiplier': 4.0
        },
        'TREND': {
            'timeframe': '1h',
            'ema_fast': 20,
            'ema_slow': 50,
            'leverage': 15,
            'atr_sl_multiplier': 2.5,
            'atr_tp1_multiplier': 3.0,
            'atr_tp2_multiplier': 5.0
        }
    }
    
    # Risk Management
    LIQUIDATION_BUFFER = 0.005   # 0.5% minimum distance from liquidation
    MIN_ADX_STRENGTH = 25
    COOLDOWN_MINUTES = 30  # Minimum time between signals for same pair
    
    # Trap Detection Thresholds
    LIQUIDITY_WICK_RATIO = 2.0  # Wick must be 2x body
    NEWS_SPIKE_THRESHOLD = 0.03  # 3% sudden move
    MAX_SPREAD_PERCENT = 0.5  # 0.5% spread warning
    
    # Power Hours (IST) - Best trading times
    POWER_HOURS = [
        (10, 13),  # 10 AM - 1 PM
        (15, 18)   # 3 PM - 6 PM
    ]
    
    # WebSocket
    WS_URL = "wss://stream.coindcx.com"
    WS_RECONNECT_DELAY = 5  # seconds
    
    # CoinDCX REST API
    COINDCX_BASE_URL = "https://api.coindcx.com"
    
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
        return True

config = Config()