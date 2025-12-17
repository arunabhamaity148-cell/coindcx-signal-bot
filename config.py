import os

class Config:
    # Telegram Settings
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

    # CoinDCX Settings
    COINDCX_BASE_URL = "https://public.coindcx.com"
    MARKETS = os.environ.get(
        'COINDCX_MARKETS',
        'B-BTC_USDT,B-ETH_USDT,B-SOL_USDT,B-XRP_USDT,B-MATIC_USDT,B-ADA_USDT'
    ).split(',')

    # 🔥 UPDATED TRADING PARAMETERS
    CANDLE_INTERVAL = '5m'
    CHECK_INTERVAL_MINUTES = 5

    MIN_SIGNAL_SCORE = 70
    COOLDOWN_MINUTES = 30
    MAX_SIGNALS_PER_DAY = 12

    # Risk Management
    ATR_SL_MULTIPLIER = 1.5
    ATR_TP1_MULTIPLIER = 2.0
    ATR_TP2_MULTIPLIER = 3.5
    MIN_RR_RATIO = 1.5

    # Technical Settings
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    EMA_SHORT = 9
    EMA_MEDIUM = 21
    EMA_LONG = 50
    ADX_PERIOD = 14
    ADX_STRONG_TREND = 25
    ATR_PERIOD = 14

    # ❌ Fake MTF removed (not used)
    # MTF_TIMEFRAMES = ['5m', '15m', '1h']

    # Volume Analysis
    VOLUME_SPIKE_THRESHOLD = 1.5
    POC_LOOKBACK = 50

    # Smart Money Concepts
    FVG_MIN_SIZE = 0.002
    LIQUIDITY_GRAB_LOOKBACK = 20

    # Signal Scoring Weights
    SCORE_WEIGHTS = {
        'trend_alignment': 20,
        'rsi_confirmation': 15,
        'macd_momentum': 15,
        'candlestick_pattern': 10,
        'volume_confirmation': 10,
        'mtf_alignment': 10,
        'smart_money': 10,
        'support_resistance': 10
    }