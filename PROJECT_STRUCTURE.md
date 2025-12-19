# 📁 ARUN Bot - Project Structure

Complete file organization and module descriptions.

## 🗂️ File Structure

```
ARUN-Bot/
│
├── 🐍 Core Python Files
│   ├── main.py                    # Main bot controller & entry point
│   ├── config.py                  # Configuration & environment settings
│   ├── indicators.py              # Technical indicators (EMA, MACD, RSI, etc.)
│   ├── trap_detector.py           # 10 trap detection filters
│   ├── coindcx_api.py            # CoinDCX API integration
│   ├── websocket_feed.py         # Real-time WebSocket price feed
│   ├── signal_generator.py       # Signal generation engine
│   ├── telegram_notifier.py      # Telegram message sender
│   └── chatgpt_advisor.py        # ChatGPT AI integration
│
├── 📦 Deployment Files
│   ├── requirements.txt           # Python dependencies
│   ├── Procfile                  # Railway process definition
│   ├── railway.json              # Railway configuration
│   └── runtime.txt               # Python version specification
│
├── 📝 Configuration Files
│   ├── .env.example              # Environment variables template
│   ├── .gitignore               # Git ignore rules
│   └── .env                     # YOUR ACTUAL KEYS (not in GitHub!)
│
└── 📚 Documentation
    ├── README.md                 # Main documentation
    ├── DEPLOYMENT_GUIDE.md       # Railway deployment guide
    └── PROJECT_STRUCTURE.md      # This file
```

---

## 📄 File Descriptions

### 🐍 main.py
**Purpose**: Main bot controller and orchestration

**Functions**:
- `ArunBot.__init__()` - Initialize all components
- `startup_checks()` - Validate config & test connections
- `scan_markets()` - Scan all pairs for signals
- `place_order()` - Execute trades on CoinDCX
- `monitor_positions()` - Track open positions
- `run()` - Main async event loop

**Usage**:
```bash
python main.py
```

---

### ⚙️ config.py
**Purpose**: Central configuration management

**Contains**:
- API credentials (from environment)
- Trading mode settings (QUICK/MID/TREND)
- Pair list (6 CoinDCX Futures)
- Risk parameters (leverage, liquidation buffer)
- Trap detection thresholds
- Power hours timing

**Key Variables**:
```python
AUTO_TRADE = true/false
MODE = QUICK/MID/TREND
MAX_SIGNALS_PER_DAY = 10
PAIRS = ['F-BTC_INR', ...]
```

---

### 📊 indicators.py
**Purpose**: Technical indicator calculations

**Indicators**:
- `ema()` - Exponential Moving Average
- `sma()` - Simple Moving Average
- `macd()` - MACD (line, signal, histogram)
- `rsi()` - Relative Strength Index
- `adx()` - Average Directional Index
- `atr()` - Average True Range
- `bollinger_bands()` - Bollinger Bands
- `stochastic()` - Stochastic Oscillator
- `volume_surge()` - Volume anomaly detection
- `trend_regime()` - Market regime (trending/ranging)
- `mtf_trend()` - Multi-timeframe trend analysis

**Usage**:
```python
from indicators import Indicators

ema_fast = Indicators.ema(close, 9)
rsi = Indicators.rsi(close, 14)
```

---

### 🛡️ trap_detector.py
**Purpose**: Detect and filter false signals

**10 Trap Filters**:
1. `check_ny_pullback_trap()` - NY session reversals
2. `check_liquidity_grab()` - Stop loss hunting
3. `check_wick_manipulation()` - Consecutive wicks
4. `check_news_spike()` - Sudden price spikes
5. `check_market_maker_trap()` - False breakouts
6. `check_orderblock_rejection()` - Supply/demand rejections
7. `check_entry_timing_lag()` - Stale signals
8. `check_spread_expansion()` - Low liquidity
9. `check_three_candle_reversal()` - Reversal patterns
10. `check_indicator_overfit()` - Too-perfect readings

**Usage**:
```python
from trap_detector import TrapDetector

traps = TrapDetector.check_all_traps(candles)
if TrapDetector.is_trapped(traps):
    return None  # Skip signal
```

---

### 🔌 coindcx_api.py
**Purpose**: CoinDCX API integration

**Methods**:
- `get_candles()` - Historical OHLCV data
- `get_ticker()` - Current price (bid/ask/last)
- `place_order()` - Place futures order
- `get_account_balance()` - Account info
- `get_open_positions()` - Active positions
- `cancel_order()` - Cancel pending order
- `test_connection()` - API health check

**Authentication**:
Uses HMAC SHA256 signature for secure requests.

---

### 📡 websocket_feed.py
**Purpose**: Real-time price feed via WebSocket

**Features**:
- 1-second latency updates
- Auto-reconnect on disconnect
- Background thread operation
- Multi-pair subscription

**Usage**:
```python
from websocket_feed import ws_feed

ws_feed.start()
price = ws_feed.get_price('F-BTC_INR')
```

---

### 🎯 signal_generator.py
**Purpose**: Core signal generation logic

**Process**:
1. Daily counter reset
2. Cooldown check (30 min per pair)
3. Power hour validation
4. Indicator calculation
5. Trap detection
6. Trend identification
7. Entry/SL/TP calculation
8. Liquidation safety check
9. Score calculation (0-100)
10. Signal generation

**Signal Format**:
```python
{
    'pair': 'F-BTC_INR',
    'direction': 'LONG',
    'entry': 3500000.0,
    'sl': 3450000.0,
    'tp1': 3570000.0,
    'tp2': 3620000.0,
    'leverage': 12,
    'score': 85,
    'rsi': 52.3,
    'adx': 38.5,
    'mtf_trend': 'STRONG_UP',
    'mode': 'QUICK',
    'timestamp': '2025-12-20 14:30:45'
}
```

---

### 📱 telegram_notifier.py
**Purpose**: Send formatted messages to Telegram

**Methods**:
- `send_signal()` - Premium signal format
- `send_alert()` - General notifications
- `send_startup_message()` - Bot start notification
- `send_daily_summary()` - End-of-day stats
- `test_connection()` - Bot health check

**Message Format**:
Rich markdown with emojis, clear structure, all key info.

---

### 🤖 chatgpt_advisor.py
**Purpose**: AI-powered decision support (MINIMAL usage)

**Functions**:
- `validate_signal()` - Risk assessment
- `suggest_mode()` - Best mode for conditions
- `explain_trap()` - Trap explanation
- `rank_pairs()` - Priority ranking
- `check_parameter_safety()` - Trade safety check

**API Efficiency**:
- Uses gpt-4o-mini (cheapest)
- Max 300 tokens per call
- Only for critical decisions
- ~5-10 calls per day max

---

### 📦 requirements.txt
**Purpose**: Python package dependencies

**Packages**:
```
pandas==2.1.4          # Data manipulation
numpy==1.26.2          # Numerical computing
requests==2.31.0       # HTTP requests
websocket-client==1.7.0 # WebSocket connection
openai==1.6.1          # ChatGPT API
asyncio==3.4.3         # Async operations
python-dotenv==1.0.0   # Environment variables
```

---

### 🚂 Procfile
**Purpose**: Railway process definition

```
worker: python main.py
```

Tells Railway to run `main.py` as a worker process.

---

### ⚙️ railway.json
**Purpose**: Railway deployment configuration

**Settings**:
- Build: Nixpacks (auto-detect Python)
- Replicas: 1 (single instance)
- Restart: On failure (auto-restart)
- Max retries: 10

---

### 🐍 runtime.txt
**Purpose**: Python version specification

```
python-3.11.7
```

Ensures Railway uses correct Python version.

---

### 🔐 .env.example
**Purpose**: Template for environment variables

**Contains**:
- API key placeholders
- Configuration examples
- Comments/instructions

**Important**: 
- Copy to `.env` with real values
- Never commit `.env` to GitHub

---

### 🚫 .gitignore
**Purpose**: Prevent sensitive files from GitHub

**Ignores**:
- `.env` (API keys)
- `__pycache__/` (Python cache)
- `*.pyc` (compiled files)
- `venv/` (virtual environment)
- IDE files (`.vscode/`, `.idea/`)

---

### 📚 README.md
**Purpose**: Main project documentation

**Sections**:
- Features overview
- Installation guide
- Configuration instructions
- Usage examples
- Risk disclaimer
- Support info

---

### 🚀 DEPLOYMENT_GUIDE.md
**Purpose**: Step-by-step Railway deployment

**Covers**:
- GitHub setup
- Railway project creation
- Environment variable configuration
- Deployment process
- Testing & verification
- Troubleshooting

---

## 🔄 Data Flow

```
1. main.py (Start)
   ↓
2. config.py (Load settings)
   ↓
3. websocket_feed.py (Connect to prices)
   ↓
4. coindcx_api.py (Fetch candles)
   ↓
5. indicators.py (Calculate indicators)
   ↓
6. trap_detector.py (Check traps)
   ↓
7. signal_generator.py (Generate signal)
   ↓
8. chatgpt_advisor.py (Optional validation)
   ↓
9. telegram_notifier.py (Send notification)
   ↓
10. coindcx_api.py (Place order if AUTO_TRADE=true)
```

---

## 🔧 Customization Points

### Adjust Trading Logic
- **File**: `signal_generator.py`
- **Change**: Indicator thresholds, scoring weights

### Add New Indicators
- **File**: `indicators.py`
- **Add**: New calculation method

### Modify Trap Filters
- **File**: `trap_detector.py`
- **Adjust**: Threshold values, add new traps

### Change Notification Format
- **File**: `telegram_notifier.py`
- **Modify**: Message templates

### Add New Trading Pairs
- **File**: `config.py`
- **Update**: `PAIRS` list

---

## 📊 Module Dependencies

```
main.py
├── config.py (settings)
├── coindcx_api.py (market data)
├── websocket_feed.py (real-time prices)
├── signal_generator.py
│   ├── indicators.py (calculations)
│   ├── trap_detector.py (filters)
│   └── coindcx_api.py (data)
├── telegram_notifier.py (notifications)
└── chatgpt_advisor.py (AI)
```

---

## 🧪 Testing Strategy

### Local Testing
```bash
# Set AUTO_TRADE=false in .env
python main.py

# Monitor Telegram for signals
# Check logs for errors
```

### Railway Testing
```bash
# Deploy with AUTO_TRADE=false
# Monitor Railway logs
# Verify signals in Telegram
# Check for 24 hours
```

### Production
```bash
# Set AUTO_TRADE=true
# Start with small position sizes
# Monitor closely for first week
```

---

## 📈 Performance Monitoring

**Watch**:
- Signal quality (score distribution)
- Trap detection rate
- API response times
- WebSocket stability
- Memory/CPU usage (Railway dashboard)

**Optimize**:
- Adjust MODE based on conditions
- Fine-tune indicator parameters
- Add/remove pairs based on volume

---

## 🆘 Emergency Procedures

### Stop Trading Immediately
```
Railway Dashboard → Stop Application
```

### Close All Positions
```
Log into CoinDCX → Close positions manually
```

### Disable Bot
```
Set AUTO_TRADE=false in Railway variables
```

---

**File structure complete! 🎉**

All components work together to create a professional, production-ready trading bot.