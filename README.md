# 🚀 Crypto Trading Bot v4.0

Advanced cryptocurrency trading signal generator with multi-timeframe analysis, orderflow detection, and smart money tracking.

## ✨ Features

### Signal Generation
- ✅ **47-Point Logic System** - Comprehensive technical analysis
- ✅ **Confidence Scoring** (0-100%) - Quality-rated signals
- ✅ **Multi-Timeframe Analysis** - 1m, 5m, 15m, 60m alignment
- ✅ **Smart Money Detection** - Track institutional orders
- ✅ **Volume Profile Analysis** - POC, Value Area
- ✅ **Support/Resistance** - Auto-detected levels
- ✅ **Fibonacci Retracements** - Key price levels
- ✅ **Pivot Points** - Daily trading levels

### Risk Management
- ✅ **Position Tracking** - Manual trade logging
- ✅ **P&L Monitoring** - Real-time performance
- ✅ **Stop Loss Alerts** - Automated notifications
- ✅ **Take Profit Alerts** - Target hit notifications
- ✅ **Trading Statistics** - Win rate, profit factor

### Technical Features
- ✅ **Real-time WebSocket** - Live market data
- ✅ **In-memory Buffering** - Fast data processing
- ✅ **Graceful Shutdown** - Safe restarts
- ✅ **Health Monitoring** - Railway/Render compatible
- ✅ **Telegram Alerts** - Rich formatted notifications

## 📦 Installation

### Prerequisites
- Python 3.11+
- Telegram Bot Token
- Railway/Render account (or run locally)

### Quick Start

1. **Clone Repository**
```bash
git clone <your-repo>
cd crypto-trading-bot
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Set Environment Variables**
```bash
export TELEGRAM_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export MIN_CONFIDENCE="60"
export SCAN_INTERVAL="25"
export COOLDOWN_MIN="30"
```

4. **Run Bot**
```bash
python main.py
```

## 🚀 Deployment

### Railway

1. Connect GitHub repository
2. Add environment variables:
   - `TELEGRAM_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `MIN_CONFIDENCE` (optional, default: 60)
   - `SCAN_INTERVAL` (optional, default: 25)
   - `COOLDOWN_MIN` (optional, default: 30)
3. Deploy!

Railway will automatically:
- Detect Python runtime
- Install dependencies
- Run health checks
- Auto-restart on failures

### Render.com

1. Create New Web Service
2. Connect repository
3. Configure:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `python main.py`
   - **Plan:** Free (512MB RAM)
4. Add environment variables
5. Deploy

### Fly.io

```bash
flyctl launch
flyctl secrets set TELEGRAM_TOKEN=xxx TELEGRAM_CHAT_ID=xxx
flyctl deploy
```

## 🎯 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_TOKEN` | Required | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Required | Your Telegram chat ID |
| `SCAN_INTERVAL` | 25 | Scan interval in seconds |
| `COOLDOWN_MIN` | 30 | Signal cooldown in minutes |
| `MIN_CONFIDENCE` | 60 | Minimum confidence % |
| `CHUNK_SIZE` | 15 | WebSocket chunk size |
| `PORT` | 8080 | Server port |

### Trading Pairs

Edit `helpers.py` to customize trading pairs:

```python
PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", 
    # Add more pairs...
]
```

## 📊 API Endpoints

### GET /
Main status endpoint
```json
{
  "status": "running",
  "version": "4.0",
  "ws_connected": true,
  "btc_trades": 150,
  "pairs": 49,
  "active_signals": 3
}
```

### GET /health
Health check for monitoring
```json
{
  "status": "healthy",
  "btc_trades": 150,
  "ws_connected": true
}
```

### GET /stats
Trading statistics
```json
{
  "trading_stats": {
    "total_trades": 50,
    "win_rate": 65.5,
    "total_pnl": 1250.50
  },
  "open_positions": {...}
}
```

### GET /signals
Recent signals (last 20)

### GET /debug
Debug information

## 📱 Telegram Commands

The bot sends rich formatted alerts with:
- 🎯 Signal type (LONG/SHORT)
- 💰 Entry price
- 📊 Confidence score
- 📈 Key support/resistance levels
- 🎯 Suggested stop loss/take profit
- 💡 Smart money indicators
- ⚠️ Risk level

### Example Signal

```
╔══════════════════════════════
║ 🟢 BTCUSDT — 🟢 LONG SIGNAL
╠══════════════════════════════

📊 SIGNAL METRICS
├ Strategy: QUICK
├ Confidence: 78.5% (GOOD)
├ Score: 12.5/17
├ Risk Level: LOW
└ Price: $45,123.45

📈 KEY LEVELS
├ Resistance: $45,500.00
├ Support: $44,800.00
└ R:R Ratio: 2.35

💡 SUGGESTED ENTRY
├ Entry Zone: $45,100 - $45,150
├ Stop Loss: $44,750.00
└ Target: $45,600.00

╚══════════════════════════════
⏰ 14:25:30 UTC
```

## 🛠️ Development

### Project Structure

```
crypto-trading-bot/
├── main.py                    # Main application
├── helpers.py                 # Data utilities
├── scorer.py                  # Signal scoring
├── signal_confidence.py       # Confidence calculation
├── price_levels.py           # S/R, pivots, fibonacci
├── volume_analysis.py        # Volume profile, smart money
├── telegram_formatter.py     # Message formatting
├── position_tracker.py       # Trade tracking
└── requirements.txt          # Dependencies
```

### Adding New Indicators

1. Add calculation in `helpers.py`
2. Add evaluation logic in `scorer.py`
3. Update confidence weights in `signal_confidence.py`

### Testing

```bash
# Test WebSocket connection
python -c "from helpers import PAIRS; print(f'Configured {len(PAIRS)} pairs')"

# Test signal generation
python -c "from scorer import compute_signal; import asyncio; asyncio.run(compute_signal('BTCUSDT', 'QUICK', {}, {}))"
```

## 📈 Performance Tuning

### For Low Memory (< 512MB)
- Reduce `CHUNK_SIZE` to 10
- Limit `PAIRS` to top 20
- Increase `SCAN_INTERVAL` to 30+

### For High Performance
- Increase `CHUNK_SIZE` to 20
- Add more `PAIRS`
- Reduce `SCAN_INTERVAL` to 15

## 🐛 Troubleshooting

### Bot Crashes on Railway
- Check credit balance
- Verify environment variables
- Review logs: `railway logs`

### No Signals Generated
- Increase `MIN_CONFIDENCE` threshold
- Check WebSocket connection
- Verify data is flowing: `/debug` endpoint

### Telegram Not Working
- Verify bot token
- Check chat ID format
- Test manually: `curl "https://api.telegram.org/bot<TOKEN>/getMe"`

## 📊 Performance Metrics

### Resource Usage (Typical)
- **Memory:** 200-400MB
- **CPU:** 0.1-0.3 vCPU
- **Network:** ~5-10 MB/hour

### Signal Quality
- **Confidence 70%+:** High quality (15-20% of signals)
- **Confidence 60-70%:** Good quality (30-40% of signals)
- **Confidence <60%:** Filtered out

## 🤝 Contributing

This is a personal trading bot. Fork it and customize for your needs!

## ⚠️ Disclaimer

**This bot is for educational purposes only.**

- Not financial advice
- Use at your own risk
- Test thoroughly before real trading
- Past performance ≠ future results

## 📝 License

MIT License - Free to use and modify

## 🙏 Credits

Built with:
- FastAPI
- WebSockets
- Binance API
- Telegram Bot API

---

**Made with ❤️ for crypto traders**

Last Updated: December 2025
Version: 4.0
