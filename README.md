# 🤖 ARUN - Advanced Crypto Trading Bot

Professional-grade automated trading bot for CoinDCX Futures with AI-powered decision making.

## ✨ Features

- **🚀 Real-time WebSocket Feed** - 1 second latency price updates
- **🎯 Multi-Mode Trading** - QUICK (5m), MID (15m), TREND (1h)
- **🛡️ 10 Trap Detection Systems** - Avoid false signals
- **📰 News Guard Protection** - Block trades during major economic events
- **📊 Advanced Technical Analysis** - EMA, MACD, RSI, ADX, ATR, MTF
- **🤖 AI Integration** - ChatGPT for decision validation
- **📈 Smart Position Sizing** - Liquidation-safe entries
- **📱 Telegram Notifications** - Premium formatted signals
- **⚡ Auto/Manual Mode** - Switch anytime via .env

## 🎯 Trading Performance

- **Max Signals**: 10 per day (Monday-Saturday)
- **Leverage**: 10x-15x (mode-dependent)
- **Accuracy Focus**: Quality over quantity
- **Risk Management**: SL always 10%+ away from liquidation

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/arun-bot.git
cd arun-bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run Bot
```bash
python main.py
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# CoinDCX API (required)
COINDCX_API_KEY=your_key
COINDCX_SECRET=your_secret

# ChatGPT (required)
CHATGPT_API_KEY=sk-...
CHATGPT_MODEL=gpt-4o-mini

# Telegram (required)
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789

# Bot Settings
AUTO_TRADE=false  # true = auto place orders
MODE=QUICK        # QUICK/MID/TREND
```

### Trading Modes

| Mode | Timeframe | Leverage | Use Case |
|------|-----------|----------|----------|
| **QUICK** | 5m | 12x | Fast scalping, high frequency |
| **MID** | 15m | 10x | Balanced, safer entries |
| **TREND** | 1h | 15x | Strong trends, high accuracy |

## 📊 How It Works

1. **Market Scanning** - Monitors 6 CoinDCX Futures pairs
2. **Indicator Analysis** - Calculates 7+ technical indicators
3. **Trap Detection** - Checks 10 trap filters
4. **Signal Generation** - Creates entry/SL/TP levels
5. **AI Validation** (optional) - ChatGPT risk assessment
6. **Order Execution** - Auto-place or manual notification
7. **Position Monitoring** - Track open positions

## 🛡️ Risk Management

### 10 Trap Filters
1. NY Pullback Trap
2. Liquidity Grab Filter
3. Wick Manipulation
4. News Spike Avoid
5. Market Maker Trap
6. Order Block Rejection
7. Entry Timing Lag
8. Spread Expansion
9. 3-Candle Reversal
10. Indicator Over-optimization

### News Guard Protection
**Blocks signals during major news** (±30 minutes):
- US CPI (Inflation Data)
- FOMC Meeting / Rate Decision
- NFP (Non-Farm Payroll)
- Fed Chair Major Speeches
- Bitcoin/Ethereum Major Upgrades

**Why?** Prevents 2-5% sudden spikes that can hit stop losses despite perfect technical analysis.

See [NEWS_GUARD_GUIDE.md](NEWS_GUARD_GUIDE.md) for details.

### Safety Features
- **Liquidation Buffer**: 10% minimum distance
- **Max Leverage**: Hard cap at 15x
- **Cooldown Period**: 30min between same-pair signals
- **Daily Limit**: Max 10 signals per day
- **Power Hours**: Trade only during optimal times

## 🚀 Railway Deployment

### 1. Create Railway Project
```bash
railway login
railway init
```

### 2. Add Environment Variables
Go to Railway dashboard → Variables → Add all from .env

### 3. Deploy
```bash
railway up
```

### 4. Monitor Logs
```bash
railway logs
```

## 📱 Telegram Setup

### Create Bot
1. Message @BotFather on Telegram
2. Send `/newbot`
3. Copy bot token

### Get Chat ID
1. Message your bot
2. Visit: `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Copy chat ID from response

## 🔍 Supported Pairs

- F-BTC_INR (Bitcoin)
- F-ETH_INR (Ethereum)
- F-SOL_INR (Solana)
- F-MATIC_INR (Polygon)
- F-ADA_INR (Cardano)
- F-DOGE_INR (Dogecoin)

## 📈 Signal Format

```
🔥 LONG SIGNAL 🟢

━━━━━━━━━━━━━━━━━━━━
📊 PAIR: F-BTC_INR
💯 SCORE: 85/100
📈 LEVERAGE: 12x

━━━━━━━━━━━━━━━━━━━━
🎯 ENTRY: ₹35,00,000
🛑 STOP LOSS: ₹34,50,000
✅ TP1: ₹35,70,000
✅ TP2: ₹36,20,000

━━━━━━━━━━━━━━━━━━━━
📉 RSI: 52.3
📊 ADX: 38.5 (Strong)
🔄 MTF: STRONG_UP
📦 Volume: 1.8x avg

━━━━━━━━━━━━━━━━━━━━
⚙️ MODE: QUICK
⏰ TIME: 2025-12-20 14:30:45
```

## ⚠️ Disclaimer

**IMPORTANT**: Trading cryptocurrencies involves substantial risk. This bot is for educational purposes. Always:
- Start with small capital
- Test in DRY RUN mode first (AUTO_TRADE=false)
- Never invest more than you can afford to lose
- Understand futures trading risks
- Monitor bot performance regularly

## 🤝 Contributing

Pull requests welcome! Please ensure:
- Code follows existing style
- All tests pass
- Documentation updated

## 📄 License

MIT License - Use at your own risk

## 🆘 Support

Issues? Open a GitHub issue or contact via Telegram.

---

**Made with ❤️ for Indian crypto traders**