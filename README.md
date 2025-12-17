# 🤖 CoinDCX Smart Trading Bot

Advanced cryptocurrency trading signal bot for CoinDCX with professional technical analysis and smart money logic.

## ✨ Features

- **Multi-Indicator Analysis**: RSI, MACD, EMA, ADX, Bollinger Bands
- **Candlestick Patterns**: 10+ patterns (Engulfing, Stars, Hammers, etc.)
- **Smart Money Logic**: 
  - Market regime detection
  - Order flow imbalance
  - Liquidity grab patterns
  - Volume profile POC
  - Fair value gaps (FVG)
  - Multi-timeframe momentum
  - Fibonacci confluence
- **Risk Management**: ATR-based dynamic SL/TP
- **Signal Control**: Cooldown periods & daily limits
- **Telegram Integration**: Real-time alerts with full signal details

## 🚀 Railway Deployment

### 1. Create Telegram Bot
1. Open [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow instructions
3. Save your bot token

### 2. Get Chat ID
1. Start chat with your bot
2. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Find `"chat":{"id":123456789}` - that's your Chat ID

### 3. Deploy to Railway
1. Push this code to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub"
4. Select your repository
5. Add environment variables (see below)
6. Deploy!

### 4. Environment Variables
