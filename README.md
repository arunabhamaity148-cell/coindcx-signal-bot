# 🚀 CoinDCX Advanced Trading Bot

50+ coins monitor kore 45 ta trading logic use kore automatic signal generate kore Telegram e send kore. Manual trading er jonno best solution.

## ✨ Features

- ✅ **50+ Coins Monitoring** - BTC, ETH, SOL, ADA সহ সব major coins
- ✅ **45 Trading Logics** - Market Health, Price Action, Momentum, Order Flow, Anti-Trap
- ✅ **3 Trading Modes** - QUICK (5m), MID (15m), TREND (1h)
- ✅ **Real CoinDCX Data** - INR futures price থেকে direct signal
- ✅ **Telegram Alerts** - Instant signal notification with entry, SL, TP
- ✅ **Both BUY & SELL** - Long এবং Short উভয় signal
- ✅ **Risk Management** - R:R ratio, Confidence score সহ
- ✅ **24/7 Running** - Railway এ host করে সব সময় চালু থাকবে

## 📁 Project Structure

```
trading-bot/
├── main.py              # Main entry point
├── helpers.py           # CoinDCX API, Telegram, Database
├── logic.py             # 45 Trading logics
├── requirements.txt     # Dependencies
├── README.md           # This file
├── .env                # Configuration (create this)
└── trading_bot.db      # SQLite database (auto-created)
```

## 🔧 Setup Instructions

### Step 1: CoinDCX API Keys নিন

1. [CoinDCX](https://coindcx.com) এ login করুন
2. Settings → API Management এ যান
3. Create New API Key → **Read Only** permission দিন
4. API Key এবং Secret save করুন

### Step 2: Telegram Bot তৈরি করুন

1. Telegram এ [@BotFather](https://t.me/BotFather) খুলুন
2. `/newbot` command দিন
3. Bot এর নাম দিন (e.g., "My Trading Bot")
4. Username দিন (e.g., "my_trading_signals_bot")
5. Bot Token save করুন
6. আপনার Chat ID পাওয়ার জন্য:
   - [@userinfobot](https://t.me/userinfobot) খুলুন
   - `/start` করুন
   - আপনার Chat ID copy করুন

### Step 3: Project Setup

```bash
# Clone or create project directory
mkdir trading-bot
cd trading-bot

# Create files
touch main.py helpers.py logic.py requirements.txt .env

# Copy code from artifacts to respective files
```

### Step 4: Configuration

`.env` file create করুন এবং এইগুলো add করুন:

```env
# CoinDCX API Credentials
COINDCX_API_KEY=your_coindcx_api_key_here
COINDCX_SECRET=your_coindcx_secret_here

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Test Locally

```bash
python main.py
```

আপনার Telegram এ signal আসা শুরু হবে! 🎉

## 🌐 Railway এ Deploy করা

### Step 1: GitHub Repository তৈরি করুন

```bash
# Initialize git
git init

# Create .gitignore
echo ".env
*.db
__pycache__/
*.pyc
*.log" > .gitignore

# Commit
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin https://github.com/yourusername/trading-bot.git
git push -u origin main
```

### Step 2: Railway Setup

1. [Railway.app](https://railway.app) এ যান
2. "Start a New Project" → "Deploy from GitHub repo"
3. আপনার repository select করুন
4. Environment Variables add করুন:
   - `COINDCX_API_KEY`
   - `COINDCX_SECRET`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
5. Deploy করুন

### Step 3: Monitor

Railway dashboard এ logs দেখতে পারবেন। Bot 24/7 চলবে!

## 📱 Mobile থেকে Use করা

1. **Telegram App** - সব signals আসবে
2. **CoinDCX App** - Manual trade execute করুন
3. **Railway App** - Bot monitor করুন

## 🎯 Signal Format

```
🚨 MID MODE SIGNAL 🚨

📌 Pair: B-SOL_USDT
📊 TF: 15m
📈 Side: BUY

💰 Entry: ₹8,450.00
🛑 SL: ₹8,350.00
🎯 TP: ₹8,650.00

📐 R:R: 1:2.0
🧠 Logic Score: 78%
🔥 Confidence: HIGH

⏱️ Mode: MID
⚠️ Trade manually on CoinDCX

🕐 13-Dec 02:45 PM
```

## 🧠 45 Trading Logics

### A) Market Health Filters (8 logics)
1. BTC Calm Check
2. Market Regime Detection
3. Funding Rate Extreme Filter
4. Fear & Greed Index Filter
5. Fragile Market Detection
6. High Impact News Time Filter
7. Spread & Slippage Safety
8. Low Liquidity Time Window

### B) Price Action & Structure (7 logics)
9. Breakout Confirmation
10. Market Structure Shift
11. Orderblock Retest
12. Fair Value Gap (FVG)
13. EMA Alignment (20/50/200)
14. ATR Volatility Filter
15. Bollinger Band Squeeze

### C) Momentum & Oscillators (6 logics)
16. RSI + Divergence
17. MACD Cross & Momentum
18. Stochastic Reversal
19. OBV Divergence
20. MFI Direction
21. ROC Momentum

### D) Order Flow & Depth (10 logics)
22. Orderbook Imbalance
23. VWAP Calculation
24. VWAP Deviation
25. VWAP Bounce/Rejection
26. CVD
27. Whale Detection
28. Spoofing Detection
29. Liquidity Depth
30. Aggression Ratio
31. Spread Velocity

### E) Derivatives & Futures (6 logics)
32. Open Interest Trend
33. OI Divergence
34. Liquidation Clusters
35. Funding Arbitrage
36. Gamma Exposure
37. Gamma Position Sizing

### F) Anti-Trap & Protection (8 logics)
38. Round Number Trap Avoidance
39. Obvious S/R Avoidance
40. Stop Hunt Zone Detection
41. Odd Time Entry Filter
42. Manipulation Candle Detection
43. Bot Rush Time Avoidance
44. Body-Wick Ratio Check
45. Consecutive Loss Cooldown

## ⚙️ Configuration Options

`main.py` এ `Config` class modify করে customize করতে পারবেন:

```python
class Config:
    COINS_TO_MONITOR = [...]  # যে coins monitor করবেন
    TIMEFRAMES = ['5m', '15m', '1h']  # Timeframes
    MIN_LOGIC_SCORE = 65  # Minimum score (65-100)
    SCAN_INTERVAL = 60  # Scan interval in seconds
```

## 📊 Database

Bot automatically SQLite database maintain করে:

- **signals table** - সব generated signals
- **trades table** - Manual trade tracking (optional)

Database দেখার জন্য:

```bash
sqlite3 trading_bot.db
.tables
SELECT * FROM signals ORDER BY timestamp DESC LIMIT 10;
```

## 🔍 Troubleshooting

### Problem: Signals আসছে না

**Solution:**
- Check CoinDCX API keys valid কিনা
- Check Telegram bot token এবং chat ID correct কিনা
- Check logs: `tail -f bot.log`
- Check MIN_LOGIC_SCORE কম করুন (e.g., 60)

### Problem: Railway এ deploy হচ্ছে না

**Solution:**
- Check সব environment variables set করা আছে কিনা
- Check requirements.txt এ সব dependencies আছে কিনা
- Railway logs check করুন

### Problem: Bot crash হয়ে যাচ্ছে

**Solution:**
- Check error logs
- Check CoinDCX API rate limits exceed হচ্ছে কিনা
- SCAN_INTERVAL বাড়ান (e.g., 120 seconds)

## 📈 Performance Tips

1. **Backtesting** - Signal history database এ save থাকে, analyze করুন
2. **Score Optimization** - যে coins বেশি accurate signal দেয় তাদের track করুন
3. **Timeframe Selection** - আপনার trading style অনুযায়ী timeframe adjust করুন
4. **Risk Management** - প্রতি trade এ capital এর 1-2% risk করুন

## 🎓 Trading Guidelines

### QUICK Mode (5m)
- ⚡ Fast scalping trades
- 🎯 Target: 0.5-1% profit
- ⏱️ Duration: 5-30 minutes
- 📊 Best for: Volatile markets

### MID Mode (15m)
- 📈 Swing trades
- 🎯 Target: 1-2% profit
- ⏱️ Duration: 1-4 hours
- 📊 Best for: Trending markets

### TREND Mode (1h+)
- 📊 Position trades
- 🎯 Target: 2-5% profit
- ⏱️ Duration: 4-24 hours
- 📊 Best for: Strong trends

## ⚠️ Risk Disclaimer

- এই bot শুধুমাত্র signal generate করে, automatic trading করে না
- সব trades manually execute করতে হবে
- Trading এ risk আছে, শুধু spare money use করুন
- Past performance future results guarantee করে না
- DYOR (Do Your Own Research)

## 🤝 Support

Issues বা questions থাকলে:
1. GitHub Issues open করুন
2. Logs সহ error details provide করুন
3. Configuration settings share করুন

## 📝 License

MIT License - Free to use and modify

---

**Made with ❤️ for Indian Crypto Traders**

Happy Trading! 🚀💰
