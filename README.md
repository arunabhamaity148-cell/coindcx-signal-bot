# 🎨 Unique Smart CoinDCX Signal Bot

**15-25 Quality Signals Daily | 10 Proprietary Indicators | Real CoinDCX Data**

## ✨ What Makes It UNIQUE

### 10 Proprietary Indicators (Nobody else has):

1. **Momentum Wave Detection** - Velocity + Acceleration
2. **Smart Money Index** - Volume + Price action signatures
3. **Market Synchrony** - Price + Volume alignment
4. **Adaptive Volatility** - Optimal trading ranges
5. **Trend Persistence** - Continuation strength
6. **Momentum Divergence** - Early reversal detection
7. **Volume Profile** - Distribution analysis
8. **Price Elasticity** - Market responsiveness
9. **Sentiment Momentum** - Pattern recognition
10. **Liquidity Flow** - Money flow tracking

## 🎯 Features

- ✅ **15-25 Signals Daily** (Not too few, not too many)
- ✅ **55% Min Score** (Balanced - not too strict)
- ✅ **Real CoinDCX Data** (Ticker-based)
- ✅ **Smart Analysis** (10 unique indicators)
- ✅ **Quality Maintained** (Multi-factor confirmation)
- ✅ **Both BUY & SELL** (All opportunities)

## 📊 Expected Performance

- **Signals/Day:** 15-25
- **Quality:** Balanced
- **Win Rate Target:** 65-70%
- **Avg R:R:** 1:1.75
- **Confidence:** HIGH/MEDIUM/GOOD

## 🚀 Quick Setup

### 1. Get API Keys

**CoinDCX:**
1. Visit [CoinDCX API](https://coindcx.com/api-trading)
2. Create API Key (Read Only)
3. Save Key + Secret

**Telegram:**
1. Message [@BotFather](https://t.me/BotFather)
2. Create new bot: `/newbot`
3. Save Bot Token
4. Get Chat ID from [@userinfobot](https://t.me/userinfobot)

### 2. Local Setup

```bash
# Clone repo
git clone <your-repo-url>
cd coindcx-signal-bot

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your credentials

# Run
python main.py
```

### 3. Railway Deploy

```bash
# Push to GitHub
git add .
git commit -m "Initial deploy"
git push

# Deploy on Railway
1. Go to railway.app
2. New Project → Deploy from GitHub
3. Select your repo
4. Add environment variables:
   - COINDCX_API_KEY
   - COINDCX_SECRET
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
5. Deploy!
```

## 📱 Signal Format

```
🎨 UNIQUE SIGNAL 🎨

📌 Pair: BTCINR
📈 Side: BUY

💰 Entry: ₹82,45,000
🛑 SL: ₹81,90,000
🎯 TP: ₹84,30,000

📐 R:R: 1:1.8
🧠 Score: 68%
🔥 Confidence: HIGH

🎨 Unique Insights:
  • ⚡ ACCELERATING_UP
  • 🐋 SMART_BUY
  • 🎯 BULLISH_SYNC

⚠️ CoinDCX Manual Trade

🕐 17-Dec 02:45 PM
```

## ⚙️ Configuration

Edit `main.py` → `Config` class:

```python
class Config:
    # More signals
    MIN_SCORE = 50  # Instead of 55
    
    # Faster scans
    SCAN_INTERVAL = 30  # Instead of 40
    
    # Add/remove coins
    COINS_TO_MONITOR = ['BTC', 'ETH', ...]
```

## 🧠 How It Works

1. **Collects live ticker data** from CoinDCX every 40s
2. **Tracks price history** (last 50 data points)
3. **Analyzes with 10 indicators** simultaneously
4. **Calculates combined score** (out of 100%)
5. **Generates signal** when score ≥ 55%
6. **Sends to Telegram** instantly
7. **Saves to database** for tracking

## 📊 Signal Quality

### Score Breakdown:
- **Momentum Wave:** 12 points
- **Smart Money:** 10 points
- **Market Sync:** 10 points
- **Volatility:** 10 points
- **Trend:** 8 points
- **Divergence:** 8 points
- **Volume Profile:** 8 points
- **Elasticity:** 7 points
- **Sentiment:** 8 points
- **Liquidity:** 7 points

**Total: 88 points → Converted to 100%**

**Minimum 55% needed = Smart but not too strict**

## 🎓 Trading Tips

### HIGH Confidence (70%+):
- Enter with 2% capital
- Trust the setup
- Multiple indicators aligned

### MEDIUM Confidence (60-69%):
- Enter with 1.5% capital
- Good setup
- Most indicators aligned

### GOOD Confidence (55-59%):
- Enter with 1% capital
- Decent setup
- Key indicators aligned

## 📈 Best Practices

1. **Use Limit Orders** - Better entry prices
2. **Set SL Immediately** - Risk management
3. **Take Partial Profits** - At 50% of TP
4. **Track Results** - Database has all data
5. **Don't Overtrade** - Follow the signals
6. **Check Chart** - Confirm before entry

## 🔧 Troubleshooting

### No Signals?
- Wait 2-3 minutes (building data)
- Check logs: `tail -f bot.log`
- Lower MIN_SCORE to 50

### Too Many Signals?
- Increase MIN_SCORE to 60-65
- Reduce COINS_TO_MONITOR

### Bot Crashes?
- Check Railway logs
- Verify API keys
- Check internet connection

## 📊 Database Queries

```bash
# View signals
sqlite3 signals.db
SELECT * FROM signals ORDER BY timestamp DESC LIMIT 10;

# Statistics
SELECT confidence, COUNT(*) 
FROM signals 
GROUP BY confidence;

# Best coins
SELECT market, AVG(logic_score) as avg_score
FROM signals
GROUP BY market
ORDER BY avg_score DESC;
```

## ⚠️ Important Notes

- **Manual Trading Only** - Bot sends signals, you trade
- **Risk Management** - Use 1-2% per trade
- **Not Financial Advice** - DYOR
- **CoinDCX Only** - Prices from CoinDCX
- **INR Markets** - Trades in INR

## 🎯 Why This Bot?

| Feature | This Bot | Others |
|---------|----------|--------|
| Indicators | 10 Unique | 2-3 Common |
| Data Source | CoinDCX Live | Generic |
| Signals/Day | 15-25 | Variable |
| Quality | Balanced | Too strict or too loose |
| Uniqueness | Proprietary | Generic RSI/MACD |

## 📞 Support

Issues? Questions?
1. Check logs first
2. Review documentation
3. Test with lower MIN_SCORE
4. Adjust settings as needed

## 📝 License

MIT License - Free to use

---

**Made for Indian Crypto Traders** 🇮🇳  
**Trade Smart, Not Hard!** 🎨💰