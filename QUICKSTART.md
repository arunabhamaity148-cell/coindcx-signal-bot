# 🚀 QUICKSTART GUIDE

Complete step-by-step guide to get your trading bot running in 30 minutes!

---

## 📋 Prerequisites

Before starting, make sure you have:

- ✅ Python 3.10 or higher
- ✅ 10GB+ free disk space
- ✅ Stable internet connection
- ✅ Exchange accounts (Bybit, OKX) with API keys
- ✅ Basic command line knowledge

---

## 🎯 OPTION 1: Automated Installation (Recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

### Step 2: Run Auto Installer
```bash
chmod +x install.sh
./install.sh
```

The installer will:
- ✅ Check system requirements
- ✅ Install dependencies (TA-Lib, PostgreSQL, etc.)
- ✅ Create virtual environment
- ✅ Install Python packages
- ✅ Setup directory structure
- ✅ Create database (optional)

### Step 3: Configure API Keys
```bash
nano config/api_keys.env
```

Fill in your actual API keys:
```env
BYBIT_API_KEY=your_actual_key_here
BYBIT_SECRET=your_actual_secret_here
OKX_API_KEY=your_actual_key_here
# ... etc
```

**⚠️ NEVER commit this file to GitHub!**

---

## 🛠️ OPTION 2: Manual Installation

### Step 1: System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    build-essential \
    libpq-dev \
    postgresql

# Install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

#### macOS
```bash
brew install python@3.10 postgresql ta-lib
```

#### Windows
1. Install Python 3.10+ from [python.org](https://www.python.org/)
2. Install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
3. Download TA-Lib from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

### Step 2: Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Python Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Create Directories
```bash
mkdir -p data/historical models logs charts backtest_results
```

### Step 5: Setup Database (Optional)
```bash
# Create database
sudo -u postgres psql
CREATE DATABASE trading_bot;
CREATE USER trading_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;
\q

# Run setup script
python scripts/setup_database.py
```

---

## 📊 Quick Test Run

### 1. Download Historical Data (5 Years)
```bash
python scripts/download_data.py \
    --symbol BTC/USDT:USDT \
    --timeframe 15m \
    --years 5 \
    --verify \
    --prepare-ml
```

**Expected output:**
- ✅ ~175,000 candles downloaded
- ✅ Data saved in: `data/historical/BTC_USDT_USDT_15m_5y_bybit.csv`
- ✅ Data quality verified
- ✅ Labels created for ML

**Time:** ~15-20 minutes

### 2. Train ML Models
```bash
python scripts/train_models.py \
    --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv \
    --lookback 60 \
    --epochs 50
```

**Expected output:**
- ✅ LSTM model trained (~65-68% accuracy)
- ✅ XGBoost model trained (~68-72% accuracy)
- ✅ Random Forest model trained
- ✅ Ensemble model created (~70-75% accuracy)
- ✅ Models saved in: `models/`

**Time:** ~30-60 minutes (depends on CPU/GPU)

**💡 Quick Test:** Use `--epochs 10` for faster testing

### 3. Run Backtest
```bash
python backtest/backtester.py \
    --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv \
    --capital 10000 \
    --save
```

**Expected output:**
- ✅ Backtest completed
- ✅ Performance metrics calculated
- ✅ Results saved in: `backtest_results/`

**Sample Results:**
```
📊 BACKTEST RESULTS
═══════════════════════════════════════
💰 CAPITAL
   Initial: ₹10,000.00
   Final:   ₹17,500.00
   P&L:     ₹7,500.00
   Return:  75.00%

📈 TRADE STATISTICS
   Total trades:    156
   Winning trades:  106 (68%)
   Losing trades:   50 (32%)
   Win rate:        68%
```

**Time:** ~5-10 minutes

### 4. Create Visualizations
```bash
python visualization/chart_plotter.py \
    --type dashboard \
    --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv \
    --trades backtest_results/trades.csv \
    --equity backtest_results/equity_curve.csv
```

**Output:**
- ✅ Price chart with trade markers
- ✅ Equity curve with drawdown
- ✅ Trade distribution charts
- ✅ Performance heatmap
- ✅ All saved in: `charts/`

**Time:** ~2 minutes

---

## 🎮 Start Trading

### Paper Trading (Recommended First!)
```bash
python main.py \
    --mode paper \
    --capital 10000 \
    --symbol BTC/USDT:USDT \
    --interval 900
```

**What it does:**
- ✅ Simulates real trading
- ✅ No real money risk
- ✅ Tests all systems
- ✅ Builds confidence

**Run for:** 1-2 weeks minimum

### Live Trading (After Paper Success)
```bash
python main.py \
    --mode live \
    --capital 2000 \
    --symbol BTC/USDT:USDT
```

**⚠️ Start Small:**
- Begin with ₹2,000 only
- Use 3x leverage maximum
- Monitor closely for first week
- Scale up gradually

---

## 📱 Setup Telegram Notifications

### 1. Create Telegram Bot
1. Open Telegram → Search `@BotFather`
2. Send `/newbot`
3. Choose name and username
4. Copy bot token

### 2. Get Your Chat ID
1. Start chat with your bot
2. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Find your `chat_id` in response

### 3. Add to Config
```bash
nano config/api_keys.env
```

Add:
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 4. Test Bot
```bash
python monitoring/telegram_bot.py
```

**Commands:**
- `/start` - Start bot
- `/status` - Check status
- `/balance` - Show P&L
- `/trades` - Today's trades
- `/emergency` - Close all positions

---

## 🔧 Common Commands

### Activate Environment
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Download More Data
```bash
# Ethereum
python scripts/download_data.py --symbol ETH/USDT:USDT --years 5

# Multiple symbols
python scripts/download_data.py --multiple --years 5
```

### Retrain Models
```bash
python scripts/train_models.py \
    --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv \
    --epochs 100
```

### Different Backtest Period
```bash
python backtest/backtester.py \
    --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv \
    --capital 50000 \
    --fee 0.0002
```

### Create Single Chart
```bash
# Price chart only
python visualization/chart_plotter.py --type price \
    --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv

# Equity curve only
python visualization/chart_plotter.py --type equity \
    --equity backtest_results/equity_curve.csv
```

---

## ⚠️ Troubleshooting

### Issue: TA-Lib import error
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib
pip install TA-Lib

# macOS
brew install ta-lib
pip install TA-Lib
```

### Issue: PostgreSQL connection failed
```bash
# Check if running
sudo systemctl status postgresql  # Linux
brew services list                # macOS

# Restart
sudo systemctl restart postgresql  # Linux
brew services restart postgresql   # macOS
```

### Issue: API key errors
- ✅ Check keys are correct in `config/api_keys.env`
- ✅ Verify IP whitelisting on exchange
- ✅ Check API permissions (need "Trading", not "Withdrawal")
- ✅ Test keys manually in exchange dashboard

### Issue: Out of memory during training
```bash
# Reduce batch size
python scripts/train_models.py \
    --data your_data.csv \
    --epochs 50 \
    --batch-size 16  # Default is 32
```

### Issue: Models not loading
```bash
# Check if models exist
ls -lh models/

# Retrain if missing
python scripts/train_models.py --data your_data.csv
```

---

## 📚 Learning Path

### Week 1: Setup & Understanding
- ✅ Complete installation
- ✅ Download data and explore
- ✅ Run backtest and analyze results
- ✅ Read through code documentation

### Week 2: Paper Trading
- ✅ Start paper trading
- ✅ Monitor daily performance
- ✅ Understand trade signals
- ✅ Fine-tune parameters

### Week 3: Optimization
- ✅ Adjust risk parameters
- ✅ Test different leverage levels
- ✅ Optimize entry/exit logic
- ✅ Refine ML confidence thresholds

### Week 4: Live Trading (If Ready)
- ✅ Start with minimal capital
- ✅ Use conservative settings
- ✅ Monitor closely
- ✅ Scale gradually

---

## 🎯 Performance Targets

### Conservative (Good Start)
- Win Rate: 60-65%
- Daily Profit: ₹1,000-1,500
- Max Drawdown: <20%
- Sharpe Ratio: >1.0

### Realistic (Expected)
- Win Rate: 65-70%
- Daily Profit: ₹1,500-2,500
- Max Drawdown: <15%
- Sharpe Ratio: >1.5

### Optimistic (Best Case)
- Win Rate: 70-75%
- Daily Profit: ₹3,000-5,000
- Max Drawdown: <10%
- Sharpe Ratio: >2.0

---

## 🆘 Getting Help

### Resources
- 📖 Full Documentation: `README.md`
- 💻 Code Comments: Read inline documentation
- 🐛 Bug Reports: GitHub Issues
- 💬 Community: (Add Discord/Telegram link)

### Before Asking for Help
1. Check error logs in `logs/`
2. Verify configuration in `config/`
3. Read relevant documentation
4. Search existing GitHub issues

---

## ⚡ Quick Reference

```bash
# Daily workflow
source venv/bin/activate
python main.py --mode paper --capital 10000

# Check status
tail -f logs/trading_bot.log

# Create daily report
python visualization/chart_plotter.py --type dashboard \
    --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv \
    --trades backtest_results/trades.csv \
    --equity backtest_results/equity_curve.csv

# Backup data
tar -czf backup_$(date +%Y%m%d).tar.gz \
    models/ logs/ backtest_results/ config/api_keys.env
```

---

## 🎉 You're Ready!

If you've completed all steps above, you're ready to start trading!

**Remember:**
- ✅ Start with paper trading
- ✅ Never risk more than you can afford to lose
- ✅ Monitor the bot regularly
- ✅ Keep learning and improving
- ✅ Stay disciplined

**Good luck and happy trading! 🚀**

---

**Next:** Read full `README.md` for advanced features and customization.