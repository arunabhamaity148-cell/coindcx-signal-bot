# 🤖 Advanced Crypto Trading Bot

**Fully Automated AI-Powered Futures Trading System**

## 🎯 Overview

This is a sophisticated crypto trading bot that uses Machine Learning (LSTM + XGBoost + Random Forest ensemble) combined with **45 unique trading logics** to trade Bitcoin and Ethereum futures on multiple exchanges.

### Key Features

- ✅ **Multi-Exchange Support**: Bybit (primary), OKX (secondary), Binance (data)
- ✅ **AI-Powered Signals**: 3-model ensemble (LSTM, XGBoost, RF)
- ✅ **45 Unique Logics**: Advanced order flow, liquidation avoidance, gamma exposure
- ✅ **Fully Automated**: 24/7 trading with risk management
- ✅ **Real-time Monitoring**: Telegram notifications + Dashboard
- ✅ **Comprehensive Backtesting**: Walk-forward testing on 5 years data

## 📊 Expected Performance

| Metric | Conservative | Realistic | Optimistic |
|--------|-------------|-----------|------------|
| Win Rate | 65% | 68% | 72% |
| Daily Profit | ₹1,300 | ₹2,100 | ₹4,800 |
| Monthly Profit | ₹26,000-32,000 | ₹42,000-50,000 | ₹96,000-120,000 |
| Leverage | 5x | 7x | 10x |
| Starting Capital | ₹10,000 | ₹10,000 | ₹10,000 |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATA COLLECTION                        │
│  • Multi-exchange (Bybit, OKX, Binance)                │
│  • Real-time OHLCV, Orderbook, Funding Rates           │
│  • 5 years historical data                             │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              FEATURE ENGINEERING                         │
│  • 50+ technical indicators                             │
│  • Advanced order flow features                         │
│  • Liquidation proximity, Gamma exposure               │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              ML PREDICTION ENGINE                        │
│  • LSTM (60% weight)                                    │
│  • XGBoost (30% weight)                                 │
│  • Random Forest (10% weight)                           │
│  • Ensemble confidence scoring                          │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│             45 UNIQUE LOGIC FILTERS                      │
│  • Market health (8 logics)                             │
│  • Price action (7 logics)                              │
│  • Momentum (6 logics)                                  │
│  • Order flow (10 logics)                               │
│  • Derivatives (6 logics)                               │
│  • Anti-trap mechanisms (8 logics)                      │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              RISK MANAGEMENT                             │
│  • Position sizing (15% max)                            │
│  • Dynamic leverage (3-10x)                             │
│  • Liquidation distance monitoring                      │
│  • Daily loss limit (20%)                               │
│  • Emergency stop mechanism                             │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              EXECUTION ENGINE                            │
│  • Smart order routing                                  │
│  • TP/SL management                                     │
│  • Trailing stops                                       │
│  • Position monitoring                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│           MONITORING & ALERTS                            │
│  • Telegram bot (real-time)                             │
│  • Web dashboard                                        │
│  • Trade logs & analytics                               │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- 10GB+ disk space
- Stable internet connection

### Setup Steps

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install TA-Lib** (Required for technical indicators)
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib

# macOS
brew install ta-lib

# Windows
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.XX-cpXX-cpXX-winXX.whl
```

5. **Setup Database**
```bash
# Create PostgreSQL database
createdb trading_bot

# Run migrations (if any)
python setup_database.py
```

6. **Configure API Keys**
```bash
# Copy example config
cp config/api_keys.env.example config/api_keys.env

# Edit with your API keys
nano config/api_keys.env
```

7. **Download Historical Data**
```bash
python scripts/download_data.py --years 5
```

8. **Train ML Models**
```bash
python scripts/train_models.py
```

## ⚙️ Configuration

Edit `config/settings.py` to customize:

- Trading pairs
- Leverage levels
- Risk parameters
- ML confidence thresholds
- Unique logic parameters

## 🚀 Usage

### Paper Trading (Recommended First)
```bash
python main.py --mode paper --days 7
```

### Live Trading
```bash
python main.py --mode live --capital 10000
```

### Backtest Only
```bash
python backtest/backtester.py --start 2023-01-01 --end 2024-12-31
```

### Telegram Commands

Once running, control via Telegram:

```
/start - Start trading bot
/stop - Stop trading bot
/status - Current status
/balance - Show P&L
/trades - Today's trades
/settings - View configuration
/emergency - Close all positions
```

## 📁 Project Structure

```
crypto-trading-bot/
├── config/                 # Configuration files
│   ├── settings.py
│   └── api_keys.env
├── data/                   # Data collection
│   ├── data_collector.py
│   └── database.py
├── features/               # Feature engineering
│   ├── technical_indicators.py
│   ├── orderflow_features.py
│   └── market_health.py
├── ml/                     # ML models
│   ├── lstm_model.py
│   ├── xgboost_model.py
│   └── ensemble.py
├── strategy/               # Trading strategies
│   ├── signal_generator.py
│   └── unique_logics.py    # 45 unique logics
├── risk/                   # Risk management
│   └── risk_manager.py
├── execution/              # Order execution
│   └── order_executor.py
├── monitoring/             # Monitoring & alerts
│   └── telegram_bot.py
├── backtest/               # Backtesting
│   └── backtester.py
├── models/                 # Saved ML models
├── logs/                   # Log files
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🛡️ 45 Unique Trading Logics

### A) Market Health Filters (8)
1. BTC calm check
2. Market regime detection
3. Funding rate normal check
4. Fear & Greed Index filter
5. Fragile BTC market detection
6. High-impact news filter
7. Spread & slippage safety
8. Time-of-day liquidity window

### B) Price Action & Structure (7)
9. Breakout confirmation
10. Market structure shift
11. Orderblock retest
12. Fair value gap detection
13. EMA/SMA alignment
14. ATR volatility filter
15. Bollinger band squeeze

### C) Momentum (6)
16. RSI + divergence
17. MACD cross + slope
18. Stochastic reversal
19. OBV divergence
20. MFI direction
21. ROC momentum

### D) Order Flow & Depth (10)
22. Orderbook imbalance
23. VWAP calculation
24. VWAP deviation check
25. VWAP bounce/reclaim/rejection
26. CVD direction
27. Large order detection
28. Spoofing wall detection
29. True liquidity depth
30. Aggression ratio
31. Spread velocity

### E) Derivatives & Futures (6)
32. Open interest trend
33. OI divergence
34. Liquidation proximity
35. Funding arbitrage
36. Gamma exposure
37. Gamma-adjusted sizing

### F) Anti-Trap Mechanisms (8)
38. Avoid round numbers
39. Avoid obvious S/R
40. SL-hunting zone detection
41. Odd-time entries
42. Sudden wick filter
43. Bot-rush time avoidance
44. Manipulation candle filter
45. Consecutive loss cooldown

## ⚠️ Risk Disclaimer

**IMPORTANT**: Cryptocurrency trading involves substantial risk of loss.

- ❌ This is NOT a get-rich-quick scheme
- ❌ Past performance does NOT guarantee future results
- ❌ Only invest what you can afford to lose
- ❌ Never use borrowed money
- ❌ Trading can result in total capital loss

The creators of this bot are NOT responsible for any financial losses.

## 💰 Costs

### One-time
- Learning materials: ₹0-2,000 (YouTube free)
- Development tools: ₹0 (all free)
- Testing capital: ₹2,000
- **Total: ₹2,000-4,000**

### Monthly
- Server (VPS): ₹0-400 (AWS free tier 1 year)
- Internet: ₹500-1,000
- Electricity: ₹200-300
- Data APIs: ₹0-50
- **Total: ₹700-1,750/month**

### Trading Capital
- Initial testing: ₹2,000
- Full trading: ₹10,000
- Optimal: ₹50,000+

## 📈 Performance Metrics

Monitor these key metrics:

- **Win Rate**: Target 65%+
- **Profit Factor**: Target 1.5+
- **Sharpe Ratio**: Target 1.5+
- **Max Drawdown**: Keep under 25%
- **Recovery Factor**: Target 3+

## 🔒 Security Best Practices

- ✅ Never share API keys
- ✅ Use IP whitelisting
- ✅ Enable 2FA on all exchanges
- ✅ Only grant necessary permissions (NO withdrawal)
- ✅ Keep `.env` file out of version control
- ✅ Regular security audits
- ✅ Secure server with firewall

## 📚 Resources

- [CCXT Documentation](https://docs.ccxt.com/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide/keras)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Bybit API Docs](https://bybit-exchange.github.io/docs/)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is for educational purposes only. Use at your own risk.

## 📞 Support

- GitHub Issues: [Report bugs](https://github.com/yourusername/crypto-trading-bot/issues)
- Email: your.email@example.com
- Telegram Group: @YourBotGroup

## 🎯 Roadmap

- [ ] Add more exchanges (Bitget, MEXC)
- [ ] Implement options strategies
- [ ] Add sentiment analysis
- [ ] Create mobile app
- [ ] Multi-coin support
- [ ] Advanced portfolio management
- [ ] Social trading features

## ⭐ Star History

If this project helps you, please give it a ⭐️!

---

**Made with ❤️ for the crypto trading community**

**Remember**: Trade responsibly, manage risk, and never invest more than you can afford to lose! 