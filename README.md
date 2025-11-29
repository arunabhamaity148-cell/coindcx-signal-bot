# 75 % Win-Rate Scalper Bot  
**Binance → Telegram** in 2 min (mobile friendly)

---

## 🔧 What it does
- Fetches live 15 m klines  
- Runs 6 quality guards (news, funding, spread, market-awake)  
- Scores 0-100; ≥ 85 → signal  
- Partial exit: 50 % at TP-1, 50 % at TP-2  
- **Target win-rate ≈ 75 %**

---

## ⚙️ ENV variables (Railway → Variables tab)
| Key | Where to get |
|----|--------------|
| `BINANCE_API_KEY` | Binance → API Management → Create |
| `BINANCE_SECRET` | same page (shown once) |
| `TELEGRAM_BOT_TOKEN` | Telegram → @BotFather → /newbot |
| `TELEGRAM_CHAT_ID` | Telegram → @userinfobot → Start |

---

## 🎛️ Tune settings
Edit `config.py` → push → Railway auto-rebuild in 30 s  
- symbols, mode (quick/mid/trend), TP/SL %, guards on/off

---

## 📱 Mobile deploy (2 min)
1. Railway app → New → GitHub repo  
2. Add 4 env vars above  
3. Deploy → logs show “75 % bot started”  
4. Telegram alerts fire automatically

---

## ⏹️ Stop / restart
- **Stop**: Railway → Settings → Delete project  
- **Restart**: push new commit → Railway rebuilds instantly

---

**Happy green pips!**  
Push this README → **deploy → send logs screenshot** for final check.
