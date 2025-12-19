# 🚀 Railway Deployment Guide

Complete step-by-step guide to deploy ARUN Bot on Railway.

## 📋 Prerequisites

1. **GitHub Account** - To store code
2. **Railway Account** - Free tier available at [railway.app](https://railway.app)
3. **API Keys Ready**:
   - CoinDCX API Key + Secret
   - OpenAI API Key (ChatGPT)
   - Telegram Bot Token + Chat ID

## 🔧 Step 1: Prepare GitHub Repository

### 1.1 Create New Repository
```bash
# On GitHub, create new repository (ARUN-Bot)
# Make it PRIVATE (API keys will be in Railway only)
```

### 1.2 Upload Files
Copy all these files to your repository:
```
ARUN-Bot/
├── main.py
├── config.py
├── indicators.py
├── trap_detector.py
├── coindcx_api.py
├── websocket_feed.py
├── signal_generator.py
├── telegram_notifier.py
├── chatgpt_advisor.py
├── requirements.txt
├── Procfile
├── railway.json
├── runtime.txt
├── .gitignore
├── .env.example
└── README.md
```

### 1.3 Important: .gitignore Check
**NEVER** commit these files:
- `.env` (contains API keys)
- `__pycache__/`
- `*.pyc`

## 🚂 Step 2: Deploy to Railway

### 2.1 Create Railway Project

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose your `ARUN-Bot` repository
6. Railway will auto-detect Python and start deployment

### 2.2 Add Environment Variables

In Railway Dashboard:

1. Click on your project
2. Go to **"Variables"** tab
3. Click **"Add Variable"**
4. Add all these (one by one):

```bash
COINDCX_API_KEY=your_actual_api_key
COINDCX_SECRET=your_actual_secret
CHATGPT_API_KEY=sk-your_openai_key
CHATGPT_MODEL=gpt-4o-mini
TELEGRAM_BOT_TOKEN=123456:ABC-your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
AUTO_TRADE=false
MODE=QUICK
```

**⚠️ Important**: 
- Use real values, not placeholders
- Double-check each key
- Start with `AUTO_TRADE=false` for testing

### 2.3 Deploy

1. Click **"Deploy"** button
2. Watch build logs
3. Wait for **"Success"** message
4. Bot will start automatically

## 📱 Step 3: Verify Deployment

### 3.1 Check Logs
```
Railway Dashboard → Your Project → Deployments → Latest → Logs
```

You should see:
```
🚀 ARUN BOT STARTING...
✅ Configuration validated
✅ CoinDCX API connection successful
✅ Telegram bot connected: @YourBotName
✅ WebSocket connected
🟢 Bot is now running
```

### 3.2 Check Telegram
- You should receive startup message
- Verify bot is online

### 3.3 Test Signal (Dry Run)
- Wait for market scan (every 60 seconds)
- Check Telegram for signals
- Verify format is correct

## 🔄 Step 4: Switch to Auto-Trade (Optional)

**⚠️ Only after testing thoroughly!**

1. Railway Dashboard → Variables
2. Change `AUTO_TRADE=false` to `AUTO_TRADE=true`
3. Bot will restart automatically
4. Orders will now be placed automatically

## 🛠️ Step 5: Monitoring & Maintenance

### Daily Checks
- Monitor Telegram for signals
- Check Railway logs for errors
- Verify positions on CoinDCX

### Adjust Mode
Change trading mode anytime:
```bash
MODE=QUICK  # Fast scalping (5m)
MODE=MID    # Balanced (15m)
MODE=TREND  # High accuracy (1h)
```

### Stop Bot
```
Railway Dashboard → Your Project → Settings → Stop
```

### Restart Bot
```
Railway Dashboard → Your Project → Deployments → Redeploy
```

## 💰 Railway Pricing

### Free Tier
- **$5 free credit per month**
- Good for testing
- ~500 hours runtime

### Paid Tier
- **$5/month** for continuous running
- Better for production use

## 🐛 Troubleshooting

### Bot Not Starting?
**Check:**
1. All environment variables set?
2. API keys correct?
3. Railway logs for error messages

### No Signals?
**Possible Reasons:**
1. Market conditions not favorable
2. All 10 signals already sent today
3. Cooldown period active (30 min per pair)
4. Traps detected (check logs)

### WebSocket Disconnected?
- Auto-reconnects after 5 seconds
- Check Railway logs
- CoinDCX may have rate limits

### Orders Not Placing?
**Check:**
1. `AUTO_TRADE=true` set?
2. CoinDCX API has trading permissions?
3. Sufficient balance in account?
4. Check Railway logs for errors

## 🔒 Security Best Practices

### API Key Safety
- ✅ Store in Railway environment only
- ❌ NEVER commit to GitHub
- ✅ Use read-only keys for testing
- ✅ Enable IP whitelist on CoinDCX

### Telegram Security
- ✅ Keep bot token private
- ✅ Only add trusted users to group
- ❌ Never share chat ID publicly

## 📊 Performance Optimization

### Reduce API Calls
```bash
# In config.py, adjust:
- Scan interval (default: 60s)
- ChatGPT usage (minimal by default)
```

### Monitor Resource Usage
```
Railway Dashboard → Metrics
- CPU usage
- Memory usage
- Network traffic
```

## 🔄 Updating Bot

### Method 1: Push to GitHub
```bash
git add .
git commit -m "Update bot"
git push origin main
```
Railway auto-deploys on push.

### Method 2: Manual Redeploy
```
Railway Dashboard → Redeploy
```

## 📞 Support

### Railway Support
- [Railway Docs](https://docs.railway.app)
- Discord: railway.app/discord

### Bot Issues
- Check GitHub Issues
- Review logs first
- Test in dry run mode

## ✅ Deployment Checklist

Before going live:

- [ ] All files uploaded to GitHub
- [ ] .gitignore configured correctly
- [ ] Railway project created
- [ ] All environment variables set
- [ ] API keys verified working
- [ ] Telegram bot responding
- [ ] Tested in dry run mode (AUTO_TRADE=false)
- [ ] Verified signals received
- [ ] Checked signal quality/accuracy
- [ ] Monitored for 24 hours
- [ ] Read all risk disclaimers
- [ ] Set appropriate position sizes

---

**🎉 Congratulations! Your ARUN Bot is now live on Railway!**

Remember:
- Start small
- Monitor closely
- Test thoroughly
- Never risk more than you can afford to lose

**Happy Trading! 🚀📈**