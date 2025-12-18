import requests
import time
from datetime import datetime, timedelta
from config import Config
from signal_logic import SignalGenerator
import json
import os

class TradingBot:
    
    def __init__(self):
        self.config = Config()
        self.signal_generator = SignalGenerator()
        self.last_signal_time = {}
        self.signals_sent_today = 0
        self.last_reset_date = datetime.now().date()
        self.state_file = 'bot_state.json'
        self.load_state()
    
    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.last_signal_time = {k: datetime.fromisoformat(v) for k, v in state.get('last_signal_time', {}).items()}
                    self.signals_sent_today = state.get('signals_sent_today', 0)
                    saved_date = state.get('last_reset_date')
                    if saved_date:
                        self.last_reset_date = datetime.fromisoformat(saved_date).date()
        except Exception as e:
            print(f"State load error: {e}")
    
    def save_state(self):
        try:
            state = {
                'last_signal_time': {k: v.isoformat() for k, v in self.last_signal_time.items()},
                'signals_sent_today': self.signals_sent_today,
                'last_reset_date': self.last_reset_date.isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"State save error: {e}")
    
    def fetch_candles(self, market, interval='15m', limit=100):
        try:
            url = f"{self.config.COINDCX_BASE_URL}/market_data/candles"
            params = {'pair': market, 'interval': interval, 'limit': limit}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return None
            
            candles = []
            for candle in data:
                candles.append({
                    'time': candle['time'],
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle['volume'])
                })
            
            return candles
            
        except Exception as e:
            print(f"Error fetching {market}: {str(e)}")
            return None
    
    def check_cooldown(self, market):
        if market not in self.last_signal_time:
            return True
        
        time_since = datetime.now() - self.last_signal_time[market]
        cooldown = timedelta(minutes=self.config.COOLDOWN_MINUTES)
        
        return time_since >= cooldown
    
    def check_daily_limit(self):
        current_date = datetime.now().date()
        
        if current_date != self.last_reset_date:
            self.signals_sent_today = 0
            self.last_reset_date = current_date
            self.save_state()
        
        return self.signals_sent_today < self.config.MAX_SIGNALS_PER_DAY
    
    def format_telegram_message(self, signal):
        emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
        
        confidence = 'HIGH' if signal['score'] >= 85 else 'MEDIUM'
        
        msg = f"{emoji * 3} {signal['direction']} SIGNAL {emoji * 3}\n\n"
        msg += f"💹 Market: {signal['market']}\n"
        msg += f"📊 Score: {signal['score']}/100\n"
        msg += f"⚡ Confidence: {confidence}\n\n"
        
        msg += f"💰 Entry: {signal['entry']:.8f}\n"
        msg += f"🎯 TP1: {signal['tp1']:.8f}\n"
        msg += f"🎯 TP2: {signal['tp2']:.8f}\n"
        msg += f"🛑 SL: {signal['sl']:.8f}\n\n"
        
        msg += f"📈 R:R = 1:{signal['rr_ratio']:.2f}\n\n"
        
        msg += "📊 Analysis:\n"
        msg += f"• RSI: {signal['analysis']['rsi']:.1f}\n"
        msg += f"• ADX: {signal['analysis']['adx']:.1f}\n"
        msg += f"• Regime: {signal['analysis']['market_regime'].upper()}\n\n"
        
        msg += "✅ Reasons:\n"
        for i, reason in enumerate(signal['reasons'][:5], 1):
            msg += f"{i}. {reason}\n"
        
        msg += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}\n"
        msg += "⚠️ Use proper risk management!"
        
        return msg
    
    def send_telegram_message(self, message):
        try:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {'chat_id': self.config.TELEGRAM_CHAT_ID, 'text': message}
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Telegram error: {str(e)}")
            return False
    
    def send_telegram_signal(self, signal):
        try:
            message = self.format_telegram_message(signal)
            
            if self.send_telegram_message(message):
                print(f"✅ Signal sent: {signal['market']} {signal['direction']}")
                return True
            return False
            
        except Exception as e:
            print(f"❌ Signal send error: {str(e)}")
            return False
    
    def send_startup_message(self):
        try:
            msg = "🚀 BOT DEPLOYED SUCCESSFULLY!\n\n"
            msg += f"✅ Status: ACTIVE\n"
            msg += f"📊 Markets: {len(self.config.MARKETS)} pairs\n"
            msg += f"⏱️ Interval: {self.config.CANDLE_INTERVAL}\n"
            msg += f"🔄 Scan Every: {self.config.CHECK_INTERVAL_MINUTES} min\n"
            msg += f"🎯 Min Score: {self.config.MIN_SIGNAL_SCORE}\n"
            msg += f"⏳ Cooldown: {self.config.COOLDOWN_MINUTES} min\n\n"
            msg += f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            msg += "Bot is now scanning markets...\n"
            msg += "Signals will arrive when conditions are met! 📈"
            
            self.send_telegram_message(msg)
            print("✅ Startup message sent to Telegram")
        except Exception as e:
            print(f"⚠️ Could not send startup message: {e}")
    
    def send_heartbeat(self):
        try:
            msg = f"💚 Bot Heartbeat\n"
            msg += f"🕐 {datetime.now().strftime('%H:%M:%S')}\n"
            msg += f"📊 Signals Today: {self.signals_sent_today}\n"
            msg += f"✅ Status: Running"
            self.send_telegram_message(msg)
        except:
            pass
    
    def scan_market(self, market):
        print(f"\n📊 Scanning {market}...")
        
        if not self.check_cooldown(market):
            mins_left = int((timedelta(minutes=self.config.COOLDOWN_MINUTES) - (datetime.now() - self.last_signal_time[market])).total_seconds() / 60)
            print(f"   ⏳ Cooldown active ({mins_left} min left)")
            return None
        
        if not self.check_daily_limit():
            print(f"   ⚠️ Daily limit reached ({self.config.MAX_SIGNALS_PER_DAY})")
            return None
        
        candles = self.fetch_candles(market, self.config.CANDLE_INTERVAL)
        
        if not candles:
            print(f"   ❌ No candle data")
            return None
        
        signal = self.signal_generator.generate_signal(market, candles)
        
        if signal:
            print(f"   🎯 ✅ SIGNAL GENERATED! Score: {signal['score']}/100")
            return signal
        else:
            print(f"   ⏭️ No signal (see details above)")
            return None
    
    def scan_all_markets(self):
        print(f"\n{'=' * 60}")
        print(f"🔍 SCAN START - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")
        
        signals_found = []
        
        for market in self.config.MARKETS:
            signal = self.scan_market(market)
            
            if signal:
                signals_found.append(signal)
            
            time.sleep(0.5)
        
        return signals_found
    
    def process_signals(self, signals):
        if not signals:
            print(f"\n📭 No signals found this scan")
            print(f"{'=' * 60}\n")
            return
        
        print(f"\n{'=' * 60}")
        print(f"🎉 Found {len(signals)} signal(s)!")
        print(f"{'=' * 60}\n")
        
        for signal in signals:
            if self.send_telegram_signal(signal):
                self.last_signal_time[signal['market']] = datetime.now()
                self.signals_sent_today += 1
                self.save_state()
                
                print(f"✅ Processed: {signal['market']}")
                print(f"📊 Today's Total: {self.signals_sent_today}/{self.config.MAX_SIGNALS_PER_DAY}\n")
            
            time.sleep(2)
        
        print(f"{'=' * 60}\n")
    
    def run(self):
        print("\n" + "=" * 60)
        print("🤖 COINDCX SIGNAL BOT - PROFESSIONAL")
        print("=" * 60)
        print(f"📊 Markets: {len(self.config.MARKETS)} pairs")
        print(f"⏱️ Interval: {self.config.CANDLE_INTERVAL}")
        print(f"🎯 Min Score: {self.config.MIN_SIGNAL_SCORE}")
        print(f"🔄 Scan Every: {self.config.CHECK_INTERVAL_MINUTES} min")
        print(f"⏳ Cooldown: {self.config.COOLDOWN_MINUTES} min")
        print(f"📈 Daily Limit: {self.config.MAX_SIGNALS_PER_DAY}")
        print("=" * 60 + "\n")
        
        # Send startup notification
        self.send_startup_message()
        
        heartbeat_counter = 0
        
        while True:
            try:
                signals = self.scan_all_markets()
                self.process_signals(signals)
                
                # Heartbeat every 12 scans (60 min if 5min interval)
                heartbeat_counter += 1
                if heartbeat_counter >= 12:
                    self.send_heartbeat()
                    heartbeat_counter = 0
                
                next_scan = datetime.now() + timedelta(minutes=self.config.CHECK_INTERVAL_MINUTES)
                print(f"⏰ Next scan: {next_scan.strftime('%H:%M:%S')}")
                print(f"💤 Sleeping {self.config.CHECK_INTERVAL_MINUTES} min...\n")
                
                time.sleep(self.config.CHECK_INTERVAL_MINUTES * 60)
                
            except KeyboardInterrupt:
                print("\n\n⏸️ Bot stopped by user")
                self.send_telegram_message("🛑 Bot stopped manually")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {str(e)}")
                print("🔄 Retrying in 2 min...\n")
                time.sleep(120)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
```

---

## ✅ **Key Features Added:**

1. ✅ **Startup Message** - Deploy হলেই Telegram এ notification
2. ✅ **Detailed Scan Logs** - প্রতিটা pair এর score দেখাবে
3. ✅ **Cooldown Display** - কত মিনিট বাকি আছে
4. ✅ **Heartbeat** - Every hour bot alive confirmation
5. ✅ **Better Formatting** - Clean, readable logs
6. ✅ **State Persistence** - Restart হলেও memory থাকবে

---

## 📱 **Telegram এ এরকম Message আসবে:**

**Deploy হলেই:**
```
🚀 BOT DEPLOYED SUCCESSFULLY!

✅ Status: ACTIVE
📊 Markets: 15 pairs
⏱️ Interval: 15m
🔄 Scan Every: 5 min
🎯 Min Score: 75
⏳ Cooldown: 30 min

🕐 Started: 2025-12-18 10:30:45

Bot is now scanning markets...
Signals will arrive when conditions are met! 📈
```

**Signal পেলে:**
```
🟢🟢🟢 LONG SIGNAL 🟢🟢🟢

💹 Market: B-BTC_USDT
📊 Score: 82/100
⚡ Confidence: MEDIUM

💰 Entry: 0.00043210
🎯 TP1: 0.00044500
🎯 TP2: 0.00046200
🛑 SL: 0.00042100

📈 R:R = 1:2.45

📊 Analysis:
- RSI: 42.5
- ADX: 28.3
- Regime: TRENDING

✅ Reasons:
1. EMA bullish
2. RSI oversold recovery (42.5)
3. MACD bullish
4. Strong trend (ADX 28.3)
5. Trending regime

🕐 15:30:22
⚠️ Use proper risk management!
```

**Heartbeat (Every Hour):**
```
💚 Bot Heartbeat
🕐 16:30:45
📊 Signals Today: 3
✅ Status: Running
```

---

## 🚀 **Deploy Instructions:**

1. **Replace bot.py** with this full code
2. **Keep signal_logic.py** updated (previous version with logging)
3. **Push to GitHub**
4. **Railway auto-deploys**
5. **Check Telegram** - startup message আসবে!

---

## 📊 **Railway Logs এ দেখবে:**
```
============================================================
🔍 SCAN START - 2025-12-18 10:35:00
============================================================

📊 Scanning B-BTC_USDT...
   📊 B-BTC_USDT: LONG score = 55/100 (need 75+)
   ⚠️ Score too low. Missing:
      • No bullish pattern
      • Weak trend (ADX: 19.2)
      • Weak order flow (0.12)
   ⏭️ No signal (see details above)

📊 Scanning B-ETH_USDT...
   ❌ B-ETH_USDT: BLOCKED: Ranging market (ADX: 14.8)
   📊 B-ETH_USDT: LONG score = 0/100 (need 75+)
   ⏭️ No signal (see details above)

...

📭 No signals found this scan
============================================================

⏰ Next scan: 10:40:00
💤 Sleeping 5 min...
