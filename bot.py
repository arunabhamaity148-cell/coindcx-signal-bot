import requests
import time
from datetime import datetime, timedelta
from config import Config
from signal_logic import SignalGenerator
import json

class TradingBot:
    
    def __init__(self):
        self.config = Config()
        self.signal_generator = SignalGenerator()
        self.last_signal_time = {}
        self.signals_sent_today = 0
        self.last_reset_date = datetime.now().date()
    
    def fetch_candles(self, market, interval='15m', limit=100):
        """Fetch candle data from CoinDCX"""
        try:
            url = f"{self.config.COINDCX_BASE_URL}/market_data/candles"
            params = {
                'pair': market,
                'interval': interval,
                'limit': limit
            }
            
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
            print(f"Error fetching candles for {market}: {str(e)}")
            return None
    
    def check_cooldown(self, market):
        """Check if market is in cooldown period"""
        if market not in self.last_signal_time:
            return True
        
        time_since_last = datetime.now() - self.last_signal_time[market]
        cooldown_period = timedelta(minutes=self.config.COOLDOWN_MINUTES)
        
        return time_since_last >= cooldown_period
    
    def check_daily_limit(self):
        """Check if daily signal limit reached"""
        current_date = datetime.now().date()
        
        if current_date != self.last_reset_date:
            self.signals_sent_today = 0
            self.last_reset_date = current_date
        
        return self.signals_sent_today < self.config.MAX_SIGNALS_PER_DAY
    
    def format_telegram_message(self, signal):
        """Format signal for Telegram"""
        emoji_dir = "🟢" if signal['direction'] == 'LONG' else "🔴"
        
        msg = f"{emoji_dir * 3} {signal['direction']} SIGNAL {emoji_dir * 3}\n\n"
        msg += f"💹 Market: {signal['market']}\n"
        msg += f"📊 Score: {signal['score']}/100\n"
        msg += f"⚡ Confidence: {'HIGH' if signal['score'] >= 85 else 'MEDIUM' if signal['score'] >= 75 else 'LOW'}\n\n"
        
        msg += f"💰 Entry: {signal['entry']:.8f}\n"
        msg += f"🎯 TP1: {signal['tp1']:.8f}\n"
        msg += f"🎯 TP2: {signal['tp2']:.8f}\n"
        msg += f"🛑 Stop Loss: {signal['sl']:.8f}\n\n"
        
        msg += f"📈 Risk/Reward: 1:{signal['rr_ratio']:.2f}\n\n"
        
        msg += "📊 Technical Analysis:\n"
        msg += f"• RSI: {signal['analysis']['rsi']:.1f}\n"
        msg += f"• ADX: {signal['analysis']['adx']:.1f}\n"
        msg += f"• Regime: {signal['analysis']['market_regime'].upper()}\n\n"
        
        msg += "✅ Signal Logic:\n"
        for i, reason in enumerate(signal['reasons'][:5], 1):
            msg += f"{i}. {reason}\n"
        
        msg += f"\n🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        msg += f"⚠️ Always use proper risk management!"
        
        return msg
    
    def send_telegram_signal(self, signal):
        """Send signal to Telegram"""
        try:
            message = self.format_telegram_message(signal)
            
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            payload = {
                'chat_id': self.config.TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            print(f"✅ Signal sent for {signal['market']} ({signal['direction']})")
            return True
            
        except Exception as e:
            print(f"❌ Error sending Telegram message: {str(e)}")
            return False
    
    def scan_market(self, market):
        """Scan single market for signals"""
        print(f"📊 Scanning {market}...")
        
        if not self.check_cooldown(market):
            print(f"⏳ {market} in cooldown period")
            return None
        
        if not self.check_daily_limit():
            print(f"⚠️ Daily signal limit reached ({self.config.MAX_SIGNALS_PER_DAY})")
            return None
        
        candles = self.fetch_candles(market, self.config.CANDLE_INTERVAL)
        
        if not candles:
            print(f"❌ No candle data for {market}")
            return None
        
        signal = self.signal_generator.generate_signal(market, candles)
        
        if signal:
            print(f"🎯 Signal found! Score: {signal['score']}")
            return signal
        else:
            print(f"⏭️ No signal")
            return None
    
    def scan_all_markets(self):
        """Scan all configured markets"""
        print(f"\n{'='*60}")
        print(f"🔍 SCANNING ALL MARKETS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        signals_found = []
        
        for market in self.config.MARKETS:
            signal = self.scan_market(market)
            
            if signal:
                signals_found.append(signal)
            
            time.sleep(1)
        
        return signals_found
    
    def process_signals(self, signals):
        """Process and send signals"""
        if not signals:
            print("\n📭 No signals found in this scan\n")
            return
        
        print(f"\n🎉 Found {len(signals)} signal(s)!")
        
        for signal in signals:
            if self.send_telegram_signal(signal):
                self.last_signal_time[signal['market']] = datetime.now()
                self.signals_sent_today += 1
                
                print(f"✅ Processed signal for {signal['market']}")
                print(f"📊 Signals sent today: {self.signals_sent_today}/{self.config.MAX_SIGNALS_PER_DAY}\n")
            
            time.sleep(2)
    
    def run(self):
        """Main bot loop"""
        print("\n" + "="*60)
        print("🤖 COINDCX PROFESSIONAL TRADING BOT STARTED")
        print("="*60)
        print(f"📊 Markets: {', '.join(self.config.MARKETS)}")
        print(f"⏱️ Interval: {self.config.CANDLE_INTERVAL}")
        print(f"🎯 Min Score: {self.config.MIN_SIGNAL_SCORE}")
        print(f"🔄 Check Every: {self.config.CHECK_INTERVAL_MINUTES} minutes")
        print(f"⏳ Cooldown: {self.config.COOLDOWN_MINUTES} minutes")
        print(f"📈 Max Signals/Day: {self.config.MAX_SIGNALS_PER_DAY}")
        print("="*60 + "\n")
        
        while True:
            try:
                signals = self.scan_all_markets()
                self.process_signals(signals)
                
                next_scan = datetime.now() + timedelta(minutes=self.config.CHECK_INTERVAL_MINUTES)
                print(f"⏰ Next scan at: {next_scan.strftime('%H:%M:%S')}")
                print(f"💤 Sleeping for {self.config.CHECK_INTERVAL_MINUTES} minutes...\n")
                
                time.sleep(self.config.CHECK_INTERVAL_MINUTES * 60)
                
            except KeyboardInterrupt:
                print("\n\n⏸️ Bot stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {str(e)}")
                print("🔄 Retrying in 5 minutes...\n")
                time.sleep(300)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()