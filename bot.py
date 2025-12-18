import time
from datetime import datetime, timedelta
from config import Config
from signal_logic import SignalGenerator
from telegram_utils import TelegramUtils
import json
import os

class TradingBot:

    def __init__(self):
        self.config = Config()
        self.signal_generator = SignalGenerator()
        self.telegram = TelegramUtils(
            self.config.TELEGRAM_BOT_TOKEN,
            self.config.TELEGRAM_CHAT_ID
        )
        
        self.last_signal_time = {}
        self.signals_sent_today = 0
        self.last_reset_date = datetime.now().date()
        self.btc_last_check_time = None
        self.btc_is_stable = True
        self.btc_block_notified = False
        
        self.state_file = 'bot_state.json'
        self.load_state()

    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.last_signal_time = {
                        k: datetime.fromisoformat(v) 
                        for k, v in state.get('last_signal_time', {}).items()
                    }
                    self.signals_sent_today = state.get('signals_sent_today', 0)
                    saved_date = state.get('last_reset_date')
                    if saved_date:
                        self.last_reset_date = datetime.fromisoformat(saved_date).date()
        except Exception as e:
            print(f"State load error: {e}")

    def save_state(self):
        try:
            state = {
                'last_signal_time': {
                    k: v.isoformat() 
                    for k, v in self.last_signal_time.items()
                },
                'signals_sent_today': self.signals_sent_today,
                'last_reset_date': self.last_reset_date.isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"State save error: {e}")

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

    def check_btc_stability_periodic(self):
        """Check BTC stability every 5 minutes"""
        now = datetime.now()
        
        if self.btc_last_check_time:
            time_since = (now - self.btc_last_check_time).total_seconds()
            if time_since < 300:  # 5 minutes
                return self.btc_is_stable
        
        print("\n🔍 Checking BTC stability...")
        stable, reason = self.signal_generator.check_btc_stability()
        
        self.btc_last_check_time = now
        
        # State changed
        if stable != self.btc_is_stable:
            self.btc_is_stable = stable
            
            if not stable:
                print(f"   ⚠️ BTC UNSTABLE: {reason}")
                if not self.btc_block_notified:
                    self.telegram.send_btc_block_message(reason)
                    self.btc_block_notified = True
            else:
                print(f"   ✅ BTC STABLE")
                self.btc_block_notified = False
        
        return self.btc_is_stable

    def scan_market(self, market):
        print(f"\n📊 Scanning {market}...")

        # Cooldown check
        if not self.check_cooldown(market):
            mins_left = int(
                (timedelta(minutes=self.config.COOLDOWN_MINUTES) - 
                (datetime.now() - self.last_signal_time[market])).total_seconds() / 60
            )
            print(f"   ⏳ Cooldown active ({mins_left} min left)")
            return None

        # Daily limit check
        if not self.check_daily_limit():
            print(f"   ⚠️ Daily limit reached ({self.config.MAX_SIGNALS_PER_DAY})")
            return None

        # Fetch all timeframes
        candles_5m = self.signal_generator.fetch_candles(
            market, self.config.SIGNAL_TIMEFRAME, 100
        )
        candles_15m = self.signal_generator.fetch_candles(
            market, self.config.TREND_TIMEFRAME, 100
        )
        candles_1h = self.signal_generator.fetch_candles(
            market, self.config.BIAS_TIMEFRAME, 100
        )

        if not candles_5m or not candles_15m or not candles_1h:
            print(f"   ❌ Incomplete data")
            return None

        # Generate signal
        signal = self.signal_generator.generate_signal(
            market, candles_5m, candles_15m, candles_1h
        )

        if signal:
            print(f"   🎯 SIGNAL! {signal['direction']} Score: {signal['score']}")
            return signal
        else:
            print(f"   ⏭️ No signal")
            return None

    def scan_all_markets(self):
        print(f"\n{'=' * 60}")
        print(f"🔍 SCAN START - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        # BTC stability check
        if not self.check_btc_stability_periodic():
            print("\n⛔ BTC UNSTABLE - All scans blocked")
            print(f"{'=' * 60}\n")
            return []

        signals_found = []

        for market in self.config.MARKETS:
            try:
                signal = self.scan_market(market)
                if signal:
                    signals_found.append(signal)
                time.sleep(0.5)
            except Exception as e:
                print(f"   ❌ Error: {e}")

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
            if self.telegram.send_signal(signal, self.config.LEVERAGE):
                self.last_signal_time[signal['market']] = datetime.now()
                self.signals_sent_today += 1
                self.save_state()
                print(f"✅ Sent: {signal['market']} {signal['direction']}")
                print(f"   Today Total: {self.signals_sent_today}/{self.config.MAX_SIGNALS_PER_DAY}\n")
            time.sleep(2)

        print(f"{'=' * 60}\n")

    def run(self):
        print("\n" + "=" * 60)
        print("🤖 COINDCX INR FUTURES SIGNAL BOT")
        print("=" * 60)
        print(f"📊 Markets: {len(self.config.MARKETS)} INR Futures pairs")
        print(f"⚡ Leverage: {self.config.LEVERAGE}x")
        print(f"⏱️ Signal TF: {self.config.SIGNAL_TIMEFRAME}")
        print(f"🎯 Min Score: {self.config.MIN_SIGNAL_SCORE}")
        print(f"🔄 Scan Every: {self.config.CHECK_INTERVAL_MINUTES} min")
        print(f"⏳ Cooldown: {self.config.COOLDOWN_MINUTES} min")
        print(f"📈 Daily Limit: {self.config.MAX_SIGNALS_PER_DAY}")
        print("=" * 60 + "\n")

        self.telegram.send_startup_message(self.config)

        heartbeat_counter = 0

        while True:
            try:
                signals = self.scan_all_markets()
                self.process_signals(signals)

                heartbeat_counter += 1
                if heartbeat_counter >= 20:  # Every ~1 hour
                    self.telegram.send_heartbeat(self.signals_sent_today)
                    heartbeat_counter = 0

                next_scan = datetime.now() + timedelta(
                    minutes=self.config.CHECK_INTERVAL_MINUTES
                )
                print(f"⏰ Next scan: {next_scan.strftime('%H:%M:%S')}")
                print(f"💤 Sleeping {self.config.CHECK_INTERVAL_MINUTES} min...\n")

                time.sleep(self.config.CHECK_INTERVAL_MINUTES * 60)

            except KeyboardInterrupt:
                print("\n\n⏸️ Bot stopped by user")
                self.telegram.send_message("🛑 Bot stopped manually")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {str(e)}")
                print("🔄 Retrying in 2 min...\n")
                time.sleep(120)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()