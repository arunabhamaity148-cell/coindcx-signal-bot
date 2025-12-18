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
        self.btc_status = 'neutral'
        self.btc_last_block_reason = None
        
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
        """
        OPTIONAL BTC check - only if enabled in config
        Checks less frequently (10 minutes)
        """
        if not self.config.ENABLE_BTC_CHECK:
            return True
        
        now = datetime.now()
        
        # Check every 10 minutes (configurable)
        if self.btc_last_check_time:
            time_since = (now - self.btc_last_check_time).total_seconds()
            if time_since < (self.config.BTC_CHECK_INTERVAL_MINUTES * 60):
                # Return cached result
                return self.btc_status not in ['volatile', 'dump']
        
        print("\n🔍 Checking BTC stability...")
        is_stable, reason, status = self.signal_generator.check_btc_stability()
        
        self.btc_last_check_time = now
        prev_status = self.btc_status
        self.btc_status = status
        
        # Log status
        if status == 'stable':
            print(f"   ✅ BTC STABLE")
        elif status == 'neutral':
            print(f"   ℹ️ BTC NEUTRAL (allowing signals)")
        elif status == 'volatile':
            print(f"   ⚠️ BTC VOLATILE: {reason}")
        elif status == 'dump':
            print(f"   🔴 BTC DUMP: {reason}")
        
        # Send notification only on critical status change
        if status != prev_status and status in ['volatile', 'dump']:
            self.telegram.send_btc_block_message(reason)
            self.btc_last_block_reason = reason
        elif prev_status in ['volatile', 'dump'] and status in ['stable', 'neutral']:
            msg = f"✅ <b>BTC STABILIZED</b>\n\n"
            msg += f"Bot resuming normal operations.\n"
            msg += f"Time: {datetime.now().strftime('%H:%M:%S')}"
            self.telegram.send_message(msg)
        
        return status not in ['volatile', 'dump']

    def scan_market(self, market):
        print(f"\n📊 {market}...", end=' ')

        # Cooldown check
        if not self.check_cooldown(market):
            mins_left = int(
                (timedelta(minutes=self.config.COOLDOWN_MINUTES) - 
                (datetime.now() - self.last_signal_time[market])).total_seconds() / 60
            )
            print(f"⏳ Cooldown ({mins_left}m)")
            return None

        # Daily limit check
        if not self.check_daily_limit():
            print(f"⚠️ Daily limit")
            return None

        # Fetch all timeframes
        candles_5m = self.signal_generator.fetch_candles(
            market, self.config.SIGNAL_TIMEFRAME, 100
        )
        
        if not candles_5m:
            print(f"❌ No 5m data")
            return None
        
        # MTF data (allow partial)
        candles_15m = self.signal_generator.fetch_candles(
            market, self.config.TREND_TIMEFRAME, 100
        )
        candles_1h = self.signal_generator.fetch_candles(
            market, self.config.BIAS_TIMEFRAME, 100
        )

        # Generate signal (works even with partial MTF data)
        signal = self.signal_generator.generate_signal(
            market, candles_5m, candles_15m, candles_1h
        )

        if signal:
            print(f"🎯 {signal['direction']} {signal['score']}")
            return signal
        else:
            print(f"⏭️")
            return None

    def scan_all_markets(self):
        print(f"\n{'=' * 60}")
        print(f"🔍 SCAN - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'=' * 60}")

        # BTC check (if enabled)
        if self.config.ENABLE_BTC_CHECK:
            btc_stable = self.check_btc_stability_periodic()
            if not btc_stable:
                print(f"\n⛔ BTC {self.btc_status.upper()} - Blocking signals")
                print(f"{'=' * 60}\n")
                return []
        else:
            print("ℹ️ BTC check disabled")

        signals_found = []
        errors = 0

        for market in self.config.MARKETS:
            try:
                signal = self.scan_market(market)
                if signal:
                    signals_found.append(signal)
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                errors += 1
                print(f"   ❌ Error: {e}")
                if errors > 5:
                    print("   ⚠️ Too many errors, continuing...")
                    break

        return signals_found

    def process_signals(self, signals):
        if not signals:
            print(f"\n📭 No signals this scan")
            print(f"{'=' * 60}\n")
            return

        print(f"\n{'=' * 60}")
        print(f"🎉 {len(signals)} signal(s) found!")
        print(f"{'=' * 60}")

        for signal in signals:
            if self.telegram.send_signal(signal, self.config.LEVERAGE):
                self.last_signal_time[signal['market']] = datetime.now()
                self.signals_sent_today += 1
                self.save_state()
                print(f"✅ Sent: {signal['market']} {signal['direction']} (Score: {signal['score']})")
                print(f"   Today: {self.signals_sent_today}/{self.config.MAX_SIGNALS_PER_DAY}")
            time.sleep(1.5)

        print(f"{'=' * 60}\n")

    def run(self):
        print("\n" + "=" * 60)
        print("🤖 COINDCX INR FUTURES BOT v2.0")
        print("=" * 60)
        print(f"📊 Markets: {len(self.config.MARKETS)} pairs")
        print(f"⚡ Leverage: {self.config.LEVERAGE}x")
        print(f"⏱️ Signal: {self.config.SIGNAL_TIMEFRAME}")
        print(f"🎯 Min Score: {self.config.MIN_SIGNAL_SCORE}")
        print(f"🔄 Scan: {self.config.CHECK_INTERVAL_MINUTES}min")
        print(f"⏳ Cooldown: {self.config.COOLDOWN_MINUTES}min")
        print(f"🔍 BTC Check: {'ON' if self.config.ENABLE_BTC_CHECK else 'OFF'}")
        print(f"📈 MTF Mode: {'STRICT' if self.config.MTF_STRICT_MODE else 'RELAXED'}")
        print("=" * 60 + "\n")

        self.telegram.send_startup_message(self.config)

        heartbeat_counter = 0
        scan_count = 0

        while True:
            try:
                scan_count += 1
                signals = self.scan_all_markets()
                self.process_signals(signals)

                heartbeat_counter += 1
                if heartbeat_counter >= 20:
                    self.telegram.send_heartbeat(self.signals_sent_today)
                    heartbeat_counter = 0

                next_scan = datetime.now() + timedelta(
                    minutes=self.config.CHECK_INTERVAL_MINUTES
                )
                print(f"⏰ Next: {next_scan.strftime('%H:%M:%S')} | Scans: {scan_count} | Signals: {self.signals_sent_today}")
                print(f"💤 Sleeping {self.config.CHECK_INTERVAL_MINUTES}m...\n")

                time.sleep(self.config.CHECK_INTERVAL_MINUTES * 60)

            except KeyboardInterrupt:
                print("\n\n⏸️ Bot stopped by user")
                self.telegram.send_message("🛑 Bot stopped manually")
                break
            except Exception as e:
                print(f"\n❌ Main loop error: {str(e)}")
                print("🔄 Retrying in 2min...\n")
                time.sleep(120)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()