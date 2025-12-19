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
        """OPTIONAL BTC check"""
        if not self.config.ENABLE_BTC_CHECK:
            return True
        
        now = datetime.now()
        
        if self.btc_last_check_time:
            time_since = (now - self.btc_last_check_time).total_seconds()
            if time_since < (self.config.BTC_CHECK_INTERVAL_MINUTES * 60):
                return self.btc_status not in ['volatile', 'dump']
        
        print("\n🔍 Checking BTC...")
        is_stable, reason, status = self.signal_generator.check_btc_stability()
        
        self.btc_last_check_time = now
        prev_status = self.btc_status
        self.btc_status = status
        
        if status == 'stable':
            print(f"   ✅ BTC OK")
        elif status == 'neutral':
            print(f"   ℹ️ BTC NEUTRAL")
        elif status in ['volatile', 'dump']:
            print(f"   ⚠️ BTC {status.upper()}: {reason}")
        
        if status != prev_status and status in ['volatile', 'dump']:
            self.telegram.send_btc_block_message(reason)
        elif prev_status in ['volatile', 'dump'] and status in ['stable', 'neutral']:
            self.telegram.send_message(f"✅ <b>BTC STABILIZED</b>\nTime: {datetime.now().strftime('%H:%M:%S')}")
        
        return status not in ['volatile', 'dump']

    def scan_market(self, market):
        print(f"📊 {market}...", end=' ')

        if not self.check_cooldown(market):
            mins_left = int(
                (timedelta(minutes=self.config.COOLDOWN_MINUTES) - 
                (datetime.now() - self.last_signal_time[market])).total_seconds() / 60
            )
            print(f"⏳ {mins_left}m")
            return None

        if not self.check_daily_limit():
            print(f"⚠️ Limit")
            return None

        # Fetch candles
        candles_5m = self.signal_generator.fetch_candles(
            market, self.config.SIGNAL_TIMEFRAME, 100
        )
        
        if not candles_5m:
            print(f"❌ No data")
            return None
        
        candles_15m = self.signal_generator.fetch_candles(
            market, self.config.TREND_TIMEFRAME, 100
        )
        candles_1h = self.signal_generator.fetch_candles(
            market, self.config.BIAS_TIMEFRAME, 100
        )

        signal = self.signal_generator.generate_signal(
            market, candles_5m, candles_15m, candles_1h
        )

        if signal:
            print(f"🎯 {signal['direction']} {signal['score']} {signal['quality_emoji']}")
            return signal
        else:
            print(f"—")
            return None

    def scan_all_markets(self):
        print(f"\n{'=' * 60}")
        print(f"🔍 SCAN - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'=' * 60}")

        # Show mode
        if self.config.USE_AUTHENTICATED_API and self.config.COINDCX_API_KEY:
            print("🔑 Mode: AUTHENTICATED (Futures)")
        else:
            print("📊 Mode: PUBLIC (Spot → Futures)")

        # BTC check
        if self.config.ENABLE_BTC_CHECK:
            btc_stable = self.check_btc_stability_periodic()
            if not btc_stable:
                print(f"\n⛔ BTC {self.btc_status.upper()} - Blocking\n")
                return []
        
        print()
        signals_found = []
        errors = 0

        for market in self.config.MARKETS:
            try:
                signal = self.scan_market(market)
                if signal:
                    signals_found.append(signal)
                time.sleep(0.3)
            except Exception as e:
                errors += 1
                print(f"❌ {market}: {e}")
                if errors > 5:
                    break

        return signals_found

    def process_signals(self, signals):
        if not signals:
            print(f"\n📭 No signals\n{'=' * 60}\n")
            return

        print(f"\n{'=' * 60}")
        print(f"🎉 {len(signals)} SIGNAL(S) FOUND!")
        print(f"{'=' * 60}\n")

        for signal in signals:
            if self.telegram.send_signal(signal, self.config.LEVERAGE):
                # Store with original market name for cooldown
                self.last_signal_time[signal['market']] = datetime.now()
                self.signals_sent_today += 1
                self.save_state()
                print(f"✅ {signal['market']} {signal['direction']} (Score: {signal['score']})")
            time.sleep(1.5)

        print(f"\n{'=' * 60}\n")

    def run(self):
        print("\n" + "=" * 60)
        print("🤖 COINDCX SIGNAL BOT v2.0")
        print("=" * 60)
        
        # Determine mode
        if self.config.USE_AUTHENTICATED_API and self.config.COINDCX_API_KEY:
            print(f"🔑 Mode: AUTHENTICATED FUTURES")
            print(f"📊 Markets: {len(self.config.FUTURES_MARKETS)} Futures")
        else:
            print(f"📊 Mode: PUBLIC SPOT → FUTURES")
            print(f"📊 Markets: {len(list(self.config.SPOT_TO_FUTURES_MAP.keys()))} Spot")
        
        print(f"⚡ Leverage: {self.config.LEVERAGE}x")
        print(f"⏱️ Signal: {self.config.SIGNAL_TIMEFRAME}")
        print(f"🎯 Min Score: {self.config.MIN_SIGNAL_SCORE}")
        print(f"🔄 Scan: {self.config.CHECK_INTERVAL_MINUTES}min")
        print(f"⏳ Cooldown: {self.config.COOLDOWN_MINUTES}min")
        print(f"🔍 BTC Check: {'ON' if self.config.ENABLE_BTC_CHECK else 'OFF'}")
        print(f"📈 MTF: {'STRICT' if self.config.MTF_STRICT_MODE else 'RELAXED'}")
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
                print(f"💤 Sleep {self.config.CHECK_INTERVAL_MINUTES}m...\n")

                time.sleep(self.config.CHECK_INTERVAL_MINUTES * 60)

            except KeyboardInterrupt:
                print("\n\n⏸️ Stopped")
                self.telegram.send_message("🛑 Bot stopped")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("🔄 Retry 2min...\n")
                time.sleep(120)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()