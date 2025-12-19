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
        self.signals_sent_this_scan = 0
        self.last_reset_date = datetime.now().date()
        self.btc_last_check_time = None
        self.btc_status = 'neutral'
        self.btc_last_block_reason = None

        self.state_file = 'bot_state.json'
        self.load_state()

    # ---------- existing methods (unchanged) ----------
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

    def check_scan_limit(self):
        return self.signals_sent_this_scan < self.config.MAX_SIGNALS_PER_SCAN

    def check_btc_stability_periodic(self):
        if not self.config.ENABLE_BTC_CHECK:
            return True
        now = datetime.now()
        if self.btc_last_check_time:
            time_since = (now - self.btc_last_check_time).total_seconds()
            if time_since < (self.config.BTC_CHECK_INTERVAL_MINUTES * 60):
                return self.btc_status not in ['volatile', 'dump']
        print("🔍 BTC...", end=' ')
        is_stable, reason, status = self.signal_generator.check_btc_stability()
        self.btc_last_check_time = now
        prev_status = self.btc_status
        self.btc_status = status
        if status == 'stable':
            print("✅")
        elif status == 'neutral':
            print("ℹ️")
        elif status in ['volatile', 'dump']:
            print(f"⚠️ {status.upper()}")
        if status != prev_status and status in ['volatile', 'dump']:
            self.telegram.send_btc_block_message(reason)
        elif prev_status in ['volatile', 'dump'] and status in ['stable', 'neutral']:
            self.telegram.send_message(f"✅ <b>BTC OK</b> | {datetime.now().strftime('%H:%M:%S')}")
        return status not in ['volatile', 'dump']

    def scan_market(self, market):
        print(f"\n   📊 {market}")
        if not self.check_cooldown(market):
            mins_left = int((timedelta(minutes=self.config.COOLDOWN_MINUTES) -
                           (datetime.now() - self.last_signal_time[market])).total_seconds() / 60)
            print(f"      ⏳ SKIP: Cooldown ({mins_left}m)")
            return None
        if not self.check_daily_limit():
            print(f"      🚫 SKIP: Daily limit")
            return None
        if not self.check_scan_limit():
            print(f"      🚫 SKIP: Scan limit")
            return None
        candles_5m = self.signal_generator.fetch_candles(market, self.config.SIGNAL_TIMEFRAME, 100)
        if not candles_5m:
            print(f"      ❌ SKIP: No 5m data")
            return None
        candles_15m = self.signal_generator.fetch_candles(market, self.config.TREND_TIMEFRAME, 100)
        candles_1h = self.signal_generator.fetch_candles(market, self.config.BIAS_TIMEFRAME, 100)
        signal = self.signal_generator.generate_signal(market, candles_5m, candles_15m, candles_1h)
        if signal:
            print(f"      ✅ SIGNAL: {signal['direction']} Score={signal['score']} {signal['quality_emoji']}")
        return signal

    def scan_all_markets(self):
        print(f"\n{'=' * 60}")
        print(f"🔍 {datetime.now().strftime('%H:%M:%S')} | Daily: {self.signals_sent_today}/{self.config.MAX_SIGNALS_PER_DAY}")
        signals_in_4h = self.signal_generator.tracker.get_signals_in_window(hours=4)
        print(f"📊 Last 4h: {signals_in_4h}/{self.config.MAX_SIGNALS_PER_4_HOURS}")
        print(f"{'=' * 60}")
        self.signals_sent_this_scan = 0
        if self.config.ENABLE_BTC_CHECK:
            btc_stable = self.check_btc_stability_periodic()
            if not btc_stable:
                print(f"⛔ BTC {self.btc_status.upper()}\n")
                return []
        if signals_in_4h >= self.config.MAX_SIGNALS_PER_4_HOURS:
            print(f"⏸️ 4-hour limit reached ({signals_in_4h}/{self.config.MAX_SIGNALS_PER_4_HOURS})\n")
            return []
        all_signals = []
        for market in self.config.MARKETS:
            try:
                signal = self.scan_market(market)
                if signal:
                    all_signals.append(signal)
                time.sleep(0.2)
            except Exception:
                pass
        if not all_signals:
            return []
        all_signals.sort(key=lambda x: x['score'], reverse=True)
        remaining_slots = min(
            self.config.MAX_SIGNALS_PER_4_HOURS - signals_in_4h,
            self.config.MAX_SIGNALS_PER_DAY - self.signals_sent_today
        )
        selected = all_signals[:remaining_slots]
        print(f"\n🎯 Found: {len(all_signals)} | Selected: {len(selected)} (TOP quality)")
        if selected:
            print(f"   Scores: {[s['score'] for s in selected]}")
        return selected

    def process_signals(self, signals):
        if not signals:
            print(f"📭 No quality signals\n{'=' * 60}\n")
            return
        print(f"\n{'=' * 60}")
        print(f"🚀 SENDING {len(signals)} SIGNAL(S)")
        print(f"{'=' * 60}\n")
        for signal in signals:
            if self.telegram.send_signal(signal, self.config.LEVERAGE):
                self.signal_generator.tracker.add_signal(signal)
                self.last_signal_time[signal['market']] = datetime.now()
                self.signals_sent_today += 1
                self.signals_sent_this_scan += 1
                self.save_state()
                print(f"✅ {signal['market']} {signal['direction']} | {signal['score']} {signal['quality_emoji']}")
            time.sleep(2)
        print(f"\n{'=' * 60}\n")

    def run(self):
        print("\n" + "=" * 60)
        print("🤖 COINDCX ULTIMATE SMART BOT")
        print("=" * 60)
        if self.config.USE_AUTHENTICATED_API and self.config.COINDCX_API_KEY:
            print("🔑 AUTHENTICATED FUTURES")
        else:
            print("📊 PUBLIC SPOT → FUTURES")
        print(f"⚡ Leverage: {self.config.LEVERAGE}x")
        print(f"🎯 Base Score: {self.config.BASE_MIN_SCORE}+")
        print(f"🏆 Perfect: {self.config.PERFECT_SETUP_THRESHOLD}+")
        print(f"📊 Max/4h: {self.config.MAX_SIGNALS_PER_4_HOURS}")
        print(f"📈 Max/Day: {self.config.MAX_SIGNALS_PER_DAY}")
        print(f"🔄 Scan: {self.config.CHECK_INTERVAL_MINUTES}min")
        print(f"⏳ Cooldown: {self.config.COOLDOWN_MINUTES}min")
        print(f"🔍 BTC: {'ON' if self.config.ENABLE_BTC_CHECK else 'OFF'}")
        print(f"📈 MTF: {'STRICT' if self.config.MTF_STRICT_MODE else 'BALANCED'}")
        print(f"🐋 Volume/Whale: {'REQUIRED' if self.config.REQUIRE_VOLUME_OR_WHALE else 'BONUS'}")
        print("=" * 60 + "\n")
        self.telegram.send_startup_message(self.config)
        scan_count = 0
        while True:
            try:
                scan_count += 1
                signals = self.scan_all_markets()
                self.process_signals(signals)
                next_scan = datetime.now() + timedelta(minutes=self.config.CHECK_INTERVAL_MINUTES)
                print(f"⏰ Next: {next_scan.strftime('%H:%M:%S')} | Scan #{scan_count}")
                print(f"💤 {self.config.CHECK_INTERVAL_MINUTES}min...\n")
                time.sleep(self.config.CHECK_INTERVAL_MINUTES * 60)
            except KeyboardInterrupt:
                print("\n⏸️ Stopped")
                self.telegram.send_message("🛑 Bot stopped")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(120)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
