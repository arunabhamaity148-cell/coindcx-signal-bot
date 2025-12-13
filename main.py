"""
🤖 CoinDCX Futures Trading Bot - Main Orchestrator (FIXED)
Daily ₹2000 Target | Manual Trading | 45 Logic Engine
"""

import time
from datetime import datetime
import sys

from config import *
from market_scanner import MarketScanner
from signal_engine import SignalEngine
from risk_manager import RiskManager
from telegram_bot import TelegramBot
from performance_tracker import PerformanceTracker


class CoinDCXTradingBot:
    def __init__(self):
        print("🚀 Initializing CoinDCX Futures Trading Bot...")

        self.scanner = MarketScanner()
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        self.telegram = TelegramBot()
        self.tracker = PerformanceTracker()

        self.signals_today = 0
        self.last_report_time = datetime.now()
        self.is_running = True
        self.current_day = datetime.now().date()

        print("✅ All modules loaded successfully!")

    # --------------------------------------------------
    # Daily reset (VERY IMPORTANT)
    # --------------------------------------------------
    def _daily_reset_if_needed(self):
        if datetime.now().date() != self.current_day:
            print("🔄 New day detected — resetting counters")
            self.signals_today = 0
            self.signal_engine.consecutive_losses = 0
            self.signal_engine.last_signal_time = {}
            self.current_day = datetime.now().date()

    # --------------------------------------------------
    def check_daily_limits(self):
        if self.signals_today >= MAX_DAILY_SIGNALS:
            print(f"⚠️ Daily signal limit reached ({self.signals_today}/{MAX_DAILY_SIGNALS})")
            return False

        stats = self.tracker.get_daily_summary()
        if stats["total_pnl"] >= TARGET_DAILY_PROFIT:
            print("🎉 Daily target achieved!")
            self.telegram.send_message(
                f"🎉 <b>DAILY TARGET ACHIEVED</b>\nPnL: ₹{stats['total_pnl']:.2f}"
            )
            return False

        return True

    # --------------------------------------------------
    def send_hourly_report(self):
        if (datetime.now() - self.last_report_time).seconds >= PERFORMANCE_REPORT_INTERVAL:
            stats = self.tracker.get_daily_summary()
            self.telegram.send_performance_report(stats)
            self.last_report_time = datetime.now()

    # --------------------------------------------------
    def process_signal_mode(self, mode, tradeable_pairs, processed_symbols):
        print(f"\n🔍 Scanning {mode.upper()} mode...")

        for pair_data in tradeable_pairs:
            if self.signals_today >= MAX_DAILY_SIGNALS:
                break

            # ❌ FIX 1: data key missing guard
            if "data" not in pair_data or pair_data["data"] is None:
                continue

            symbol = pair_data["symbol"]

            # ❌ FIX 2: one symbol → one mode only
            if symbol in processed_symbols:
                continue

            try:
                signal = self.signal_engine.generate_signal(pair_data, mode)
                if signal is None:
                    continue

                levels = self.risk_manager.calculate_tp_sl(signal)
                if levels is None:
                    continue

                # ✅ SIGNAL ACCEPTED
                processed_symbols.add(symbol)
                self.signals_today += 1

                print("\n🎯 SIGNAL GENERATED")
                print(f"Pair      : {symbol}")
                print(f"Direction : {signal['direction']}")
                print(f"Entry     : ₹{levels['entry']:.2f}")
                print(f"TP1       : ₹{levels['tp1']:.2f}")
                print(f"TP2       : ₹{levels['tp2']:.2f}")
                print(f"SL        : ₹{levels['sl']:.2f}")
                print(f"Score     : {signal['score']}/15")
                print(f"Mode      : {mode.upper()}")

                # Telegram
                self.telegram.send_signal_alert(signal, levels, pair_data["data"])

                # Track
                self.tracker.add_signal(signal, levels)

            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {e}")

    # --------------------------------------------------
    def run_scan_cycle(self):
        try:
            print("\n" + "=" * 60)
            print(f"🔄 Scan Cycle @ {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)

            # Daily reset
            self._daily_reset_if_needed()

            # Market health
            print("\n📊 Checking Market Health...")
            health_score = self.scanner.calculate_market_health()
            if health_score < 4:
                print("⚠️ Market health low — skipping cycle")
                return

            # Scan pairs
            tradeable_pairs = self.scanner.scan_all_pairs()
            if not tradeable_pairs:
                print("⚠️ No tradeable pairs found")
                return

            processed_symbols = set()

            # Modes (order matters)
            for mode in ["quick", "mid", "trend"]:
                if not self.check_daily_limits():
                    break
                self.process_signal_mode(mode, tradeable_pairs, processed_symbols)

            print(f"\n✅ Cycle done | Signals today: {self.signals_today}/{MAX_DAILY_SIGNALS}")

        except Exception as e:
            print(f"❌ Scan cycle error: {e}")

    # --------------------------------------------------
    def start(self):
        print("\n" + "=" * 60)
        print("🤖 COINDCX FUTURES TRADING BOT (FIXED)")
        print("=" * 60)
        print(f"🎯 Target: ₹{TARGET_DAILY_PROFIT}/day")
        print(f"📊 Signals: {MIN_DAILY_SIGNALS}-{MAX_DAILY_SIGNALS}/day")
        print(f"⚡ Leverage: {LEVERAGE}x")
        print("=" * 60)

        self.telegram.send_startup_message()
        print("\n🚀 Bot running...\n")

        try:
            while self.is_running:
                self.run_scan_cycle()
                self.send_hourly_report()
                print(f"\n⏳ Waiting {SCAN_INTERVAL}s...")
                time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            self.shutdown()

    # --------------------------------------------------
    def shutdown(self):
        print("\n🛑 Shutting down...")
        stats = self.tracker.get_daily_summary()
        self.telegram.send_performance_report(stats)
        self.tracker.save_to_file()
        print("✅ Shutdown complete")
        sys.exit(0)


# --------------------------------------------------
if __name__ == "__main__":
    bot = CoinDCXTradingBot()
    bot.start()