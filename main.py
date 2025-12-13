import time as sleep_time
from datetime import datetime
import sys

from config import *
from market_scanner import MarketScanner
from signal_engine import SignalEngine


class CoinDCXFuturesBot:
    def __init__(self):
        self.scanner = MarketScanner()
        self.engine = SignalEngine()
        self.signals_today = 0

        # 🔒 banner যেন একবারই আসে
        self.started = False

    def startup_banner(self):
        print("""
🤖 CoinDCX FUTURES BOT ACTIVATED 🤖

━━━━━━━━━━━━━━━━━━
⚙️ CONFIGURATION
━━━━━━━━━━━━━━━━━━
💰 Margin/Trade: ₹3,000
⚡ Leverage: 5x
🎯 Daily Target: ₹2,000
📊 Signal Range: 10-15/day

━━━━━━━━━━━━━━━━━━
🔍 WATCHLIST
━━━━━━━━━━━━━━━━━━
📈 Monitoring: 20 pairs

━━━━━━━━━━━━━━━━━━
🛡️ SAFETY FEATURES
━━━━━━━━━━━━━━━━━━
✅ 45-Logic Filter System
✅ Market Health Check
✅ Liquidation Protection
✅ Anti-Manipulation Filter
✅ Risk-Reward Validation

🚀 BOT IS NOW SCANNING... 🚀
""")

    def run_cycle(self):
        print("\n================================================")
        print(f"🔄 Scan Cycle @ {datetime.now().strftime('%H:%M:%S')}")
        print("================================================")

        health = self.scanner.calculate_market_health()
        if health < 4:
            print("⚠️ Market not healthy. Skipping cycle.")
            return

        pairs = self.scanner.scan_all_pairs()
        if not pairs:
            print("⚠️ No tradeable pairs found")
            return

        for mode in ["quick", "mid", "trend"]:
            for pair in pairs:
                signal = self.engine.generate_signal(pair, mode)
                if signal:
                    print(f"🎯 SIGNAL: {signal['symbol']} | {signal['direction']} | {mode.upper()}")
                    self.signals_today += 1

                    if self.signals_today >= MAX_DAILY_SIGNALS:
                        print("⚠️ Daily signal limit reached")
                        return

    def start(self):
        # 🔐 banner only once
        if not self.started:
            self.startup_banner()
            self.started = True

        while True:
            try:
                self.run_cycle()
                print(f"⏳ Waiting {SCAN_INTERVAL}s...")
                sleep_time.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("🛑 Bot stopped manually")
                sys.exit(0)

            except Exception as e:
                print(f"❌ ERROR: {e}")
                # 🔥 crash না করে wait
                sleep_time.sleep(10)


if __name__ == "__main__":
    bot = CoinDCXFuturesBot()
    bot.start()