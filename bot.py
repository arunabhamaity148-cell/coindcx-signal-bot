import requests
import time
import math
from datetime import datetime, timedelta

from config import Config
from signal_logic import SignalGenerator


class TradingBot:

    def __init__(self):
        self.cfg = Config()
        self.gen = SignalGenerator()

        self.last_candle = {}
        self.last_signal = {}

    # ==================================================
    # DATA FETCH
    # ==================================================
    def fetch(self, market, tf, limit=120):
        try:
            url = f"{self.cfg.COINDCX_BASE_URL}/market_data/candles"
            params = {
                "pair": market,
                "interval": tf,
                "limit": limit
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()

            data = r.json()
            if not data:
                return None

            return [{
                "time": c["time"],
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"])
            } for c in data]

        except Exception as e:
            print(f"❌ Fetch error {market} {tf}: {e}")
            return None

    # ==================================================
    # BTC CALM FILTER
    # ==================================================
    def btc_calm(self):
        btc = self.fetch("B-BTC_USDT", "5m", 20)
        if not btc or len(btc) < 5:
            return False

        c = btc[-1]
        range_pct = (c["high"] - c["low"]) / c["close"] * 100
        return range_pct < 0.6

    # ==================================================
    # TELEGRAM
    # ==================================================
    def send_telegram(self, signal):
        try:
            emoji = "🟢" if signal["direction"] == "LONG" else "🔴"

            msg = (
                f"{emoji*3} {signal['direction']} SIGNAL {emoji*3}\n\n"
                f"💹 Market: {signal['market']}\n"
                f"📊 Score: {signal['score']}\n"
                f"⏱️ TF: 5m → 15m (MTF Confirmed)\n\n"
                f"💰 Entry: {signal['entry']:.6f}\n"
                f"🎯 TP: {signal['tp']:.6f}\n"
                f"🛑 SL: {signal['sl']:.6f}\n\n"
                f"📈 R:R ≈ {signal['rr']:.2f}\n\n"
                f"✅ Reasons:\n"
            )

            for r in signal["reasons"][:4]:
                msg += f"• {r}\n"

            msg += (
                f"\n🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"⚠️ Always use proper risk management"
            )

            url = f"https://api.telegram.org/bot{self.cfg.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": self.cfg.TELEGRAM_CHAT_ID,
                "text": msg
            }

            res = requests.post(url, json=payload, timeout=10)
            res.raise_for_status()

            print(f"✅ Telegram sent | {signal['market']}")

        except Exception as e:
            print(f"❌ Telegram error: {e}")

    # ==================================================
    # DYNAMIC SLEEP (NEXT 5m CANDLE)
    # ==================================================
    def sleep_next_5m(self):
        return max(5, math.ceil(time.time() / 300) * 300 - time.time())

    # ==================================================
    # MAIN LOOP
    # ==================================================
    def run(self):
        print("🤖 CLEAN SCORING + MTF BOT (Telegram Enabled)")

        while True:
            try:
                if not self.btc_calm():
                    print("⚠️ BTC volatile → skipping this round")
                else:
                    for m in self.cfg.MARKETS:
                        c5 = self.fetch(m, "5m")
                        c15 = self.fetch(m, "15m")

                        if not c5 or not c15:
                            continue

                        # same candle skip
                        if self.last_candle.get(m) == c5[-1]["time"]:
                            continue
                        self.last_candle[m] = c5[-1]["time"]

                        signal = self.gen.generate_signal(m, c5, c15)
                        if not signal:
                            continue

                        # direction-based cooldown
                        last = self.last_signal.get(m)
                        if last and last["direction"] == signal["direction"]:
                            if datetime.now() - last["time"] < timedelta(minutes=30):
                                continue

                        self.last_signal[m] = {
                            "direction": signal["direction"],
                            "time": datetime.now()
                        }

                        print(
                            f"🎯 {signal['market']} "
                            f"{signal['direction']} "
                            f"Score {signal['score']}"
                        )

                        self.send_telegram(signal)

                time.sleep(self.sleep_next_5m())

            except KeyboardInterrupt:
                print("🛑 Bot stopped by user")
                break

            except Exception as e:
                print(f"❌ Runtime error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    TradingBot().run()