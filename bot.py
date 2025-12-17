import requests
import time
import math
from datetime import datetime, timedelta

from config import Config
from signal_logic import SignalGenerator


class TradingBot:

    def __init__(self):
        self.config = Config()
        self.generator = SignalGenerator()

        self.last_candle_time = {}     # same-candle skip
        self.last_signal_time = {}     # direction-based cooldown

    # ======================================================
    # DATA FETCH
    # ======================================================
    def fetch_candles(self, market, interval, limit=120):
        try:
            url = f"{self.config.COINDCX_BASE_URL}/market_data/candles"
            params = {
                "pair": market,
                "interval": interval,
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
            print(f"❌ Candle fetch error {market} {interval}: {e}")
            return None

    # ======================================================
    # BTC CALM FILTER
    # ======================================================
    def btc_is_calm(self):
        btc = self.fetch_candles("B-BTC_USDT", "5m", 20)
        if not btc or len(btc) < 5:
            return False

        last = btc[-1]
        range_pct = (last["high"] - last["low"]) / last["close"] * 100

        return range_pct < 0.6   # BTC calm threshold

    # ======================================================
    # TELEGRAM
    # ======================================================
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
                f"⚠️ Use proper risk management"
            )

            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": self.config.TELEGRAM_CHAT_ID,
                "text": msg
            }

            res = requests.post(url, json=payload, timeout=10)
            res.raise_for_status()

            print(f"✅ Telegram sent | {signal['market']}")

        except Exception as e:
            print(f"❌ Telegram error: {e}")

    # ======================================================
    # CORE SCAN
    # ======================================================
    def scan_market(self, market):
        candles_5m = self.fetch_candles(market, "5m")
        candles_15m = self.fetch_candles(market, "15m")

        if not candles_5m or not candles_15m:
            return None

        # same candle skip
        latest_time = candles_5m[-1]["time"]
        if self.last_candle_time.get(market) == latest_time:
            return None
        self.last_candle_time[market] = latest_time

        signal = self.generator.generate_signal(
            market,
            candles_5m,
            candles_15m
        )

        if not signal:
            return None

        # direction-based cooldown
        last = self.last_signal_time.get(market)
        if last:
            if last["direction"] == signal["direction"]:
                if datetime.now() - last["time"] < timedelta(minutes=30):
                    return None

        self.last_signal_time[market] = {
            "direction": signal["direction"],
            "time": datetime.now()
        }

        return signal

    # ======================================================
    # DYNAMIC SLEEP (NEXT 5m CANDLE)
    # ======================================================
    def next_candle_sleep(self):
        now = time.time()
        next_5m = math.ceil(now / 300) * 300
        return max(5, next_5m - now)

    # ======================================================
    # MAIN LOOP
    # ======================================================
    def run(self):
        print("🤖 COINDCX BOT STARTED")
        print(f"Interval: 5m | Scoring + MTF + BTC Calm\n")

        while True:
            try:
                if not self.btc_is_calm():
                    print("⚠️ BTC volatile → skipping this round")
                else:
                    for market in self.config.MARKETS:
                        sig = self.scan_market(market)
                        if sig:
                            print(
                                f"🎯 {sig['market']} "
                                f"{sig['direction']} | "
                                f"Score {sig['score']}"
                            )
                            self.send_telegram(sig)

                sleep_sec = self.next_candle_sleep()
                print(f"💤 Sleeping {int(sleep_sec)} sec\n")
                time.sleep(sleep_sec)

            except KeyboardInterrupt:
                print("🛑 Bot stopped by user")
                break

            except Exception as e:
                print(f"❌ Runtime error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    TradingBot().run()