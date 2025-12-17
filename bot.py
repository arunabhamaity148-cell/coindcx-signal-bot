import requests
import time
from datetime import datetime, timedelta
from config import Config
from signal_logic import SignalGenerator
import math

class TradingBot:

    def __init__(self):
        self.config = Config()
        self.generator = SignalGenerator()
        self.last_candle = {}
        self.last_signal = {}

    # ===================== DATA =====================
    def fetch(self, market, interval):
        url = f"{self.config.COINDCX_BASE_URL}/market_data/candles"
        params = {"pair": market, "interval": interval, "limit": 120}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        return [{
            "time": c["time"],
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"])
        } for c in r.json()]

    # ===================== FILTERS =====================
    def btc_calm(self):
        btc = self.fetch("B-BTC_USDT", "5m")
        if not btc or len(btc) < 10:
            return False
        last = btc[-1]
        rng = (last["high"] - last["low"]) / last["close"] * 100
        return rng < 0.6   # BTC calm threshold

    def next_candle_sleep(self):
        now = time.time()
        next_5m = math.ceil(now / 300) * 300
        return max(5, next_5m - now)

    # ===================== TELEGRAM =====================
    def send_telegram(self, signal):
        try:
            emoji = "🟢" if signal["direction"] == "LONG" else "🔴"

            msg = (
                f"{emoji*3} {signal['direction']} SIGNAL {emoji*3}\n\n"
                f"💹 Market: {signal['market']}\n"
                f"🎯 Mode: Institutional (One-TP)\n\n"
                f"💰 Entry: {signal['entry']:.6f}\n"
                f"🎯 TP: {signal['tp']:.6f}\n"
                f"🛑 SL: {signal['sl']:.6f}\n\n"
                f"📈 R:R ≈ {signal['rr']:.2f}\n"
                f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"⚠️ Use proper risk management"
            )

            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": self.config.TELEGRAM_CHAT_ID,
                "text": msg
            }

            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()
            print(f"✅ Telegram sent | {signal['market']}")

        except Exception as e:
            print(f"❌ Telegram error: {e}")

    # ===================== CORE =====================
    def scan(self, market):
        c5 = self.fetch(market, "5m")
        c15 = self.fetch(market, "15m")
        if not c5 or not c15:
            return None

        if self.last_candle.get(market) == c5[-1]["time"]:
            return None
        self.last_candle[market] = c5[-1]["time"]

        signal = self.generator.generate_signal(market, c5, c15)
        if not signal:
            return None

        last = self.last_signal.get(market)
        if last and last["direction"] == signal["direction"]:
            if datetime.now() - last["time"] < timedelta(minutes=30):
                return None

        self.last_signal[market] = {
            "direction": signal["direction"],
            "time": datetime.now()
        }

        return signal

    # ===================== RUN =====================
    def run(self):
        print("🤖 INSTITUTIONAL MODE STARTED (Telegram Enabled)")

        while True:
            try:
                if not self.btc_calm():
                    print("⚠️ BTC volatile — skipping all signals")
                else:
                    for m in self.config.MARKETS:
                        s = self.scan(m)
                        if s:
                            print(f"🎯 {s['market']} {s['direction']} | sending Telegram")
                            self.send_telegram(s)

                sleep_sec = self.next_candle_sleep()
                print(f"💤 Sleep {int(sleep_sec)} sec\n")
                time.sleep(sleep_sec)

            except Exception as e:
                print("❌ Runtime error:", e)
                time.sleep(60)

if __name__ == "__main__":
    TradingBot().run()