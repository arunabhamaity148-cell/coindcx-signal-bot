import requests
import time
from datetime import datetime, timedelta
from config import Config
from signal_logic import SignalGenerator

class TradingBot:

    def __init__(self):
        self.config = Config()
        self.signal_generator = SignalGenerator()
        self.last_signal_time = {}
        self.last_candle_time = {}     # ✅ NEW
        self.signals_sent_today = 0
        self.last_reset_date = datetime.now().date()

    def fetch_candles(self, market, interval='5m', limit=100):
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
            print(f"❌ Error fetching candles for {market}: {str(e)}")
            return None

    def check_cooldown(self, market):
        if market not in self.last_signal_time:
            return True
        return datetime.now() - self.last_signal_time[market] >= timedelta(
            minutes=self.config.COOLDOWN_MINUTES
        )

    def check_daily_limit(self):
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.signals_sent_today = 0
            self.last_reset_date = today
        return self.signals_sent_today < self.config.MAX_SIGNALS_PER_DAY

    def scan_market(self, market):
        print(f"📊 Scanning {market}...")

        if not self.check_cooldown(market):
            print("⏳ Cooldown active")
            return None

        if not self.check_daily_limit():
            print("⚠️ Daily limit reached")
            return None

        candles = self.fetch_candles(market, self.config.CANDLE_INTERVAL)
        if not candles:
            return None

        # ✅ SAME CANDLE SKIP LOGIC
        latest_time = candles[-1]['time']
        if self.last_candle_time.get(market) == latest_time:
            print("⏭️ Same candle, skipping")
            return None
        self.last_candle_time[market] = latest_time

        signal = self.signal_generator.generate_signal(market, candles)
        if signal:
            print(f"🎯 Signal found | Score: {signal['score']}")
            return signal

        print("⏭️ No signal")
        return None

    def scan_all_markets(self):
        print("\n" + "="*60)
        print(f"🔍 SCAN START | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        signals = []
        for market in self.config.MARKETS:
            sig = self.scan_market(market)
            if sig:
                signals.append(sig)
            time.sleep(1)
        return signals

    def process_signals(self, signals):
        if not signals:
            print("📭 No signals this round\n")
            return

        for signal in signals:
            self.last_signal_time[signal['market']] = datetime.now()
            self.signals_sent_today += 1
            print(f"✅ Signal processed | {signal['market']}")

    def run(self):
        print("\n🤖 COINDCX BOT STARTED")
        print(f"Interval: {self.config.CANDLE_INTERVAL}")
        print(f"Check every: {self.config.CHECK_INTERVAL_MINUTES} min\n")

        while True:
            try:
                signals = self.scan_all_markets()
                self.process_signals(signals)

                next_scan = datetime.now() + timedelta(
                    minutes=self.config.CHECK_INTERVAL_MINUTES
                )
                print(f"⏰ Next scan: {next_scan.strftime('%H:%M:%S')}")
                print(f"💤 Sleeping {self.config.CHECK_INTERVAL_MINUTES} minutes...\n")

                time.sleep(self.config.CHECK_INTERVAL_MINUTES * 60)

            except KeyboardInterrupt:
                print("🛑 Bot stopped")
                break

            except Exception as e:
                print(f"❌ Error: {str(e)} | Retry in 5 min")
                time.sleep(300)

if __name__ == "__main__":
    TradingBot().run()