"""
COINDCX BALANCED EDGE BOT – FINAL VERSION
Signal Only | Manual Trading
Orderbook OFF | Stable for 24x7 deployment
Includes HTTP Ping Server to prevent container stop
"""

import asyncio
import logging
import threading
import os
from datetime import datetime, time
from typing import Dict, List, Optional
from collections import deque
from dotenv import load_dotenv
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

from helpers import CoinDCXAPI, TelegramNotifier, DatabaseManager

# =====================================================
# LOAD ENV + LOGGING
# =====================================================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("coindcx_final.log")
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# CONFIG (LOCKED – DO NOT CHANGE)
# =====================================================
class Config:
    COINDCX_API_KEY = os.getenv("COINDCX_API_KEY")
    COINDCX_SECRET = os.getenv("COINDCX_SECRET")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    COINS = ["BTC", "ETH", "MATIC", "USDT"]

    SCAN_INTERVAL = 30
    PRICE_HISTORY = 40
    MIN_DATA_POINTS = 20

    DAILY_SIGNAL_LIMIT = 15
    COOLDOWN_MINUTES = 30
    MIN_SCORE = 58

    # Fixed TP / SL range
    TP_MIN = 0.018
    TP_MAX = 0.024
    SL_MIN = 0.009
    SL_MAX = 0.012

    # BTC calm check (~5m proxy)
    BTC_RANGE_LIMIT = 0.0035  # 0.35%

    # Safe trading windows (IST)
    TIME_WINDOWS = [
        (time(9, 15), time(12, 0)),
        (time(20, 0), time(23, 0)),
    ]


config = Config()

# =====================================================
# HEALTH / PING SERVER (REQUIRED FOR CONTAINER)
# =====================================================
def start_health_server():
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, format, *args):
            return

    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


# =====================================================
# DATA TRACKER (NO ORDERBOOK)
# =====================================================
class Tracker:
    def __init__(self, size: int):
        self.prices = {}
        self.volumes = {}
        self.size = size

    def add(self, market: str, price: float, volume: float):
        if market not in self.prices:
            self.prices[market] = deque(maxlen=self.size)
            self.volumes[market] = deque(maxlen=self.size)

        self.prices[market].append(price)
        self.volumes[market].append(volume)

    def ready(self, market: str) -> bool:
        return market in self.prices and len(self.prices[market]) >= config.MIN_DATA_POINTS


# =====================================================
# FINAL BOT
# =====================================================
class CoinDCXFinalBot:

    def __init__(self):
        self.api = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.tg = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager("coindcx_final.db")

        self.tracker = Tracker(config.PRICE_HISTORY)

        self.cooldown = {}
        self.daily_count = 0
        self.today = datetime.now().date()

        logger.info("✅ COINDCX FINAL BOT INITIALIZED")

    # ---------------- TIME WINDOW ----------------
    def in_time_window(self) -> bool:
        now = datetime.now().time()
        for start, end in config.TIME_WINDOWS:
            if start <= now <= end:
                return True
        return False

    # ---------------- BTC CALM CHECK ----------------
    def btc_calm(self) -> bool:
        prices = self.tracker.prices.get("BTCINR", [])
        if len(prices) < 10:
            return False
        rng = (max(prices[-10:]) - min(prices[-10:])) / prices[-1]
        return rng < config.BTC_RANGE_LIMIT

    # ---------------- COOLDOWN ----------------
    def in_cooldown(self, market: str) -> bool:
        last = self.cooldown.get(market)
        if not last:
            return False
        return (datetime.now() - last).seconds < config.COOLDOWN_MINUTES * 60

    # ---------------- UPDATE DATA ----------------
    async def update_data(self, markets: List[str]):
        tickers = await self.api.get_tickers()
        for m in markets:
            t = next((x for x in tickers if x.get("market") == m), None)
            if not t:
                continue
            price = float(t["last_price"])
            volume = float(t["volume"])
            self.tracker.add(m, price, volume)

    # ---------------- ANALYZE ----------------
    def analyze(self, market: str) -> Optional[Dict]:
        if not self.tracker.ready(market):
            return None
        if self.in_cooldown(market):
            return None

        prices = self.tracker.prices[market]
        volumes = self.tracker.volumes[market]

        # Momentum + volume agreement
        change = (prices[-1] - prices[-6]) / prices[-6]
        vol_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-15:-5])

        score = 0
        if abs(change) > 0.008:
            score += 30
        if vol_ratio > 1.2:
            score += 30

        if score < config.MIN_SCORE:
            return None

        side = "BUY" if change > 0 else "SELL"
        entry = prices[-1]

        tp_pct = np.random.uniform(config.TP_MIN, config.TP_MAX)
        sl_pct = np.random.uniform(config.SL_MIN, config.SL_MAX)

        if side == "BUY":
            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)
        else:
            tp = entry * (1 - tp_pct)
            sl = entry * (1 + sl_pct)

        return {
            "market": market,
            "side": side,
            "entry": round(entry, 2),
            "tp": round(tp, 2),
            "sl": round(sl, 2),
            "score": score
        }

    # ---------------- SEND SIGNAL ----------------
    async def send(self, signal: Dict):
        msg = f"""
🚀 *COINDCX CONFIRMED SIGNAL*

🪙 Pair: {signal['market']}
📊 Side: *{signal['side']}*

💰 Entry: ₹{signal['entry']}
🎯 TP: ₹{signal['tp']}
🛑 SL: ₹{signal['sl']}

🧠 Score: {signal['score']}%
₿ BTC: Calm
⏳ Cooldown: OK

⏰ {datetime.now().strftime('%d-%b %I:%M %p')}
"""
        await self.tg.send_message(msg)

    # ---------------- RUN LOOP ----------------
    async def run(self):
        await self.tg.send_message(
            "🚀 *COINDCX FINAL BOT STARTED*\n\n"
            "🔒 Orderbook OFF | Stable Mode\n"
            "🫀 Health Ping Enabled\n"
            "⏳ System Locked – 30 Days"
        )

        while True:
            if datetime.now().date() != self.today:
                self.today = datetime.now().date()
                self.daily_count = 0

            if not self.in_time_window():
                await asyncio.sleep(60)
                continue

            if self.daily_count >= config.DAILY_SIGNAL_LIMIT:
                await asyncio.sleep(300)
                continue

            markets = [f"{c}INR" for c in config.COINS]
            await self.update_data(markets)

            if not self.btc_calm():
                await asyncio.sleep(60)
                continue

            for m in markets:
                sig = self.analyze(m)
                if sig:
                    await self.send(sig)
                    self.db.save_signal(sig)
                    self.cooldown[m] = datetime.now()
                    self.daily_count += 1

            await asyncio.sleep(config.SCAN_INTERVAL)


# =====================================================
# START
# =====================================================
if __name__ == "__main__":
    # Start health server (PORT ping)
    threading.Thread(target=start_health_server, daemon=True).start()

    # Start trading bot
    bot = CoinDCXFinalBot()
    asyncio.run(bot.run())