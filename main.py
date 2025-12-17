"""
COINDCX BALANCED EDGE BOT – FINAL (1 MONTH LOCK)
Signal-only | Manual Trading
No auto-execution | No overtrading
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from collections import deque
import numpy as np

from helpers import CoinDCXAPI, TelegramNotifier, DatabaseManager

# =====================================================
# ENV + LOGGING
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
# CONFIG (LOCKED)
# =====================================================
class Config:
    COINDCX_API_KEY = os.getenv("COINDCX_API_KEY")
    COINDCX_SECRET = os.getenv("COINDCX_SECRET")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    COINS = ["BTC", "ETH", "SOL", "XRP", "MATIC", "DOGE", "ADA"]

    SCAN_INTERVAL = 30
    PRICE_HISTORY = 40
    MIN_DATA_POINTS = 20

    DAILY_SIGNAL_LIMIT = 15
    COOLDOWN_MINUTES = 30
    MIN_SCORE = 58

    # TP / SL (FIXED RANGE)
    TP_MIN = 0.018
    TP_MAX = 0.024
    SL_MIN = 0.009
    SL_MAX = 0.012

    # BTC CALM
    BTC_RANGE_5M = 0.0035  # 0.35%

    # SPREAD
    MAX_SPREAD = 0.35

    # TIME WINDOWS
    TIME_WINDOWS = [
        (time(9, 15), time(12, 0)),
        (time(20, 0), time(23, 0)),
    ]


config = Config()

# =====================================================
# DATA TRACKER
# =====================================================
class Tracker:
    def __init__(self, size: int):
        self.prices = {}
        self.volumes = {}
        self.orderbooks = {}
        self.spreads = {}
        self.size = size

    def add(self, m: str, price: float, vol: float, ob: dict):
        if m not in self.prices:
            self.prices[m] = deque(maxlen=self.size)
            self.volumes[m] = deque(maxlen=self.size)
            self.spreads[m] = deque(maxlen=self.size)

        self.prices[m].append(price)
        self.volumes[m].append(vol)
        self.orderbooks[m] = ob

        if ob and ob.get("bids") and ob.get("asks"):
            bid = ob["bids"][0][0]
            ask = ob["asks"][0][0]
            if bid > 0:
                self.spreads[m].append(((ask - bid) / bid) * 100)

    def ready(self, m: str) -> bool:
        return m in self.prices and len(self.prices[m]) >= config.MIN_DATA_POINTS


# =====================================================
# CORE BOT
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

        logger.info("✅ FINAL COINDCX BOT INITIALIZED")

    # ---------------- TIME CHECK ----------------
    def in_time_window(self) -> bool:
        now = datetime.now().time()
        for start, end in config.TIME_WINDOWS:
            if start <= now <= end:
                return True
        return False

    # ---------------- BTC CALM ----------------
    def btc_calm(self) -> bool:
        prices = self.tracker.prices.get("BTCINR", [])
        if len(prices) < 10:
            return False
        rng = (max(prices[-10:]) - min(prices[-10:])) / prices[-1]
        return rng < config.BTC_RANGE_5M

    # ---------------- COOLDOWN ----------------
    def in_cooldown(self, m: str) -> bool:
        t = self.cooldown.get(m)
        if not t:
            return False
        return (datetime.now() - t).seconds < config.COOLDOWN_MINUTES * 60

    # ---------------- UPDATE DATA ----------------
    async def update_data(self, markets: List[str]):
        tickers = await self.api.get_tickers()
        for m in markets:
            t = next((x for x in tickers if x.get("market") == m), None)
            if not t:
                continue
            price = float(t["last_price"])
            vol = float(t["volume"])
            ob = await self.api.get_orderbook(m)
            self.tracker.add(m, price, vol, ob)

    # ---------------- ANALYZE ----------------
    def analyze(self, m: str) -> Optional[Dict]:
        if not self.tracker.ready(m):
            return None
        if self.in_cooldown(m):
            return None

        prices = self.tracker.prices[m]
        vols = self.tracker.volumes[m]
        spreads = self.tracker.spreads[m]

        if spreads and spreads[-1] > config.MAX_SPREAD:
            return None

        change = (prices[-1] - prices[-6]) / prices[-6]
        vol_ratio = np.mean(vols[-5:]) / np.mean(vols[-15:-5])

        score = 0
        if vol_ratio > 1.2:
            score += 25
        if abs(change) > 0.008:
            score += 25
        if spreads and spreads[-1] < 0.2:
            score += 15

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
            "market": m,
            "side": side,
            "entry": round(entry, 2),
            "tp": round(tp, 2),
            "sl": round(sl, 2),
            "score": score
        }

    # ---------------- SEND ----------------
    async def send(self, s: Dict):
        msg = f"""
🚀 *COINDCX CONFIRMED SIGNAL*

🪙 Pair: {s['market']}
📊 Side: *{s['side']}*

💰 Entry: ₹{s['entry']}
🎯 TP: ₹{s['tp']}
🛑 SL: ₹{s['sl']}

🧠 Score: {s['score']}%
₿ BTC: Calm
⏳ Cooldown: OK

⏰ {datetime.now().strftime('%d-%b %I:%M %p')}
"""
        await self.tg.send_message(msg)

    # ---------------- RUN ----------------
    async def run(self):
        await self.tg.send_message(
            "🚀 *COINDCX FINAL BOT STARTED*\n\n"
            "🔒 System Locked – 30 Days\n"
            "📊 Balanced Edge Mode"
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

            markets = await self.api.get_inr_markets(config.COINS)
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
    bot = CoinDCXFinalBot()
    asyncio.run(bot.run())