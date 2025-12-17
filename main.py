"""
FINAL CoinDCX-ONLY SMART Signal Bot
----------------------------------
• Tuned for CoinDCX INR markets
• 10 Smart Indicators (Relaxed)
• Target: 5–12 QUALITY signals/day
• Manual trading friendly
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
from collections import deque
import numpy as np

from helpers import CoinDCXAPI, TelegramNotifier, DatabaseManager

# ======================================================
# ENV & LOGGING
# ======================================================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================================
# CONFIG (CoinDCX Tuned)
# ======================================================
class Config:
    COINDCX_API_KEY = os.getenv("COINDCX_API_KEY")
    COINDCX_SECRET = os.getenv("COINDCX_SECRET")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    COINS_TO_MONITOR = [
        "BTC","ETH","SOL","XRP","BNB","ADA","DOGE","MATIC",
        "DOT","AVAX","LINK","UNI","ATOM","LTC","NEAR","TRX"
    ]

    MIN_SCORE = 42
    SCAN_INTERVAL = 45
    PRICE_HISTORY_SIZE = 50
    MIN_DATA_POINTS = 18
    PAIR_COOLDOWN_SEC = 45 * 60   # 45 minutes

config = Config()

# ======================================================
# PRICE TRACKER
# ======================================================
class SmartPriceTracker:
    def __init__(self, max_size=50):
        self.data = {}
        self.last_update = {}
        self.max_size = max_size

    def add_price(self, market, price, volume):
        if market not in self.data:
            self.data[market] = deque(maxlen=self.max_size)

        # bad tick filter
        if self.data[market]:
            last_price = self.data[market][-1]["price"]
            if abs(price - last_price) / last_price > 0.15:
                return

        self.data[market].append({
            "price": price,
            "volume": volume,
            "time": datetime.now()
        })
        self.last_update[market] = datetime.now()

    def has_enough_data(self, market):
        if market not in self.data:
            return False
        if len(self.data[market]) < config.MIN_DATA_POINTS:
            return False
        if (datetime.now() - self.last_update.get(market)).seconds > 180:
            return False
        return True

    def get_history(self, market):
        return list(self.data.get(market, []))

# ======================================================
# ANALYZER (CoinDCX Relaxed)
# ======================================================
class Analyzer:

    @staticmethod
    def momentum(history):
        prices = [h["price"] for h in history]
        velocity = (prices[-1] - prices[-5]) / prices[-5]
        if velocity > 0.004:
            return "UP", 8
        if velocity < -0.004:
            return "DOWN", 8
        return "FLAT", 0

    @staticmethod
    def smart_money(history):
        vols = [h["volume"] for h in history]
        prices = [h["price"] for h in history]
        vr = np.mean(vols[-5:]) / max(np.mean(vols[-15:]), 1)
        pc = (prices[-1] - prices[-5]) / prices[-5]
        if vr > 1.2 and pc > 0:
            return "SMART_BUY", 8
        if vr > 1.2 and pc < 0:
            return "SMART_SELL", 8
        return "NEUTRAL", 0

    @staticmethod
    def volatility(history):
        prices = [h["price"] for h in history[-20:]]
        rets = [(prices[i]-prices[i-1])/prices[i-1] for i in range(1,len(prices))]
        v = np.std(rets)
        if 0.002 <= v <= 0.04:
            return v, 8
        return v, 0

    @staticmethod
    def trend(history):
        prices = [h["price"] for h in history[-10:]]
        ups = sum(prices[i] > prices[i-1] for i in range(1,len(prices)))
        if ups >= 6:
            return "UP", 6
        if ups <= 3:
            return "DOWN", 6
        return "NONE", 0

    @staticmethod
    def sentiment(history):
        prices = [h["price"] for h in history[-8:]]
        bulls = sum(prices[i] > prices[i-1] for i in range(1,len(prices)))
        if bulls >= 5:
            return "BULLISH", 5
        if bulls <= 2:
            return "BEARISH", 5
        return "NEUTRAL", 0

# ======================================================
# MAIN BOT
# ======================================================
class CoinDCXSmartBot:

    def __init__(self):
        self.api = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.tg = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager("signals.db")
        self.tracker = SmartPriceTracker()
        self.analyzer = Analyzer()

        self.last_signal_time = {}
        self.today = datetime.now().date()
        self.daily_count = 0

        logger.info("✅ CoinDCX SMART Bot Ready")

    async def update_prices(self, markets):
        tickers = await self.api.get_tickers()
        for t in tickers:
            m = t.get("market")
            if m in markets:
                price = float(t.get("last_price",0))
                vol = float(t.get("volume",0))
                if price > 0:
                    self.tracker.add_price(m, price, vol)

    def analyze(self, market):
        if not self.tracker.has_enough_data(market):
            return None

        last = self.last_signal_time.get(market)
        if last and (datetime.now() - last).seconds < config.PAIR_COOLDOWN_SEC:
            return None

        h = self.tracker.get_history(market)

        score = 0
        bull = 0
        bear = 0

        m, s = self.analyzer.momentum(h); score+=s
        if m=="UP": bull+=2
        if m=="DOWN": bear+=2

        sm, s = self.analyzer.smart_money(h); score+=s
        if "BUY" in sm: bull+=2
        if "SELL" in sm: bear+=2

        _, s = self.analyzer.volatility(h); score+=s
        t, s = self.analyzer.trend(h); score+=s
        if t=="UP": bull+=1
        if t=="DOWN": bear+=1

        sen, s = self.analyzer.sentiment(h); score+=s
        if sen=="BULLISH": bull+=1
        if sen=="BEARISH": bear+=1

        final_score = int((score / 40) * 100)
        if final_score < config.MIN_SCORE:
            logger.info(f"REJECT {market} score={final_score}")
            return None

        if abs(bull - bear) < 1:
            return None

        side = "BUY" if bull > bear else "SELL"

        price = h[-1]["price"]
        atr = np.mean([abs(h[i]["price"]-h[i-1]["price"]) for i in range(1,len(h))][-15:])

        if side=="BUY":
            sl = price - atr*1.6
            tp = price + atr*2.5
        else:
            sl = price + atr*1.6
            tp = price - atr*2.5

        return {
            "market": market,
            "side": side,
            "entry": round(price,2),
            "sl": round(sl,2),
            "tp": round(tp,2),
            "score": final_score
        }

    async def run(self):
        await self.tg.send_message("✅ CoinDCX SMART Bot Started")
        while True:
            markets = await self.api.get_inr_markets(config.COINS_TO_MONITOR)
            await self.update_prices(markets)

            for m in markets:
                sig = self.analyze(m)
                if sig:
                    self.last_signal_time[m] = datetime.now()
                    self.daily_count += 1
                    await self.tg.send_signal(sig)
                    self.db.save_signal(sig)
                    logger.info(f"SIGNAL {sig['market']} {sig['side']}")

            await asyncio.sleep(config.SCAN_INTERVAL)

# ======================================================
# START
# ======================================================
async def main():
    bot = CoinDCXSmartBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())