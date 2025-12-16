"""
UNIQUE PROPRIETARY TRADING BOT
Nobody else has this logic combination
Smart, Different, Profitable
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict
import os
from dotenv import load_dotenv

from helpers import CoinDCXAPI, TelegramNotifier, DatabaseManager
from logic import UniqueSignalGenerator

load_dotenv()

class Config:
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    COINS_TO_MONITOR = [
        'BTC','ETH','BNB','SOL','XRP','ADA','DOGE','MATIC',
        'DOT','AVAX','LINK','UNI','ATOM','LTC','ETC',
        'NEAR','FTM','SAND','MANA','AXS','APE','OP','ARB',
        'SUI','TRX','AAVE','GRT','ALGO','VET','ICP',
        'FIL','THETA','XLM','EOS','CHZ','ENJ','ROSE','LRC',
        'IMX','GMT','GAL','ONE','HBAR','EGLD','ZIL','WAVES'
    ]

    TIMEFRAMES = ['5m', '15m', '1h']
    MIN_SCORE = 45
    SCAN_INTERVAL = 75
    DB_PATH = 'unique_signals.db'

config = Config()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UniqueTradingBot:

    def __init__(self):
        self.dcx = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager(config.DB_PATH)
        self.signal_gen = UniqueSignalGenerator()

        self.processed = set()
        self.daily_signals = 0
        self.last_date = datetime.now().date()

        logger.info("🚀 UNIQUE Bot Initialized")

    async def get_markets(self) -> List[str]:
        """FINAL & CORRECT CoinDCX INR market filter"""
        try:
            markets = await self.dcx.get_markets()

            if not isinstance(markets, list):
                return []

            inr_markets = []

            for m in markets:
                if not isinstance(m, dict):
                    continue

                symbol = m.get('symbol', '')
                if not symbol:
                    continue

                symbol_upper = symbol.upper()

                # 🔥 FINAL WORKING LOGIC
                if 'INR' in symbol_upper:
                    for coin in config.COINS_TO_MONITOR:
                        if symbol_upper.startswith(coin) or symbol_upper.startswith(f'I-{coin}') or symbol_upper.startswith(f'B-{coin}'):
                            inr_markets.append(symbol)
                            break

            logger.info(f"📊 {len(inr_markets)} markets loaded")
            return inr_markets

        except Exception as e:
            logger.error(f"Market load error: {e}")
            return []

    async def analyze_market(self, market: str, timeframe: str):
        try:
            candles = await self.dcx.get_candles(market, timeframe, limit=200)
            if len(candles) < 100:
                return None

            orderbook = await self.dcx.get_orderbook(market)
            ticker = await self.dcx.get_ticker(market)
            price = float(ticker.get('last_price', 0)) if ticker else 0

            if price == 0:
                return None

            return await self.signal_gen.generate_signal(
                market=market,
                candles=candles,
                orderbook=orderbook,
                timeframe=timeframe,
                current_price_inr=price
            )
        except:
            return None

    async def scan_all(self):
        logger.info("🔍 Scanning with UNIQUE logic...")

        markets = await self.get_markets()
        if not markets:
            logger.warning("⚠️ No markets found, retrying next scan...")
            return

        for market in markets:
            for tf in config.TIMEFRAMES:
                signal = await self.analyze_market(market, tf)
                if signal:
                    await self.telegram.send_message(str(signal))
                await asyncio.sleep(0.2)

    async def run(self):
        await self.telegram.send_message("🚀 UNIQUE Bot Started")
        while True:
            await self.scan_all()
            await asyncio.sleep(config.SCAN_INTERVAL)

async def main():
    bot = UniqueTradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
