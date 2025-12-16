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
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'MATIC',
        'DOT', 'AVAX', 'LINK', 'UNI', 'ATOM', 'LTC', 'ETC',
        'NEAR', 'FTM', 'SAND', 'MANA', 'AXS', 'APE', 'OP', 'ARB',
        'SUI', 'TRX', 'AAVE', 'GRT', 'ALGO', 'VET', 'ICP',
        'FIL', 'THETA', 'XLM', 'EOS', 'CHZ', 'ENJ', 'ROSE', 'LRC',
        'IMX', 'GMT', 'GAL', 'ONE', 'HBAR', 'EGLD', 'ZIL', 'WAVES'
    ]

    TIMEFRAMES = ['5m', '15m', '1h']
    MIN_SCORE = 45
    SCAN_INTERVAL = 75
    DB_PATH = 'unique_signals.db'

config = Config()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unique_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UniqueTradingBot:

    def __init__(self):
        if not config.COINDCX_API_KEY or not config.COINDCX_SECRET:
            raise ValueError("❌ CoinDCX API credentials missing")

        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            raise ValueError("❌ Telegram credentials missing")

        self.dcx = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager(config.DB_PATH)
        self.signal_gen = UniqueSignalGenerator()

        self.processed = set()
        self.daily_signals = 0
        self.last_date = datetime.now().date()

        logger.info("🚀 UNIQUE Bot Initialized")

    async def get_markets(self) -> List[str]:
        """Get CoinDCX markets (FINAL FIX)"""
        try:
            markets = await self.dcx.get_markets()

            if not markets or not isinstance(markets, list):
                logger.error("❌ Invalid markets response")
                return []

            inr_markets = []

            for m in markets:
                if not isinstance(m, dict):
                    continue

                # 🔥 FINAL FIX (fallback added)
                base = m.get('base_currency_short_name') or m.get('base_currency')
                target = m.get('target_currency_short_name') or m.get('target_currency')
                symbol = m.get('symbol')

                if base in config.COINS_TO_MONITOR and target == 'INR':
                    inr_markets.append(symbol)

            logger.info(f"📊 {len(inr_markets)} markets loaded")
            return inr_markets

        except Exception as e:
            logger.error(f"❌ Market load error: {e}", exc_info=True)
            return []

    async def analyze_market(self, market: str, timeframe: str):
        try:
            candles = await self.dcx.get_candles(market, timeframe, limit=200)
            if candles is None or len(candles) < 100:
                return None

            orderbook = await self.dcx.get_orderbook(market)
            ticker = await self.dcx.get_ticker(market)
            if not ticker:
                return None

            price = float(ticker.get('last_price', 0))
            if price == 0:
                return None

            return await self.signal_gen.generate_signal(
                market=market,
                candles=candles,
                orderbook=orderbook,
                timeframe=timeframe,
                current_price_inr=price
            )

        except Exception as e:
            logger.error(f"❌ Analysis error {market}: {e}")
            return None

    async def scan_all(self):
        today = datetime.now().date()
        if today != self.last_date:
            self.daily_signals = 0
            self.last_date = today

        logger.info("🔍 Scanning with UNIQUE logic...")

        markets = await self.get_markets()
        if not markets:
            logger.warning("⚠️ No markets found, retrying next scan...")
            return

        found = 0

        for market in markets:
            for tf in config.TIMEFRAMES:
                try:
                    signal = await self.analyze_market(market, tf)
                    if signal and signal['logic_score'] >= config.MIN_SCORE:
                        key = f"{market}_{tf}_{signal['side']}_{datetime.now().strftime('%Y%m%d%H')}"
                        if key not in self.processed:
                            await self.send_unique_signal(signal)
                            self.db.save_signal(signal)
                            self.processed.add(key)
                            self.daily_signals += 1
                            found += 1
                    await asyncio.sleep(0.2)
                except Exception as e:
                    logger.error(f"❌ {market} {tf}: {e}")

        logger.info(f"✅ Scan done. Found: {found} | Today: {self.daily_signals}")

    async def send_unique_signal(self, signal: Dict):
        details = signal.get('details', {})
        insights = []

        if 'ACCELERATING' in details.get('momentum_wave', ''):
            insights.append("⚡ Momentum Accelerating")
        if details.get('smart_money') in ['BULLISH', 'BEARISH']:
            insights.append(f"🐋 Smart Money: {details['smart_money']}")
        if details.get('near_vacuum'):
            insights.append("🚀 Near Liquidity Vacuum")

        insight_text = "\n".join([f"  • {i}" for i in insights[:3]]) if insights else "  • Standard setup"
        side_emoji = "📈" if signal['side'] == "BUY" else "📉"
        conf_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "⚠️"}[signal['confidence']]

        message = f"""🚨 *UNIQUE {signal['mode']} SIGNAL* 🚨

📌 *Pair:* {signal['market']}
📊 *TF:* {signal['timeframe']}
{side_emoji} *Side:* *{signal['side']}*

💰 *Entry:* ₹{signal['entry']:,.2f}
🛑 *SL:* ₹{signal['sl']:,.2f}
🎯 *TP:* ₹{signal['tp']:,.2f}

📐 *R:R:* 1:{signal['rr_ratio']:.1f}
🧠 *Score:* {signal['logic_score']}%
{conf_emoji} *Confidence:* {signal['confidence']}

🎨 *Unique Insights:*
{insight_text}

🕐 _{datetime.now().strftime("%d-%b %I:%M %p")}_
"""
        await self.telegram.send_message(message)

    async def run(self):
        await self.telegram.send_message(
            f"🚀 *UNIQUE Bot Started*\n\n"
            f"📊 Markets: {len(config.COINS_TO_MONITOR)}\n"
            f"🎯 Min Score: {config.MIN_SCORE}%\n"
            f"⏱️ Scan: {config.SCAN_INTERVAL}s"
        )

        while True:
            await self.scan_all()
            await asyncio.sleep(config.SCAN_INTERVAL)

async def main():
    bot = UniqueTradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
