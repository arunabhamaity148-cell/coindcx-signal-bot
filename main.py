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
    # API Credentials
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    # UNIQUE SETTINGS - Optimized for quality
    COINS_TO_MONITOR = [
        # High liquidity (Best for our unique logic)
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'MATIC',
        'DOT', 'AVAX', 'LINK', 'UNI', 'ATOM', 'LTC', 'ETC',

        # Good movers (Chaos oscillator works best)
        'NEAR', 'FTM', 'SAND', 'MANA', 'AXS', 'APE', 'OP', 'ARB',
        'SUI', 'TRX', 'AAVE', 'GRT', 'ALGO', 'VET', 'ICP',

        # Volume-based opportunities
        'FIL', 'THETA', 'XLM', 'EOS', 'CHZ', 'ENJ', 'ROSE', 'LRC',
        'IMX', 'GMT', 'GAL', 'ONE', 'HBAR', 'EGLD', 'ZIL', 'WAVES'
    ]

    TIMEFRAMES = ['5m', '15m', '1h']
    MIN_SCORE = 45  # Lower threshold for unique logic (more signals)
    SCAN_INTERVAL = 75  # 75 seconds (optimal for API limits)
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
    """PROPRIETARY Trading Bot - Unique Logic"""

    def __init__(self):
        # Validate credentials
        if not config.COINDCX_API_KEY or not config.COINDCX_SECRET:
            raise ValueError("❌ CoinDCX API credentials missing! Check .env file")
        
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            raise ValueError("❌ Telegram credentials missing! Check .env file")

        self.dcx = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager(config.DB_PATH)
        self.signal_gen = UniqueSignalGenerator()

        self.processed = set()
        self.daily_signals = 0
        self.last_date = datetime.now().date()

        logger.info("🚀 UNIQUE Bot Initialized")

    async def get_markets(self) -> List[str]:
        """Get CoinDCX markets"""
        try:
            markets = await self.dcx.get_markets()
            
            # Validate response
            if not markets:
                logger.error("❌ Empty markets response")
                return []
            
            if not isinstance(markets, list):
                logger.error(f"❌ Invalid markets data type: {type(markets)}")
                logger.error(f"Response: {str(markets)[:200]}")
                return []

            inr_markets = []
            for m in markets:
                # Ensure each market is a dictionary
                if not isinstance(m, dict):
                    continue
                    
                symbol = m.get('symbol', '')
                if not symbol:
                    continue
                
                for coin in config.COINS_TO_MONITOR:
                    if f'{coin}INR' in symbol or f'I-{coin}_INR' in symbol or symbol.startswith(f'B-{coin}'):
                        inr_markets.append(symbol)
                        break

            logger.info(f"📊 {len(inr_markets)} markets loaded")
            return inr_markets

        except Exception as e:
            logger.error(f"❌ Market load error: {e}", exc_info=True)
            return []

    async def analyze_market(self, market: str, timeframe: str):
        """Analyze with unique logic"""
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

            # UNIQUE SIGNAL GENERATION
            signal = await self.signal_gen.generate_signal(
                market=market,
                candles=candles,
                orderbook=orderbook,
                timeframe=timeframe,
                current_price_inr=price
            )

            return signal

        except Exception as e:
            logger.error(f"❌ Analysis error {market}: {e}")
            return None

    async def scan_all(self):
        """Full market scan"""

        today = datetime.now().date()
        if today != self.last_date:
            self.daily_signals = 0
            self.last_date = today
            logger.info(f"📅 New trading day: {today}")

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

                            # Enhanced signal message
                            await self.send_unique_signal(signal)

                            self.db.save_signal(signal)
                            self.processed.add(key)
                            self.daily_signals += 1
                            found += 1

                            logger.info(
                                f"✅ UNIQUE Signal #{self.daily_signals}: "
                                f"{market} {signal['side']} | "
                                f"Score: {signal['logic_score']}% | "
                                f"Conf: {signal['confidence']}"
                            )

                            if len(self.processed) > 300:
                                old = list(self.processed)[:100]
                                for o in old:
                                    self.processed.remove(o)

                    await asyncio.sleep(0.2)

                except Exception as e:
                    logger.error(f"❌ {market} {tf}: {e}")
                    continue

        logger.info(f"✅ Scan done. Found: {found} | Today: {self.daily_signals}")

    async def send_unique_signal(self, signal: Dict):
        """Send enhanced signal with unique insights"""

        details = signal.get('details', {})

        # Build unique insights
        insights = []

        if 'ACCELERATING' in details.get('momentum_wave', ''):
            insights.append("⚡ Momentum Accelerating")

        if details.get('smart_money') in ['BULLISH', 'BEARISH']:
            insights.append(f"🐋 Smart Money: {details['smart_money']}")

        if details.get('near_vacuum'):
            insights.append("🚀 Near Liquidity Vacuum")

        if 'SYNC' in details.get('market_sync', ''):
            insights.append("🎯 Market Synchronized")

        if 'EXHAUSTION' in details.get('exhaustion', ''):
            insights.append("🔄 Reversal Setup")

        if details.get('accumulation'):
            insights.append("📊 Stealth Accumulation")

        if 'SHIFT' in details.get('sentiment', ''):
            insights.append("💫 Sentiment Shifting")

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

⏱️ *Mode:* {signal['mode']}
⚠️ *Manual trade on CoinDCX*

🕐 _{datetime.now().strftime("%d-%b %I:%M %p")}_
"""

        try:
            await self.telegram.send_message(message)
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")

    async def run(self):
        """Main loop"""

        logger.info("🚀 UNIQUE Trading Bot Started!")
        logger.info("=" * 50)
        logger.info("📊 PROPRIETARY LOGIC ACTIVE")
        logger.info("=" * 50)
        logger.info(f"Markets: {len(config.COINS_TO_MONITOR)}")
        logger.info(f"Timeframes: {config.TIMEFRAMES}")
        logger.info(f"Min Score: {config.MIN_SCORE}%")
        logger.info(f"Scan Interval: {config.SCAN_INTERVAL}s")
        logger.info("=" * 50)

        await self.telegram.send_message(
            f"🚀 *UNIQUE Bot Started*\n\n"
            f"🎨 Proprietary Logic Active\n"
            f"📊 Markets: {len(config.COINS_TO_MONITOR)}\n"
            f"🎯 Min Score: {config.MIN_SCORE}%\n"
            f"⏱️ Scan: {config.SCAN_INTERVAL}s\n\n"
            f"Target: 15-25 quality signals/day"
        )

        while True:
            try:
                await self.scan_all()

                logger.info(f"⏳ Next scan in {config.SCAN_INTERVAL}s...")
                await asyncio.sleep(config.SCAN_INTERVAL)

            except KeyboardInterrupt:
                logger.info("🛑 Stopped")
                await self.telegram.send_message("🛑 UNIQUE Bot Stopped")
                break

            except Exception as e:
                logger.error(f"❌ Loop error: {e}", exc_info=True)
                await asyncio.sleep(10)

        await self.dcx.close()

async def main():
    try:
        bot = UniqueTradingBot()
        await bot.run()
    except ValueError as e:
        logger.error(str(e))
        print("\n" + "="*60)
        print("⚠️  SETUP REQUIRED")
        print("="*60)
        print("\n1. Create a .env file in the project root")
        print("2. Add your credentials:")
        print("   COINDCX_API_KEY=your_key")
        print("   COINDCX_SECRET=your_secret")
        print("   TELEGRAM_BOT_TOKEN=your_token")
        print("   TELEGRAM_CHAT_ID=your_chat_id")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Shutdown")