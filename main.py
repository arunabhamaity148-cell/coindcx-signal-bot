"""
UNIQUE Trading Bot - FIXED
"""

import asyncio
import logging
from datetime import datetime
from typing import List
import os
from dotenv import load_dotenv

from helpers import CoinDCXAPI, TelegramNotifier, DatabaseManager
from logic import UniqueSignalGenerator

load_dotenv()

class Config:
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY', 'your_api_key')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET', 'your_secret')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_bot_token')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_chat_id')
    
    # FIXED: More flexible coin matching
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
        self.dcx = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager(config.DB_PATH)
        self.signal_gen = UniqueSignalGenerator()
        
        self.processed = set()
        self.daily_signals = 0
        self.last_date = datetime.now().date()
        
        logger.info("🚀 UNIQUE Bot Initialized")
    
    async def get_markets(self) -> List[str]:
        """FIXED: Better market detection"""
        try:
            all_markets = await self.dcx.get_markets()
            
            if not all_markets:
                logger.error("❌ No markets returned from API")
                return []
            
            # Debug: Show sample market
            if len(all_markets) > 0:
                sample = all_markets[0]
                logger.info(f"Sample market structure: {sample}")
                logger.info(f"Market keys: {sample.keys() if isinstance(sample, dict) else 'Not a dict'}")
            
            inr_markets = []
            
            for market in all_markets:
                if not isinstance(market, dict):
                    continue
                
                # Get symbol/pair - try different field names
                symbol = (
                    market.get('symbol') or 
                    market.get('pair') or 
                    market.get('market') or 
                    market.get('coindcx_name') or 
                    ''
                )
                
                symbol = symbol.upper()
                
                # Skip if empty
                if not symbol:
                    continue
                
                # Check for INR/INRT pairs
                is_inr = ('INR' in symbol or 'INRT' in symbol)
                
                if not is_inr:
                    continue
                
                # Match with our coins
                for coin in config.COINS_TO_MONITOR:
                    # Multiple pattern matching
                    patterns = [
                        f'{coin}INR',
                        f'{coin}INRT',
                        f'I-{coin}',
                        f'B-{coin}',
                        f'{coin}_INR',
                        f'{coin}_INRT',
                    ]
                    
                    if any(pattern in symbol for pattern in patterns):
                        inr_markets.append(symbol)
                        logger.debug(f"✅ Matched: {symbol} for {coin}")
                        break
            
            logger.info(f"📊 {len(inr_markets)} INR markets found")
            
            if len(inr_markets) > 0:
                logger.info(f"Markets: {inr_markets[:5]}...")  # Show first 5
            else:
                logger.warning("⚠️ No INR markets matched!")
                logger.warning(f"Sample symbols from API: {[m.get('symbol', m.get('pair', '?')) for m in all_markets[:10]]}")
            
            return inr_markets
        
        except Exception as e:
            logger.error(f"❌ get_markets error: {e}", exc_info=True)
            return []
    
    async def analyze_market(self, market: str, timeframe: str):
        """Analyze with error handling"""
        try:
            candles = await self.dcx.get_candles(market, timeframe, limit=200)
            
            if candles is None or len(candles) < 100:
                logger.debug(f"Insufficient candles for {market}")
                return None
            
            orderbook = await self.dcx.get_orderbook(market)
            ticker = await self.dcx.get_ticker(market)
            
            price = float(ticker.get('last_price', 0)) if ticker else 0
            
            if price == 0:
                logger.debug(f"No price for {market}")
                return None
            
            signal = await self.signal_gen.generate_signal(
                market=market,
                candles=candles,
                orderbook=orderbook,
                timeframe=timeframe,
                current_price_inr=price
            )
            
            return signal
        
        except Exception as e:
            logger.error(f"❌ analyze_market error {market}: {e}")
            return None
    
    async def scan_all(self):
        """FIXED: Better error handling"""
        
        today = datetime.now().date()
        if today != self.last_date:
            self.daily_signals = 0
            self.last_date = today
            logger.info(f"📅 New day: {today}")
        
        logger.info("🔍 Scanning...")
        
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
                            
                            logger.info(
                                f"✅ Signal #{self.daily_signals}: "
                                f"{market} {signal['side']} | "
                                f"Score: {signal['logic_score']}%"
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
    
    async def send_unique_signal(self, signal: dict):
        """Send signal"""
        details = signal.get('details', {})
        
        insights = []
        
        if 'ACCELERATING' in details.get('momentum_wave', ''):
            insights.append("⚡ Momentum Accelerating")
        
        if details.get('smart_money') in ['BULLISH', 'BEARISH']:
            insights.append(f"🐋 Smart Money: {details['smart_money']}")
        
        if details.get('near_vacuum'):
            insights.append("🚀 Liquidity Vacuum")
        
        if 'SYNC' in details.get('market_sync', ''):
            insights.append("🎯 Market Synchronized")
        
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

🎨 *Insights:*
{insight_text}

⏱️ *Mode:* {signal['mode']}

🕐 _{datetime.now().strftime("%d-%b %I:%M %p")}_
"""
        
        try:
            await self.telegram.bot.send_message(
                chat_id=self.telegram.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"❌ Telegram: {e}")
    
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
            f"🎨 Proprietary Logic\n"
            f"📊 Coins: {len(config.COINS_TO_MONITOR)}\n"
            f"🎯 Min Score: {config.MIN_SCORE}%\n"
            f"Target: 15-25 signals/day"
        )
        
        while True:
            try:
                await self.scan_all()
                
                logger.info(f"⏳ Next scan in {config.SCAN_INTERVAL}s...")
                await asyncio.sleep(config.SCAN_INTERVAL)
            
            except KeyboardInterrupt:
                logger.info("🛑 Stopped")
                await self.telegram.send_message("🛑 Bot Stopped")
                break
            
            except Exception as e:
                logger.error(f"❌ Loop error: {e}", exc_info=True)
                await asyncio.sleep(10)
        
        await self.dcx.close()

async def main():
    bot = UniqueTradingBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Shutdown")