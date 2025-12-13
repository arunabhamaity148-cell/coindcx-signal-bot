"""
CoinDCX Trading Bot - Main Entry Point
Monitors 50+ coins on CoinDCX and sends signals to Telegram
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List
import os
from dotenv import load_dotenv

from helpers import CoinDCXAPI, TelegramNotifier, DatabaseManager
from logic import SignalGenerator, TradeMode

# Load environment variables
load_dotenv()

# Configuration
class Config:
    # CoinDCX API (Get from https://coindcx.com/api-trading)
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY', 'your_api_key')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET', 'your_secret')
    
    # Telegram Bot (Get from @BotFather)
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_bot_token')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_chat_id')
    
    # Trading Parameters
    COINS_TO_MONITOR = [
        'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC',
        'LINK', 'UNI', 'ATOM', 'LTC', 'ETC', 'XLM', 'ALGO', 'VET', 'ICP', 'FIL',
        'THETA', 'TRX', 'AAVE', 'EOS', 'AXS', 'SAND', 'MANA', 'GRT', 'ENJ', 'CHZ',
        'NEAR', 'FTM', 'ONE', 'HBAR', 'EGLD', 'ZIL', 'BAT', 'WAVES', 'KAVA', 'CELO',
        'AR', 'ROSE', 'LRC', 'IMX', 'GAL', 'APE', 'GMT', 'OP', 'ARB', 'SUI'
    ]
    
    TIMEFRAMES = ['5m', '15m', '1h']  # Quick, Mid, Trend modes
    MIN_LOGIC_SCORE = 65  # Minimum score to send signal
    SCAN_INTERVAL = 60  # seconds
    
    # Database
    DB_PATH = 'trading_bot.db'

config = Config()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """Main Trading Bot Class"""
    
    def __init__(self):
        self.dcx_api = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager(config.DB_PATH)
        self.signal_generator = SignalGenerator()
        self.processed_signals = set()
        
        logger.info("🤖 Trading Bot Initialized")
    
    async def get_coindcx_markets(self) -> List[str]:
        """Get available trading pairs from CoinDCX"""
        try:
            markets = await self.dcx_api.get_markets()
            
            # Filter INR futures markets for configured coins
            inr_futures = []
            for market in markets:
                symbol = market.get('symbol', '')
                # Example: B-SOL_USDT, I-BTC_INRT (futures)
                for coin in config.COINS_TO_MONITOR:
                    if f'{coin}INRT' in symbol or f'{coin}INR' in symbol:
                        inr_futures.append(symbol)
                        break
            
            logger.info(f"📊 Found {len(inr_futures)} INR futures markets")
            return inr_futures
        
        except Exception as e:
            logger.error(f"❌ Error fetching markets: {e}")
            return []
    
    async def analyze_coin(self, market: str, timeframe: str) -> Dict:
        """Analyze a single coin and generate signal"""
        try:
            # Fetch OHLCV data from CoinDCX
            candles = await self.dcx_api.get_candles(market, timeframe, limit=500)
            
            if not candles or len(candles) < 200:
                return None
            
            # Fetch orderbook
            orderbook = await self.dcx_api.get_orderbook(market)
            
            # Get current price in INR
            ticker = await self.dcx_api.get_ticker(market)
            current_price_inr = float(ticker.get('last_price', 0))
            
            if current_price_inr == 0:
                return None
            
            # Generate signal using 45 logics
            signal = await self.signal_generator.generate_signal(
                market=market,
                candles=candles,
                orderbook=orderbook,
                timeframe=timeframe,
                current_price_inr=current_price_inr
            )
            
            return signal
        
        except Exception as e:
            logger.error(f"❌ Error analyzing {market} on {timeframe}: {e}")
            return None
    
    async def scan_markets(self):
        """Scan all markets and generate signals"""
        logger.info("🔍 Starting market scan...")
        
        markets = await self.get_coindcx_markets()
        
        if not markets:
            logger.warning("⚠️ No markets found to scan")
            return
        
        signals_found = 0
        
        for market in markets:
            for timeframe in config.TIMEFRAMES:
                try:
                    signal = await self.analyze_coin(market, timeframe)
                    
                    if signal and signal['logic_score'] >= config.MIN_LOGIC_SCORE:
                        # Create unique signal key
                        signal_key = f"{market}_{timeframe}_{signal['side']}_{datetime.now().strftime('%Y%m%d%H')}"
                        
                        # Check if already processed
                        if signal_key not in self.processed_signals:
                            # Send to Telegram
                            await self.telegram.send_signal(signal)
                            
                            # Save to database
                            self.db.save_signal(signal)
                            
                            # Mark as processed
                            self.processed_signals.add(signal_key)
                            signals_found += 1
                            
                            logger.info(f"✅ Signal sent: {market} {timeframe} {signal['side']} - Score: {signal['logic_score']}%")
                            
                            # Cleanup old signals (keep last 500)
                            if len(self.processed_signals) > 500:
                                old_signals = list(self.processed_signals)[:100]
                                for old in old_signals:
                                    self.processed_signals.remove(old)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                
                except Exception as e:
                    logger.error(f"❌ Error processing {market} {timeframe}: {e}")
                    continue
        
        logger.info(f"✅ Market scan completed. Signals found: {signals_found}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Trading Bot Started!")
        logger.info(f"📊 Monitoring {len(config.COINS_TO_MONITOR)} coins")
        logger.info(f"⏱️ Timeframes: {config.TIMEFRAMES}")
        logger.info(f"🎯 Min Logic Score: {config.MIN_LOGIC_SCORE}%")
        
        # Send startup message
        await self.telegram.send_message(
            "🤖 *Trading Bot Started*\n\n"
            f"📊 Coins: {len(config.COINS_TO_MONITOR)}\n"
            f"⏱️ Timeframes: {', '.join(config.TIMEFRAMES)}\n"
            f"🎯 Min Score: {config.MIN_LOGIC_SCORE}%\n"
            f"🔄 Scan Interval: {config.SCAN_INTERVAL}s"
        )
        
        while True:
            try:
                await self.scan_markets()
                
                # Wait before next scan
                logger.info(f"⏳ Waiting {config.SCAN_INTERVAL}s before next scan...")
                await asyncio.sleep(config.SCAN_INTERVAL)
            
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                await self.telegram.send_message("🛑 *Trading Bot Stopped*")
                break
            
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(10)

async def main():
    """Entry point"""
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Bot shutdown complete")
