import asyncio
import os
import logging
from datetime import datetime
from helpers import MarketData, calculate_quick_signals, calculate_mid_signals, calculate_trend_signals, send_telegram_message
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET = os.getenv('BINANCE_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
QUICK_MIN_SCORE = 5
MID_MIN_SCORE = 5
TREND_MIN_SCORE = 5

class SignalBot:
    def __init__(self):
        self.market = MarketData(BINANCE_API_KEY, BINANCE_SECRET)
        
    async def analyze_symbol(self, symbol):
        """Analyze symbol for signals"""
        try:
            data = await self.market.get_all_data(symbol)
            
            # Calculate all signals
            quick = calculate_quick_signals(data)
            mid = calculate_mid_signals(data)
            trend = calculate_trend_signals(data)
            
            # Send signals if threshold met
            if quick['score'] >= QUICK_MIN_SCORE and quick['direction'] != 'none':
                await self.send_quick_signal(symbol, quick, data)
            
            if mid['score'] >= MID_MIN_SCORE and mid['direction'] != 'none':
                await self.send_mid_signal(symbol, mid, data)
            
            if trend['score'] >= TREND_MIN_SCORE and trend['direction'] != 'none':
                await self.send_trend_signal(symbol, trend, data)
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    async def send_quick_signal(self, symbol, result, data):
        """Send Quick signal"""
        direction = "🟢 BUY" if result['direction'] == 'long' else "🔴 SELL"
        price = data['price']
        
        if result['direction'] == 'long':
            tp1, tp2 = price * 1.004, price * 1.008
            sl = price * 0.9975
        else:
            tp1, tp2 = price * 0.996, price * 0.992
            sl = price * 1.0025
        
        msg = f"""⚡ QUICK {direction} SIGNAL
━━━━━━━━━━━━━━━━━━
Pair: {symbol}
Entry: ${price:.2f}
TP1: ${tp1:.2f} (0.4%)
TP2: ${tp2:.2f} (0.8%)
SL: ${sl:.2f} (0.25%)

Score: {result['score']}/10
Confidence: {result['score']*10}%

Triggers:
{result['triggers']}

Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC
━━━━━━━━━━━━━━━━━━"""
        
        await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
        logger.info(f"✅ Quick {direction} signal sent for {symbol}")
    
    async def send_mid_signal(self, symbol, result, data):
        """Send Mid signal"""
        direction = "🟢 BUY" if result['direction'] == 'long' else "🔴 SELL"
        price = data['price']
        
        if result['direction'] == 'long':
            tp1, tp2, tp3 = price * 1.015, price * 1.025, price * 1.04
            sl = price * 0.992
        else:
            tp1, tp2, tp3 = price * 0.985, price * 0.975, price * 0.96
            sl = price * 1.008
        
        msg = f"""🔵 MID {direction} SIGNAL
━━━━━━━━━━━━━━━━━━
Pair: {symbol}
Entry: ${price:.2f}
TP1: ${tp1:.2f} (1.5%)
TP2: ${tp2:.2f} (2.5%)
TP3: ${tp3:.2f} (4%)
SL: ${sl:.2f} (0.8%)

Score: {result['score']}/10
Confidence: {result['score']*10}%

Triggers:
{result['triggers']}

Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC
━━━━━━━━━━━━━━━━━━"""
        
        await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
        logger.info(f"✅ Mid {direction} signal sent for {symbol}")
    
    async def send_trend_signal(self, symbol, result, data):
        """Send Trend signal"""
        direction = "🟢 BUY" if result['direction'] == 'long' else "🔴 SELL"
        price = data['price']
        
        if result['direction'] == 'long':
            tp1, tp2, tp3 = price * 1.03, price * 1.06, price * 1.09
            sl = price * 0.985
        else:
            tp1, tp2, tp3 = price * 0.97, price * 0.94, price * 0.91
            sl = price * 1.015
        
        msg = f"""🟣 TREND {direction} SIGNAL
━━━━━━━━━━━━━━━━━━
Pair: {symbol}
Entry: ${price:.2f}
TP1: ${tp1:.2f} (3%)
TP2: ${tp2:.2f} (6%)
TP3: ${tp3:.2f} (9%)
SL: ${sl:.2f} (1.5%)

Score: {result['score']}/10
Confidence: {result['score']*10}%

Triggers:
{result['triggers']}

Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC
━━━━━━━━━━━━━━━━━━"""
        
        await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
        logger.info(f"✅ Trend {direction} signal sent for {symbol}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Signal Bot Started!")
        await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, "🤖 30 Logic Signal Bot Online!")
        
        while True:
            try:
                for symbol in TRADING_PAIRS:
                    await self.analyze_symbol(symbol)
                    await asyncio.sleep(2)
                
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = SignalBot()
    asyncio.run(bot.run())