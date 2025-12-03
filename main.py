import asyncio
import os
import logging
from datetime import datetime
from helpers import (
    MarketData, 
    calculate_quick_signals, 
    calculate_mid_signals, 
    calculate_trend_signals,
    telegram_formatter_style_c,
    send_telegram_message,
    send_copy_block,
    cooldown_manager,
    btc_calm_check,
    spread_and_depth_check,
    safety_check_sl_vs_liq,
    calc_single_tp_sl
)
from config import (
    BINANCE_API_KEY,
    BINANCE_SECRET,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    TRADING_PAIRS,
    QUICK_MIN_SCORE,
    MID_MIN_SCORE,
    TREND_MIN_SCORE,
    COOLDOWN_SECONDS
)
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SignalBot:
    def __init__(self):
        self.market = MarketData(BINANCE_API_KEY, BINANCE_SECRET)
        self.running = True
        
    async def analyze_symbol(self, symbol):
        """Analyze symbol for all signal types"""
        try:
            # Fetch market data
            data = await self.market.get_all_data(symbol)
            if not data:
                return
            
            # Safety checks
            if not spread_and_depth_check(data):
                logger.info(f"❌ {symbol} failed spread/depth check")
                return
            
            # Calculate all signals
            quick = calculate_quick_signals(data)
            mid = calculate_mid_signals(data)
            trend = calculate_trend_signals(data)
            
            # Send signals if threshold met
            if quick['score'] >= QUICK_MIN_SCORE and quick['direction'] != 'none':
                await self.process_signal(symbol, 'QUICK', quick, data)
            
            if mid['score'] >= MID_MIN_SCORE and mid['direction'] != 'none':
                await self.process_signal(symbol, 'MID', mid, data)
            
            if trend['score'] >= TREND_MIN_SCORE and trend['direction'] != 'none':
                await self.process_signal(symbol, 'TREND', trend, data)
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    async def process_signal(self, symbol, mode, result, data):
        """Process and send signal with all checks"""
        try:
            direction = result['direction']
            price = data['price']
            
            # Cooldown check
            cooldown = COOLDOWN_SECONDS.get(mode, 1800)
            if not cooldown_manager.can_send(symbol, mode, cooldown):
                logger.info(f"⏳ {symbol} {mode} in cooldown")
                return
            
            # Dedupe check
            rule_key = f"{symbol}_{mode}"
            if not cooldown_manager.ensure_single_alert(rule_key, result['triggers'], price, mode):
                logger.info(f"🔁 {symbol} {mode} duplicate signal")
                return
            
            # BTC calm check
            if not await btc_calm_check(self.market):
                logger.warning(f"⚠️ BTC volatile, skipping {symbol} {mode} signal")
                return
            
            # Calculate TP/SL
            tp_sl = calc_single_tp_sl(price, direction, mode)
            
            # Safety check
            if not safety_check_sl_vs_liq(price, tp_sl['sl'], tp_sl['leverage']):
                logger.warning(f"⚠️ {symbol} {mode} SL too close to liquidation")
                return
            
            # Format and send message
            message, code_block = telegram_formatter_style_c(symbol, mode, direction, result, data)
            
            await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
            await asyncio.sleep(1)
            await send_copy_block(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, code_block)
            
            logger.info(f"✅ {mode} {direction.upper()} signal sent for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Signal Bot Started!")
        logger.info(f"📊 Monitoring: {', '.join(TRADING_PAIRS)}")
        
        try:
            await send_telegram_message(
                TELEGRAM_TOKEN, 
                TELEGRAM_CHAT_ID, 
                "🤖 <b>30 Logic Signal Bot Online!</b>\n\n"
                f"📊 Pairs: {', '.join(TRADING_PAIRS)}\n"
                f"⚡ Quick Min Score: {QUICK_MIN_SCORE}/10\n"
                f"🔵 Mid Min Score: {MID_MIN_SCORE}/10\n"
                f"🟣 Trend Min Score: {TREND_MIN_SCORE}/10\n\n"
                "✅ All systems operational"
            )
        except Exception as e:
            logger.error(f"Failed to send startup message: {e}")
        
        while self.running:
            try:
                for symbol in TRADING_PAIRS:
                    await self.analyze_symbol(symbol)
                    await asyncio.sleep(2)
                
                # Wait before next cycle
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("⚠️ Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)
        
        # Cleanup
        await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down...")
        try:
            await self.market.close()
            await send_telegram_message(
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                "🛑 <b>Signal Bot Stopped</b>\n\nBot has been shut down gracefully."
            )
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        logger.info("✅ Shutdown complete")

if __name__ == "__main__":
    bot = SignalBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")