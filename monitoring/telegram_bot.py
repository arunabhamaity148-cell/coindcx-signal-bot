"""
Telegram Bot for Real-time Monitoring and Control
Sends notifications and accepts commands
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging
from datetime import datetime
from typing import Dict
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingBotNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token from BotFather
            chat_id: Your Telegram chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.app = None
        self.bot_status = {
            'running': False,
            'capital': 0,
            'daily_pnl': 0,
            'total_trades': 0,
            'open_positions': 0
        }
    
    async def initialize(self):
        """Initialize the bot application"""
        self.app = Application.builder().token(self.bot_token).build()
        
        # Register command handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("balance", self.cmd_balance))
        self.app.add_handler(CommandHandler("trades", self.cmd_trades))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("emergency", self.cmd_emergency))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        
        logger.info("✅ Telegram bot initialized")
    
    async def send_message(self, message: str):
        """Send a message to Telegram"""
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
    
    # ==================== NOTIFICATIONS ====================
    
    async def notify_bot_started(self, capital: float, mode: str):
        """Notify when bot starts"""
        message = f"""
🤖 <b>TRADING BOT STARTED</b>

Mode: <b>{mode.upper()}</b>
Capital: ₹{capital:,.2f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Bot is now monitoring the market...
        """
        await self.send_message(message.strip())
    
    async def notify_bot_stopped(self):
        """Notify when bot stops"""
        message = "🛑 <b>TRADING BOT STOPPED</b>"
        await self.send_message(message)
    
    async def notify_trade_opened(self, trade: Dict):
        """Notify when a trade is opened"""
        emoji = "📈" if trade['side'] == 'LONG' else "📉"
        message = f"""
{emoji} <b>POSITION OPENED</b>

Symbol: {trade['symbol']}
Side: <b>{trade['side']}</b>
Entry: ₹{trade['entry_price']:,.2f}
Size: ₹{trade['size']:,.2f}
Leverage: {trade['leverage']}x

Stop Loss: ₹{trade['stop_loss']:,.2f}
Take Profit: ₹{trade['take_profit']:,.2f}

ML Confidence: {trade.get('confidence', 0):.1%}
        """
        await self.send_message(message.strip())
    
    async def notify_trade_closed(self, trade: Dict):
        """Notify when a trade is closed"""
        pnl = trade.get('pnl', 0)
        emoji = "✅" if pnl > 0 else "❌"
        
        message = f"""
{emoji} <b>POSITION CLOSED</b>

Symbol: {trade['symbol']}
Side: {trade['side']}
Exit: ₹{trade['exit_price']:,.2f}

Entry: ₹{trade['entry_price']:,.2f}
P&L: ₹{pnl:,.2f} ({pnl/trade['size']*100:+.2f}%)

Duration: {trade.get('duration_hours', 0):.1f}h
Reason: {trade.get('close_reason', 'unknown')}
        """
        await self.send_message(message.strip())
    
    async def notify_daily_summary(self, summary: Dict):
        """Send daily summary"""
        message = f"""
📊 <b>DAILY SUMMARY</b>

Date: {datetime.now().strftime('%Y-%m-%d')}

💰 P&L: ₹{summary['daily_pnl']:,.2f}
📈 Trades: {summary['total_trades']}
✅ Wins: {summary['wins']} ({summary['win_rate']:.1%})
❌ Losses: {summary['losses']}

💵 Current Capital: ₹{summary['capital']:,.2f}
📊 ROI: {summary['roi']:.2%}
        """
        await self.send_message(message.strip())
    
    async def notify_error(self, error_msg: str):
        """Notify about errors"""
        message = f"""
⚠️ <b>ERROR ALERT</b>

{error_msg}

Time: {datetime.now().strftime('%H:%M:%S')}
        """
        await self.send_message(message.strip())
    
    async def notify_warning(self, warning_msg: str):
        """Notify about warnings"""
        message = f"⚠️ <b>WARNING:</b> {warning_msg}"
        await self.send_message(message)
    
    async def notify_daily_loss_limit(self, loss: float, limit: float):
        """Notify when daily loss limit is reached"""
        message = f"""
🚨 <b>DAILY LOSS LIMIT REACHED</b>

Loss: ₹{abs(loss):,.2f}
Limit: ₹{limit:,.2f}

All positions closed.
Bot paused until tomorrow.
        """
        await self.send_message(message.strip())
    
    # ==================== COMMAND HANDLERS ====================
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        self.bot_status['running'] = True
        await update.message.reply_text(
            "✅ Bot started!\n\n"
            "Use /help to see available commands."
        )
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        self.bot_status['running'] = False
        await update.message.reply_text("🛑 Bot stopped!")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        status = "🟢 Running" if self.bot_status['running'] else "🔴 Stopped"
        
        message = f"""
📊 <b>BOT STATUS</b>

Status: {status}
Open Positions: {self.bot_status['open_positions']}
Total Trades Today: {self.bot_status['total_trades']}
Daily P&L: ₹{self.bot_status['daily_pnl']:,.2f}
        """
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command"""
        message = f"""
💰 <b>ACCOUNT BALANCE</b>

Current Capital: ₹{self.bot_status['capital']:,.2f}
Daily P&L: ₹{self.bot_status['daily_pnl']:,.2f}
        """
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command"""
        message = f"""
📈 <b>TODAY'S TRADES</b>

Total: {self.bot_status['total_trades']}
Open: {self.bot_status['open_positions']}

Use /status for more details
        """
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command"""
        await update.message.reply_text(
            "⏸️ Trading paused!\n\n"
            "Open positions will remain open.\n"
            "Use /resume to continue trading."
        )
    
    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command"""
        self.bot_status['running'] = True
        await update.message.reply_text("▶️ Trading resumed!")
    
    async def cmd_emergency(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /emergency command"""
        await update.message.reply_text(
            "🚨 <b>EMERGENCY STOP ACTIVATED</b>\n\n"
            "Closing all positions at market price...",
            parse_mode='HTML'
        )
        # This would trigger emergency close in main bot
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        message = """
📚 <b>AVAILABLE COMMANDS</b>

/start - Start the bot
/stop - Stop the bot
/status - Show current status
/balance - Show account balance
/trades - Show today's trades
/pause - Pause trading
/resume - Resume trading
/emergency - Emergency close all
/help - Show this help message
        """
        await update.message.reply_text(message, parse_mode='HTML')
    
    # ==================== UPDATE STATUS ====================
    
    def update_status(self, **kwargs):
        """Update bot status"""
        self.bot_status.update(kwargs)
    
    async def run_polling(self):
        """Run the bot with polling"""
        await self.initialize()
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        
        logger.info("✅ Telegram bot is running...")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()


# ==================== USAGE EXAMPLE ====================
async def main():
    """Test the bot"""
    # Get credentials from environment or config
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN')
    CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID')
    
    if BOT_TOKEN == 'YOUR_BOT_TOKEN':
        logger.error("❌ Please set TELEGRAM_BOT_TOKEN environment variable")
        return
    
    # Initialize notifier
    notifier = TradingBotNotifier(BOT_TOKEN, CHAT_ID)
    await notifier.initialize()
    
    # Test notifications
    logger.info("📱 Sending test notifications...")
    
    await notifier.notify_bot_started(capital=10000, mode='paper')
    
    # Simulate trade
    test_trade = {
        'symbol': 'BTC/USDT:USDT',
        'side': 'LONG',
        'entry_price': 50000,
        'size': 1000,
        'leverage': 5,
        'stop_loss': 49000,
        'take_profit': 51500,
        'confidence': 0.75
    }
    
    await notifier.notify_trade_opened(test_trade)
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Simulate close
    test_trade.update({
        'exit_price': 51000,
        'pnl': 200,
        'duration_hours': 2.5,
        'close_reason': 'take_profit'
    })
    
    await notifier.notify_trade_closed(test_trade)
    
    logger.info("✅ Test notifications sent!")
    logger.info("Bot is now running. Press Ctrl+C to stop.")
    
    # Run the bot
    await notifier.run_polling()


if __name__ == "__main__":
    asyncio.run(main())


# ==================== SETUP INSTRUCTIONS ====================
"""
1. Create Telegram Bot:
   - Open Telegram and search for @BotFather
   - Send /newbot
   - Follow instructions to create bot
   - Copy the bot token

2. Get Your Chat ID:
   - Send a message to your bot
   - Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   - Find your chat_id in the response

3. Set Environment Variables:
   export TELEGRAM_BOT_TOKEN="your_token_here"
   export TELEGRAM_CHAT_ID="your_chat_id_here"

4. Add to config/api_keys.env:
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here

5. Test the bot:
   python monitoring/telegram_bot.py

6. Integrate with main bot in main.py
"""