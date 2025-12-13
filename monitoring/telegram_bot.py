"""
Premium-styled Telegram messages (Royal / Minimal Dark theme)
"""
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from telegram import Bot
from telegram.error import TelegramError, RetryAfter

logger = logging.getLogger(__name__)


class TradingBotNotifier:
    def __init__(self,
                 bot_token: Optional[str] = None,
                 chat_id: Optional[str] = None) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or ""
        self.chat_id   = chat_id   or os.getenv("TELEGRAM_CHAT_ID")   or ""
        self._bot: Optional[Bot] = None
        if self.bot_token and self.bot_token != "YOUR_BOT_TOKEN":
            try:
                self._bot = Bot(self.bot_token)
                logger.info("Premium Telegram client ready.")
            except Exception as e:
                logger.warning("Telegram init failed: %s", e)
        else:
            logger.warning("Telegram token not set – notifications disabled.")

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    async def _send(self, text: str) -> None:
        if not self._bot or not self.chat_id:
            return
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after)
            await self._send(text)  # one retry
        except TelegramError as e:
            logger.error("Telegram error: %s", e)

    # ------------------------------------------------------------------
    # Premium styled messages
    # ------------------------------------------------------------------
    async def notify_bot_started(self, capital: float, mode: str) -> None:
        msg = f"""
〽️ <b><u>Trading-Bot Activated</u></b>
├ Mode: <code>{mode.upper()}</code>
├ Capital: <b>₹{capital:,.2f}</b>
└ Time: <i>{datetime.now():%d %b %Y, %H:%M}</i>
"""
        await self._send(msg.strip())

    async def notify_bot_stopped(self) -> None:
        await self._send("⏹️ <b><u>Bot Deactivated</u></b>")

    async def notify_trade_opened(self, trade: Dict) -> None:
        side_emoji = "🛫" if trade.get("side") == "LONG" else "🛬"
        msg = f"""
{side_emoji} <b><u>New Position</u></b>
├ Pair: <code>{trade.get('symbol', 'N/A')}</code>
├ Side: <b>{trade.get('side', 'N/A')}</b>
├ Entry: <b>₹{trade.get('entry_price', 0):,.2f}</b>
├ Size: <b>₹{trade.get('position_size', 0):,.2f}</b>
├ Leverage: <code>{trade.get('leverage', 0)}×</code>
├ SL: <b>₹{trade.get('stop_loss', 0):,.2f}</b>
├ TP: <b>₹{trade.get('take_profit', 0):,.2f}</b>
└ Conf: <i>{trade.get('confidence', 0):.1%}</i>
"""
        await self._send(msg.strip())

    async def notify_trade_closed(self, trade: Dict) -> None:
        pnl = trade.get("pnl", 0)
        emoji = "🎯" if pnl > 0 else "⚠️"
        msg = f"""
{emoji} <b><u>Position Closed</u></b>
├ Pair: <code>{trade.get('symbol', 'N/A')}</code>
├ Side: <b>{trade.get('side', 'N/A')}</b>
├ Entry: <b>₹{trade.get('entry_price', 0):,.2f}</b>
├ Exit: <b>₹{trade.get('exit_price', 0):,.2f}</b>
├ P&L: <b>₹{pnl:+,.2f}</b> (<i>{pnl / trade.get('entry_price', 1) * 100:+.2f}%</i>)
└ Reason: <code>{trade.get('close_reason', 'unknown')}</code>
"""
        await self._send(msg.strip())

    async def notify_daily_summary(self, summary: Dict) -> None:
        msg = f"""
📊 <b><u>Daily Ledger</u></b>
├ Date: <i>{datetime.now():%d %b %Y}</i>
├ P&L: <b>₹{summary.get('daily_pnl', 0):+,.2f}</b>
├ Trades: <b>{summary.get('total_trades', 0)}</b>
├ Wins: <b>{summary.get('wins', 0)}</b> (<i>{summary.get('win_rate', 0):.1%}</i>)
├ Losses: <b>{summary.get('losses', 0)}</b>
├ Capital: <b>₹{summary.get('capital', 0):,.2f}</b>
└ ROI: <b>{summary.get('roi', 0):+.2%}</b>
"""
        await self._send(msg.strip())

    async def notify_error(self, error_msg: str) -> None:
        await self._send(f"🚨 <b><u>System Error</u></b>\n<code>{error_msg}</code>")

    async def notify_warning(self, warning_msg: str) -> None:
        await self._send(f"⚠️ <b><u>Alert</u></b>\n{warning_msg}")

    async def notify_daily_loss_limit(self, loss: float, limit: float) -> None:
        msg = f"""
🔒 <b><u>Daily Loss-Limit Hit</u></b>
├ Loss: <b>₹{abs(loss):,.2f}</b>
├ Limit: <b>₹{limit:,.2f}</b>
└ Bot <i>paused</i> until next day.
"""
        await self._send(msg.strip())


# --------------------------------------------------------------------------
# Quick self-test (run: python telegram_bot.py)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    async def _test():
        notifier = TradingBotNotifier()
        await notifier.notify_bot_started(capital=10000, mode="paper")
        await asyncio.sleep(1)
        await notifier.notify_bot_stopped()

    asyncio.run(_test())
