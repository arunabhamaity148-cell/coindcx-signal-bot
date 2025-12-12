"""
Final Telegram Bot for Trading Bot – Railway-ready
No polling, no extra threads.  Just “send me a message” helper.
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
    """
    Thin wrapper around python-telegram-bot Bot object.
    - Auto-loads token / chat-id from Railway env-vars
    - Fire-and-forget coroutines for every message
    - Graceful fail + log (never crash the trading loop)
    """

    def __init__(self,
                 bot_token: Optional[str] = None,
                 chat_id: Optional[str] = None) -> None:
        # 1. Priority: argument → env-var → fallback
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or ""
        self.chat_id   = chat_id   or os.getenv("TELEGRAM_CHAT_ID")   or ""

        self._bot: Optional[Bot] = None
        if self.bot_token and self.bot_token != "YOUR_BOT_TOKEN":
            try:
                self._bot = Bot(self.bot_token)
                logger.info("Telegram bot client created.")
            except Exception as e:
                logger.warning("Telegram Bot init failed: %s", e)
        else:
            logger.warning("Telegram token not set – notifications disabled.")

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    async def _send(self, text: str, parse_mode: str = "HTML") -> None:
        """Actually send a message (with retry)."""
        if not self._bot or not self.chat_id:
            return

        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
            )
        except RetryAfter as e:
            logger.warning("Telegram flood-control: retrying in %s sec", e.retry_after)
            await asyncio.sleep(e.retry_after)
            await self._send(text, parse_mode)  # one recursive retry
        except TelegramError as e:
            logger.error("Telegram error while sending: %s", e)

    # ------------------------------------------------------------------
    # Public high-level helpers
    # ------------------------------------------------------------------
    async def notify_bot_started(self, capital: float, mode: str) -> None:
        msg = f"""
🤖 <b>Trading-Bot Started</b>
Mode: <b>{mode.upper()}</b>
Capital: ₹{capital:,.2f}
Time: {datetime.now():%Y-%m-%d %H:%M:%S}
"""
        await self._send(msg.strip())

    async def notify_bot_stopped(self) -> None:
        await self._send("🛑 <b>Trading-Bot Stopped</b>")

    async def notify_trade_opened(self, trade: Dict) -> None:
        emoji = "📈" if trade.get("side") == "LONG" else "📉"
        msg = f"""
{emoji} <b>Position Opened</b>
Symbol: <code>{trade.get('symbol', 'N/A')}</code>
Side: <b>{trade.get('side', 'N/A')}</b>
Entry: ₹{trade.get('entry_price', 0):,.2f}
Size: ₹{trade.get('position_size', 0):,.2f}
Leverage: {trade.get('leverage', 0)}×
SL: ₹{trade.get('stop_loss', 0):,.2f}
TP: ₹{trade.get('take_profit', 0):,.2f}
ML-Conf: {trade.get('confidence', 0):.1%}
"""
        await self._send(msg.strip())

    async def notify_trade_closed(self, trade: Dict) -> None:
        pnl = trade.get("pnl", 0)
        emoji = "✅" if pnl > 0 else "❌"
        msg = f"""
{emoji} <b>Position Closed</b>
Symbol: <code>{trade.get('symbol', 'N/A')}</code>
Side: {trade.get('side', 'N/A')}
Entry: ₹{trade.get('entry_price', 0):,.2f}
Exit: ₹{trade.get('exit_price', 0):,.2f}
P&L: ₹{pnl:+,.2f} ({pnl / trade.get('entry_price', 1) * 100:+.2f}%)
Reason: {trade.get('close_reason', 'unknown')}
"""
        await self._send(msg.strip())

    async def notify_daily_summary(self, summary: Dict) -> None:
        msg = f"""
📊 <b>Daily Summary</b>
Date: {datetime.now():%Y-%m-%d}
P&L: ₹{summary.get('daily_pnl', 0):+,.2f}
Trades: {summary.get('total_trades', 0)}
Wins: {summary.get('wins', 0)} ({summary.get('win_rate', 0):.1%})
Losses: {summary.get('losses', 0)}
Capital: ₹{summary.get('capital', 0):,.2f}
ROI: {summary.get('roi', 0):+.2%}
"""
        await self._send(msg.strip())

    async def notify_error(self, error_msg: str) -> None:
        await self._send(f"⚠️ <b>Error</b>\n{error_msg}")

    async def notify_warning(self, warning_msg: str) -> None:
        await self._send(f"⚠️ <b>Warning</b>\n{warning_msg}")

    async def notify_daily_loss_limit(self, loss: float, limit: float) -> None:
        msg = f"""
🚨 <b>Daily Loss-Limit Hit</b>
Loss: ₹{abs(loss):,.2f}
Limit: ₹{limit:,.2f}
Bot paused until next day.
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
