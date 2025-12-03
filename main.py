#!/usr/bin/env python3
"""
30-Logic Binance Futures Signal Bot
Railway-ready entry point
"""

import asyncio
import os
import signal
import sys
import logging
from datetime import datetime

from dotenv import load_dotenv
from helpers import (
    MarketData,
    calculate_quick_signals,
    calculate_mid_signals,
    calculate_trend_signals,
    telegram_formatter_style_c,
    send_telegram_message,
    send_copy_block,
    cooldown_manager,
    spread_and_depth_check,
    btc_calm_check,
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
    COOLDOWN_SECONDS,
)

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("signal-bot")

# ---------- graceful shutdown ----------
shutdown_event = asyncio.Event()

def _handle_signal(signum, frame):
    logger.warning(f"Signal {signum} received – shutting down gracefully…")
    shutdown_event.set()

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------- bot ----------
class SignalBot:
    def __init__(self):
        self.market = MarketData(BINANCE_API_KEY, BINANCE_SECRET)

    # ---------------- analyse one symbol ----------------
    async def analyse_symbol(self, symbol: str) -> None:
        try:
            data = await self.market.get_all_data(symbol)
            if not data:
                return

            # safety checks
            if not spread_and_depth_check(data):
                return
            if not await btc_calm_check(self.market):
                return

            price = data["price"]

            # ---------- QUICK ----------
            quick = calculate_quick_signals(data)
            if (
                quick["score"] >= QUICK_MIN_SCORE
                and quick["direction"] != "none"
                and cooldown_manager.can_send(
                    symbol, "QUICK", COOLDOWN_SECONDS["QUICK"]
                )
                and cooldown_manager.ensure_single_alert(
                    "QUICK", quick["triggers"], price, "QUICK"
                )
            ):
                await self._dispatch_signal(symbol, "QUICK", quick, data)

            # ---------- MID ----------
            mid = calculate_mid_signals(data)
            if (
                mid["score"] >= MID_MIN_SCORE
                and mid["direction"] != "none"
                and cooldown_manager.can_send(
                    symbol, "MID", COOLDOWN_SECONDS["MID"]
                )
                and cooldown_manager.ensure_single_alert(
                    "MID", mid["triggers"], price, "MID"
                )
            ):
                await self._dispatch_signal(symbol, "MID", mid, data)

            # ---------- TREND ----------
            trend = calculate_trend_signals(data)
            if (
                trend["score"] >= TREND_MIN_SCORE
                and trend["direction"] != "none"
                and cooldown_manager.can_send(
                    symbol, "TREND", COOLDOWN_SECONDS["TREND"]
                )
                and cooldown_manager.ensure_single_alert(
                    "TREND", trend["triggers"], price, "TREND"
                )
            ):
                await self._dispatch_signal(symbol, "TREND", trend, data)

        except Exception as e:
            logger.exception(f"Error analysing {symbol}: {e}")

    # ---------------- send one signal ----------------
    async def _dispatch_signal(self, symbol: str, mode: str, result: dict, data: dict):
        try:
            msg, code = telegram_formatter_style_c(
                symbol, mode, result["direction"], result, data
            )
            await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
            await send_copy_block(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, code)
            logger.info(f"📤 {mode} {result['direction'].upper()} signal sent for {symbol}")
        except Exception as e:
            logger.exception(f"Telegram dispatch failed: {e}")

    # ---------------- main loop ----------------
    async def run(self) -> None:
        logger.info("🚀 30-Logic Signal-Bot started!")
        await send_telegram_message(
            TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, "🤖 <b>30-Logic Signal-Bot is ONLINE</b>"
        )

        while not shutdown_event.is_set():
            try:
                for sym in TRADING_PAIRS:
                    await self.analyse_symbol(sym)
                    await asyncio.sleep(1)  # tiny breath between symbols
                await asyncio.sleep(8)  # ~10 s total per round
            except Exception as e:
                logger.exception("Outer loop error – retrying in 5 s")
                await asyncio.sleep(5)

        logger.info("Bot loop terminated.")
        await self.market.close()

# ---------- entry ----------
if __name__ == "__main__":
    # Railway provides PORT env var – bind to keep container alive
    port = int(os.getenv("PORT", 8080))

    async def dummy_server():
        """Dummy async server to bind PORT (Railway requirement)"""
        from aiohttp import web

        async def handle(_):
            return web.Response(text="30-Logic Signal-Bot is running\n")

        app = web.Application()
        app.router.add_get("/", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        logger.info(f"Dummy web-server bound to port {port}")

    async def main():
        await asyncio.gather(
            dummy_server(),
            SignalBot().run(),
        )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard exit")
