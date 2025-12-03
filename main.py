# ==========================================
# main.py — FINAL PRO VERSION
# ==========================================

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
    spread_ok,
    depth_ok,
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
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# Logging Config
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("main_bot")


# ==========================================
# Signal Bot Class
# ==========================================
class SignalBot:
    def __init__(self):
        self.market = MarketData(BINANCE_API_KEY, BINANCE_SECRET)
        self.running = True

    # --------------------------------------
    async def analyze_symbol(self, symbol):
        """Fetch market data → run all signals → process"""
        try:
            data = await self.market.get_all_data(symbol)
            if not data:
                logger.warning(f"⚠️ No data for {symbol}")
                return

            # -------- Safety Layer --------
            if not spread_ok(data):
                logger.info(f"❌ {symbol} rejected (spread too high)")
                return

            if not depth_ok(data["orderbook"]):
                logger.info(f"❌ {symbol} rejected (low orderbook depth)")
                return

            # -------- Calculate Signals --------
            quick = calculate_quick_signals(data)
            mid = calculate_mid_signals(data)
            trend = calculate_trend_signals(data)

            # -------- Process Signals --------
            if quick["score"] >= QUICK_MIN_SCORE and quick["direction"] != "none":
                await self.process_signal(symbol, "QUICK", quick, data)

            if mid["score"] >= MID_MIN_SCORE and mid["direction"] != "none":
                await self.process_signal(symbol, "MID", mid, data)

            if trend["score"] >= TREND_MIN_SCORE and trend["direction"] != "none":
                await self.process_signal(symbol, "TREND", trend, data)

        except Exception as e:
            logger.error(f"❌ Analyze error for {symbol}: {e}")

    # --------------------------------------
    async def process_signal(self, symbol, mode, result, data):
        """Final safety + cooldown + dedupe + Telegram send"""
        try:
            direction = result["direction"]
            entry = data["price"]

            # 1) Cooldown
            cooldown = COOLDOWN_SECONDS.get(mode, 1800)
            if not cooldown_manager.can_send(symbol, mode, cooldown):
                logger.info(f"⏳ {symbol} {mode} skipped (cooldown)")
                return

            # 2) Dedupe
            if not cooldown_manager.ensure_single_alert(
                f"{symbol}_{mode}", 
                result["triggers"],
                entry,
                mode
            ):
                logger.info(f"🔁 Duplicate {symbol} {mode} skipped")
                return

            # 3) BTC Calm Check
            if not await btc_calm_check(self.market):
                logger.warning(f"⚠️ BTC volatile → {symbol} {mode} ignored")
                return

            # 4) Format Telegram Message
            message, code_block = telegram_formatter_style_c(symbol, mode, direction, result, data)
            if not message:
                logger.warning(f"⚠️ SL too close to liquidation for {symbol} {mode}")
                return

            # 5) Send Telegram
            await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
            await asyncio.sleep(1)
            await send_copy_block(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, code_block)

            logger.info(f"✅ SENT {mode} {direction.upper()} → {symbol} @ {entry}")

        except Exception as e:
            logger.error(f"❌ Error in process_signal for {symbol}: {e}")

    # --------------------------------------
    async def run(self):
        logger.info("🚀 Signal Bot Online!")
        logger.info(f"📊 Tracking {len(TRADING_PAIRS)} pairs")

        # Startup Telegram Message
        try:
            await send_telegram_message(
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                "🤖 <b>Signal Bot Online (PRO Version)</b>\n"
                f"📈 Coins Loaded: {len(TRADING_PAIRS)}\n"
                f"⚡ Quick ≥ {QUICK_MIN_SCORE}\n"
                f"🔵 Mid ≥ {MID_MIN_SCORE}\n"
                f"🟣 Trend ≥ {TREND_MIN_SCORE}\n\n"
                "🔥 Systems Ready"
            )
        except:
            pass

        # Main Loop
        while self.running:
            try:
                for symbol in TRADING_PAIRS:
                    await self.analyze_symbol(symbol)
                    await asyncio.sleep(0.8)   # Coins faster = 50 coins = required

                await asyncio.sleep(5)

            except KeyboardInterrupt:
                logger.info("🛑 Bot Stopped Manually")
                self.running = False
                break

            except Exception as e:
                logger.error(f"Main Loop Error: {e}")
                await asyncio.sleep(3)

        await self.shutdown()

    # --------------------------------------
    async def shutdown(self):
        logger.info("🛑 Shutting Down...")
        try:
            await self.market.close()
            await send_telegram_message(
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                "🛑 <b>Signal Bot Stopped</b>"
            )
        except:
            pass
        logger.info("✅ Shutdown Complete")


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    bot = SignalBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye")