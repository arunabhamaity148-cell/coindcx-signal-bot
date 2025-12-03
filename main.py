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
    calc_single_tp_sl,
)

from ai_layer import ai_review_signal

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("main_bot")


class SignalBot:
    def __init__(self):
        self.market = MarketData(BINANCE_API_KEY, BINANCE_SECRET)
        self.running = True

    async def analyze_symbol(self, symbol):
        try:
            data = await self.market.get_all_data(symbol)
            if not data:
                return

            quick = calculate_quick_signals(data)
            mid = calculate_mid_signals(data)
            trend = calculate_trend_signals(data)

            if quick["score"] >= QUICK_MIN_SCORE and quick["direction"] != "none":
                await self.process_signal(symbol, "QUICK", quick, data)

            if mid["score"] >= MID_MIN_SCORE and mid["direction"] != "none":
                await self.process_signal(symbol, "MID", mid, data)

            if trend["score"] >= TREND_MIN_SCORE and trend["direction"] != "none":
                await self.process_signal(symbol, "TREND", trend, data)

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    async def process_signal(self, symbol, mode, result, data):
        try:
            direction = result["direction"]
            price = data["price"]

            # Cooldown
            cooldown = COOLDOWN_SECONDS.get(mode, 1800)
            if not cooldown_manager.can_send(symbol, mode, cooldown):
                logger.info(f"⏳ {symbol} {mode} skipped (cooldown)")
                return

            # BTC calm
            if not await btc_calm_check(self.market):
                logger.warning("⚠️ BTC volatile — skipping signal")
                return

            # AI review layer
            ai_res = await ai_review_signal(
                symbol=symbol,
                mode=mode,
                direction=direction,
                score=result["score"],
                triggers=result["triggers"],
                price=price
            )

            if not ai_res.get("allow", False):
                logger.info(f"🤖 AI BLOCKED {symbol} {mode}: {ai_res['reason']}")
                return

            confidence = ai_res.get("confidence")

            # TP/SL
            tp_sl = calc_single_tp_sl(price, direction, mode, confidence)
            tp = tp_sl["tp"]
            sl = tp_sl["sl"]
            lev = tp_sl["leverage"]

            # FORMAT + SEND
            msg, code_block = telegram_formatter_style_c(symbol, mode, direction, result, data)

            await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
            await asyncio.sleep(1)
            await send_copy_block(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, tp_sl["code"])

            logger.info(f"✅ SENT {mode} {direction.upper()} for {symbol}")

        except Exception as e:
            logger.error(f"❌ Error in process_signal for {symbol}: {e}")

    async def run(self):
        logger.info("🚀 Signal Bot Started!")
        logger.info(f"📊 Tracking {len(TRADING_PAIRS)} pairs")

        while self.running:
            try:
                for symbol in TRADING_PAIRS:
                    await self.analyze_symbol(symbol)
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)

    async def shutdown(self):
        await self.market.close()


if __name__ == "__main__":
    bot = SignalBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually")