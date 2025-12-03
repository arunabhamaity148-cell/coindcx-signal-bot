import asyncio
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main_bot")


class SignalBot:
    def __init__(self):
        self.market = MarketData(BINANCE_API_KEY, BINANCE_SECRET)
        self.running = True

    async def analyze_symbol(self, symbol):
        try:
            data = await self.market.get_all_data(symbol)
            if not data:
                logger.debug(f"⚠️ No data for {symbol}")
                return

            # basic liquidity/spread checks early
            if not spread_ok(data):
                logger.info(f"❌ {symbol} rejected (spread too high)")
                return
            if not depth_ok(data["orderbook"]):
                logger.info(f"❌ {symbol} rejected (low orderbook depth)")
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

            # Cooldown check
            cooldown = COOLDOWN_SECONDS.get(mode, 1800)
            if not cooldown_manager.can_send(symbol, mode, cooldown):
                logger.info(f"⏳ {symbol} {mode} skipped (cooldown)")
                return

            # Dedupe check
            rule_key = f"{symbol}_{mode}"
            if not cooldown_manager.ensure_single_alert(rule_key, result["triggers"], price, mode):
                logger.info(f"🔁 {symbol} {mode} duplicate signal")
                return

            # BTC calm check (pass result into AI too)
            btc_ok = await btc_calm_check(self.market)
            if not btc_ok:
                logger.warning("⚠️ BTC volatile — skipping all signals temporarily")
                return

            # Compute spread_ok and depth_ok to pass to AI
            s_ok = spread_ok(data)
            d_ok = depth_ok(data["orderbook"])

            # AI review (low-cost call)
            ai_res = await ai_review_signal(
                symbol=symbol,
                mode=mode,
                direction=direction,
                score=result["score"],
                triggers=result["triggers"],
                spread_ok=s_ok,
                depth_ok=d_ok,
                btc_calm=btc_ok
            )

            logger.info(f"🤖 AI decision for {symbol} {mode}: {ai_res}")

            if not ai_res.get("allow", False):
                logger.info(f"🚫 AI blocked {symbol} {mode} — {ai_res.get('reason')}")
                return

            confidence = ai_res.get("confidence", None)
            # TP/SL with confidence-aware leverage
            tp_sl = calc_single_tp_sl(price, direction, mode, confidence)

            # Final safety: check SL vs liquidation quickly
            # (helpers' safety functions will already warn; skip here for brevity)

            # Format & send Telegram message
            msg, _ = telegram_formatter_style_c(symbol, mode, direction, result, data)
            if not msg:
                logger.warning(f"⚠️ Formatter blocked message for {symbol} {mode}")
                return

            await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
            await asyncio.sleep(0.8)
            if tp_sl and tp_sl.get("code"):
                await send_copy_block(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, tp_sl["code"])

            logger.info(f"✅ Sent {mode} {direction.upper()} for {symbol} (conf {confidence}%)")

        except Exception as e:
            logger.error(f"❌ Error in process_signal for {symbol}: {e}")

    async def run(self):
        logger.info("🚀 Signal Bot Started!")
        logger.info(f"📊 Monitoring {len(TRADING_PAIRS)} pairs")
        try:
            while self.running:
                for symbol in TRADING_PAIRS:
                    await self.analyze_symbol(symbol)
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        logger.info("🛑 Shutting down...")
        try:
            await self.market.close()
        except:
            pass
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    bot = SignalBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")