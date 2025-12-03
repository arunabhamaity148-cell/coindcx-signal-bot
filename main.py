# ==========================================
# main.py — Final PRO (with AI Decision Layer)
# ==========================================

import asyncio
import os
import logging
from datetime import datetime

from dotenv import load_dotenv

# helpers + decision layer + config imports
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
    calc_liquidation
)
from ai_decision_layer import decision_layer_v2

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

load_dotenv()

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("main_bot")


# -------------------------
# Signal Bot
# -------------------------
class SignalBot:
    def __init__(self):
        self.market = MarketData(BINANCE_API_KEY, BINANCE_SECRET)
        self.running = True

    async def analyze_symbol(self, symbol):
        """Fetch market data → run all signals → process"""
        try:
            data = await self.market.get_all_data(symbol)
            if not data:
                logger.warning(f"⚠️ No data for {symbol}")
                return

            # Safety: spread + depth
            if not spread_ok(data):
                logger.info(f"❌ {symbol} rejected (spread too high)")
                return

            if not depth_ok(data["orderbook"]):
                logger.info(f"❌ {symbol} rejected (low orderbook depth)")
                return

            # Calculate signals
            quick = calculate_quick_signals(data)
            mid = calculate_mid_signals(data)
            trend = calculate_trend_signals(data)

            # Process signals based on thresholds
            if quick["score"] >= QUICK_MIN_SCORE and quick["direction"] != "none":
                await self.process_signal(symbol, "QUICK", quick, data)

            if mid["score"] >= MID_MIN_SCORE and mid["direction"] != "none":
                await self.process_signal(symbol, "MID", mid, data)

            if trend["score"] >= TREND_MIN_SCORE and trend["direction"] != "none":
                await self.process_signal(symbol, "TREND", trend, data)

        except Exception as e:
            logger.error(f"❌ Analyze error for {symbol}: {e}")

    async def process_signal(self, symbol, mode, result, data):
        """Final safety + cooldown + dedupe + AI decision + Telegram send"""
        try:
            direction = result["direction"]
            price = data["price"]

            # 1) Cooldown check
            cooldown = COOLDOWN_SECONDS.get(mode, 1800)
            if not cooldown_manager.can_send(symbol, mode, cooldown):
                logger.info(f"⏳ {symbol} {mode} skipped (cooldown)")
                return

            # 2) Dedupe check
            rule_key = f"{symbol}_{mode}"
            if not cooldown_manager.ensure_single_alert(rule_key, result["triggers"], price, mode):
                logger.info(f"🔁 {symbol} {mode} duplicate signal")
                return

            # 3) BTC calm check (quick pre-check)
            btc_ok = await btc_calm_check(self.market)
            if not btc_ok:
                logger.warning(f"⚠️ BTC volatile, skipping {symbol} {mode} signal")
                return

            # 4) Initial TP/SL calculation (no confidence yet)
            initial = calc_single_tp_sl(price, direction, mode, confidence_pct=None)
            side_for_liq = "long" if initial["sl"] < price else "short"
            liq_info = calc_liquidation(price, initial["sl"], initial["leverage"], side_for_liq)

            # 5) Build payload for AI decision
            bids = data.get("orderbook", {}).get("bids", [])
            depth_val = sum([b[0] * b[1] for b in bids[:5]]) if bids else 0

            payload = {
                "symbol": symbol,
                "mode": mode,
                "price": price,
                "score": result["score"],
                "triggers": result["triggers"].split("\n") if isinstance(result["triggers"], str) else result["triggers"],
                "spread": data.get("spread", 0),
                "depth": depth_val,
                "btc_calm": btc_ok,
                "sl_distance_pct": liq_info.get("dist_pct", 0),
                "tp_value": initial["tp"],
                "sl_value": initial["sl"],
                "default_leverage": initial.get("leverage", 20)
            }

            # 6) Call AI Decision Layer
            decision = await decision_layer_v2(payload)

            # 7) Handle AI response
            if not decision or decision.get("decision") != "APPROVE":
                reason = decision.get("reason", "No reason")
                logger.info(f"❌ {symbol} {mode} rejected by AI: {reason}")
                return

            # 8) Apply AI recommendations
            recommended_lev = decision.get("recommended_leverage", initial.get("leverage", 20))
            tp_mod = decision.get("tp_modifier", 1.0)
            sl_mod = decision.get("sl_modifier", 1.0)
            confidence = decision.get("confidence", 0)

            # Recalculate final TP/SL using calc_single_tp_sl with confidence
            final_calc = calc_single_tp_sl(price, direction, mode, confidence_pct=confidence)
            # Override leverage with AI recommendation
            final_calc["leverage"] = recommended_lev
            final_tp = final_calc["tp"] * tp_mod
            final_sl = final_calc["sl"] * sl_mod

            # Extra high-lev safety: re-check liquidation distance
            side_for_liq = "long" if final_sl < price else "short"
            final_liq = calc_liquidation(price, final_sl, final_calc["leverage"], side_for_liq)
            if final_liq["dist_pct"] <  (5.0 if final_calc["leverage"] < 50 else 7.5):
                logger.warning(f"⚠️ {symbol} {mode} final SL too close to liq after AI changes")
                return

            # 9) Prepare message & code
            final_code = f"""```python
ENTRY = {price}
TP = {final_tp}
SL = {final_sl}
LEVERAGE = {recommended_lev}x
CONFIDENCE = {confidence}%
```"""

            message, _ = telegram_formatter_style_c(symbol, mode, direction, result, data)
            # append AI reason + metrics
            message += f"\n\n<b>AI_DECISION:</b> {decision.get('decision')} — {decision.get('reason')}\n"
            message += f"<b>AI_CONF:</b> {confidence}%  <b>LEV:</b> {recommended_lev}x\n"

            # 10) Send Telegram
            await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, message)
            await asyncio.sleep(1)
            await send_copy_block(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, final_code)

            logger.info(f"✅ {mode} {direction.upper()} AI-approved signal sent for {symbol} — CONF {confidence}% LEV {recommended_lev}x")

        except Exception as e:
            logger.error(f"❌ Error in process_signal for {symbol}: {e}")

    async def run(self):
        logger.info("🚀 Signal Bot Online!")
        logger.info(f"📊 Tracking {len(TRADING_PAIRS)} pairs")

        # Startup Telegram message
        try:
            await send_telegram_message(
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                "🤖 <b>Signal Bot + AI Decision Layer Online (PRO)</b>\n"
                f"📈 Pairs: {len(TRADING_PAIRS)}\n"
                f"⚡ Quick ≥ {QUICK_MIN_SCORE}\n"
                f"🔵 Mid ≥ {MID_MIN_SCORE}\n"
                f"🟣 Trend ≥ {TREND_MIN_SCORE}\n\n"
                "✅ System ready"
            )
        except Exception as e:
            logger.warning(f"Startup Telegram failed: {e}")

        # Main loop
        while self.running:
            try:
                for symbol in TRADING_PAIRS:
                    await self.analyze_symbol(symbol)
                    await asyncio.sleep(0.8)   # tuned for 50 coins

                await asyncio.sleep(5)

            except KeyboardInterrupt:
                logger.info("🛑 Bot Stopped Manually")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main Loop Error: {e}")
                await asyncio.sleep(3)

        await self.shutdown()

    async def shutdown(self):
        logger.info("🛑 Shutting Down...")
        try:
            await self.market.close()
            await send_telegram_message(
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                "🛑 <b>Signal Bot Stopped</b>"
            )
        except Exception as e:
            logger.warning(f"Shutdown message failed: {e}")

        logger.info("✅ Shutdown Complete")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    bot = SignalBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye")