import asyncio
import logging
import time
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
    save_pending_signal,
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
    AUTO_PUBLISH,
    ASSISTANT_CONTROLLED_PUBLISH,
    AI_MIN_CONFIDENCE_TO_SEND,
    GLOBAL_SYMBOL_COOLDOWN,
    SUGGESTED_LEVERAGE,
    SCORE_FOR_50X,
    PENDING_SIGNALS_FILE,
    MAX_TELEGRAM_PER_MINUTE,
    MAX_TELEGRAM_PER_HOUR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main_bot")


class SignalBot:
    def __init__(self):
        self.market = MarketData(BINANCE_API_KEY, BINANCE_SECRET)
        self.running = True
        # rate limiter trackers
        self.tg_rate = {"by_min": {}, "by_hour": {}}
        # per-symbol last publish time
        self._global_symbol_last = {}

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
            logger.exception(f"Error analyzing {symbol}: {e}")

    async def process_signal(self, symbol, mode, result, data):
        try:
            direction = result.get("direction", "none")
            price = data.get("price")

            if direction == "none":
                return

            # Mode cooldown
            cooldown = COOLDOWN_SECONDS.get(mode, 1800)
            if not cooldown_manager.can_send(symbol, mode, cooldown):
                logger.info(f"⏳ {symbol} {mode} skipped (mode cooldown)")
                return

            # Dedupe identical content
            rule_key = f"{symbol}_{mode}"
            if not cooldown_manager.ensure_single_alert(rule_key, result.get("triggers", ""), price, mode):
                logger.info(f"🔁 {symbol} {mode} duplicate signal")
                return

            # Spread/depth quick reject
            s_ok = spread_ok(data)
            d_ok = depth_ok(data.get("orderbook", {}))
            if not s_ok or not d_ok:
                logger.info(f"❌ {symbol} rejected (spread/depth fail) s_ok={s_ok} d_ok={d_ok}")
                return

            # BTC calm check
            btc_ok = await btc_calm_check(self.market)
            if not btc_ok:
                logger.warning("⚠️ BTC volatile — skipping signals temporarily")
                return

            # AI review
            ai_res = await ai_review_signal(
                symbol=symbol,
                mode=mode,
                direction=direction,
                score=result.get("score", 0),
                triggers=result.get("triggers", ""),
                spread_ok=s_ok,
                depth_ok=d_ok,
                btc_calm=btc_ok
            )
            logger.info(f"🤖 AI decision for {symbol} {mode}: {ai_res}")

            # If AI denies -> stop
            if not ai_res.get("allow", False):
                logger.info(f"🚫 AI rejected {symbol} {mode} — {ai_res.get('reason')}")
                return

            # Enforce confidence threshold
            conf = int(ai_res.get("confidence", 0))
            if conf < AI_MIN_CONFIDENCE_TO_SEND:
                logger.info(f"🚫 Not sending {symbol} {mode}: AI conf {conf} < {AI_MIN_CONFIDENCE_TO_SEND}")
                return

            # Prevent multi-mode publishes for same symbol
            now_ts = datetime.utcnow().timestamp()
            last_ts = self._global_symbol_last.get(symbol)
            if last_ts and (now_ts - last_ts) < GLOBAL_SYMBOL_COOLDOWN:
                logger.info(f"⏳ {symbol} recently published ({now_ts-last_ts:.0f}s) — skipping {mode}")
                return

            # Decide leverage (score-based)
            score_val = float(result.get("score", 0))
            if score_val >= SCORE_FOR_50X:
                leverage_use = 50
            else:
                lev_sugg = SUGGESTED_LEVERAGE.get(mode, 20)
                leverage_use = max(15, min(30, int(lev_sugg)))

            # Calculate TP/SL (support both calc_single_tp_sl returns)
            tp, sl, codeblock = None, None, None
            try:
                tp_sl = calc_single_tp_sl(price, direction, mode)
                if isinstance(tp_sl, dict):
                    tp = tp_sl.get("tp")
                    sl = tp_sl.get("sl")
                    codeblock = tp_sl.get("code")
                elif isinstance(tp_sl, (list, tuple)) and len(tp_sl) >= 2:
                    tp, sl = tp_sl[0], tp_sl[1]
                else:
                    # fallback: try tuple return (tp, sl, lev, code)
                    try:
                        tp, sl, _, codeblock = calc_single_tp_sl(price, direction, mode)
                    except Exception:
                        tp, sl, codeblock = None, None, None
            except Exception:
                logger.debug("calc_single_tp_sl primary parsing failed, attempted fallback", exc_info=True)
                try:
                    tp, sl, _, codeblock = calc_single_tp_sl(price, direction, mode)
                except Exception:
                    tp, sl, codeblock = None, None, None

            # Format message (USD)
            msg, code_msg = telegram_formatter_style_c(symbol, mode, direction, result, data)
            # Ensure leverage in codeblock if exists
            if code_msg and isinstance(code_msg, str):
                # try to replace placeholder or add line with LEVERAGE
                if "LEVERAGE" in code_msg:
                    code_msg = code_msg.replace("LEVERAGE = ", f"LEVERAGE = {leverage_use}")
                else:
                    code_msg = (code_msg + f"\n```python\nLEVERAGE = {leverage_use}\n```")

            # Auto-send or save candidate
            if AUTO_PUBLISH and not ASSISTANT_CONTROLLED_PUBLISH:
                # Rate limiter check
                minute_key = int(time.time()) // 60
                hour_key = int(time.time()) // 3600
                # cleanup old keys (keep only current window)
                self.tg_rate["by_min"] = {k: v for k, v in self.tg_rate["by_min"].items() if k == minute_key}
                self.tg_rate["by_hour"] = {k: v for k, v in self.tg_rate["by_hour"].items() if k == hour_key}

                min_count = self.tg_rate["by_min"].get(minute_key, 0)
                hour_count = self.tg_rate["by_hour"].get(hour_key, 0)

                if min_count >= MAX_TELEGRAM_PER_MINUTE:
                    logger.warning(f"🔇 Telegram per-minute limit reached ({min_count} >= {MAX_TELEGRAM_PER_MINUTE}) — skipping send")
                    return
                if hour_count >= MAX_TELEGRAM_PER_HOUR:
                    logger.warning(f"🔇 Telegram per-hour limit reached ({hour_count} >= {MAX_TELEGRAM_PER_HOUR}) — skipping send")
                    return

                # send
                await send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
                await asyncio.sleep(0.6)
                if code_msg:
                    await send_copy_block(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, code_msg)

                # update counters & last published
                self.tg_rate["by_min"][minute_key] = self.tg_rate["by_min"].get(minute_key, 0) + 1
                self.tg_rate["by_hour"][hour_key] = self.tg_rate["by_hour"].get(hour_key, 0) + 1
                self._global_symbol_last[symbol] = now_ts

                logger.info(f"✅ Auto-published {mode} {direction} for {symbol} (conf {conf}%)")
            else:
                # Save candidate for manual review / assisted publish
                candidate = {
                    "symbol": symbol,
                    "mode": mode,
                    "direction": direction,
                    "score": result.get("score"),
                    "triggers": result.get("triggers"),
                    "price_usd": price,
                    "tp_usd": tp,
                    "sl_usd": sl,
                    "ai": ai_res,
                    "saved_at": datetime.utcnow().isoformat()
                }
                # use symbol_mode key for dedupe & easy lookup
                saved = save_pending_signal(f"{symbol}_{mode}", candidate)
                if saved:
                    self._global_symbol_last[symbol] = now_ts
                    logger.info(f"💾 Candidate saved for manual review: {symbol} {mode} (conf {conf}%)")
                else:
                    logger.error("❌ Failed to save candidate")

        except Exception as e:
            logger.exception(f"❌ Error in process_signal for {symbol}: {e}")

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
            logger.exception(f"Main loop error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        logger.info("🛑 Shutting down...")
        try:
            await self.market.close()
        except Exception:
            pass
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    bot = SignalBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")