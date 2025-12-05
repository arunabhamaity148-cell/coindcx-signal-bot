# main.py — FINAL STABLE VERSION

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
import uvicorn
import aiohttp

from helpers import (
    WS, Exchange, calculate_advanced_score, ai_review_ensemble,
    calc_tp1_tp2_sl_liq, position_size_iceberg, send_telegram,
    check_cooldown, set_cooldown, check_daily_signal_limit,
    increment_signal_count, cleanup, CFG, STRATEGY_CONFIG
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

bot_task = None
ping_task = None
ws_task = None
deploy_notify_task = None

# ---------------------------------------------------------
# HEALTH KEEPALIVE
# ---------------------------------------------------------
async def keep_alive():
    port = int(os.getenv("PORT", 8080))
    url = f"http://0.0.0.0:{port}/health"

    while True:
        try:
            await asyncio.sleep(60)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        log.info("✓ Health ping OK")
        except Exception as e:
            log.error(f"Keep-alive: {e}")


# ---------------------------------------------------------
# BOT LOOP — with full BLOCK LOGS
# ---------------------------------------------------------
async def bot_loop():
    log.info("🤖 Starting CoinDCX signal generator...")

    try:
        ex = Exchange()
        signal_count = 0

        await asyncio.sleep(8)
        log.info(f"✓ Bot initialized - Monitoring {len(CFG['pairs'])} pairs")

        while True:
            try:
                # Daily limit
                limit_ok, limit_reason = await check_daily_signal_limit()
                if not limit_ok:
                    log.warning(f"⚠️ Daily limit reached: {limit_reason} - sleeping 1h")
                    await asyncio.sleep(3600)
                    continue

                # Loop pairs
                for sym in CFG["pairs"]:
                    try:
                        for strategy in ["QUICK", "MID", "TREND"]:
                            try:
                                # Cooldown check
                                cooldown_ok, cd_reason = await check_cooldown(sym, strategy)
                                if not cooldown_ok:
                                    log.debug(f"⏳ Cooldown active [{strategy}] {sym}: {cd_reason}")
                                    continue

                                # Calculate heuristic score
                                signal = await calculate_advanced_score(sym, strategy)
                                if not signal:
                                    log.debug(f"🚫 No heuristic score for {sym} [{strategy}]")
                                    continue
                                if signal["side"] == "none":
                                    log.debug(f"⛔ No side for {sym} [{strategy}] (score={signal['score']})")
                                    continue

                                side = signal["side"]
                                score = signal["score"]
                                last = signal["last"]

                                # AI/ML review
                                ai = await ai_review_ensemble(sym, side, score, strategy)
                                if not ai["allow"]:
                                    log.info(
                                        f"🚫 BLOCKED [{strategy}] {sym} — Reason: {ai['reason']} "
                                        f"| Conf={ai['confidence']} | Score={score:.2f}"
                                    )
                                    continue

                                confidence = ai["confidence"]

                                # TP / SL / Leverage calc
                                tp1, tp2, sl, leverage, liq_price = await calc_tp1_tp2_sl_liq(
                                    sym, side, last, confidence, strategy
                                )

                                # RRR check
                                risk = abs(last - sl)
                                reward1 = abs(tp1 - last)
                                reward2 = abs(tp2 - last)
                                rrr1 = reward1 / risk if risk > 0 else 0
                                rrr2 = reward2 / risk if risk > 0 else 0

                                if rrr1 < 1.0:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — RRR low ({rrr1:.2f})")
                                    continue

                                # Orderflow safety check
                                from helpers import orderflow_analysis
                                flow = await orderflow_analysis(sym)
                                if not flow:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — no-flow")
                                    continue

                                if flow["spread"] > 0.25:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — spread {flow['spread']:.3f}%")
                                    continue
                                if flow["depth_usd"] < 300000:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — low depth {flow['depth_usd']}")
                                    continue

                                # Position sizing
                                pos = position_size_iceberg(CFG["equity"], last, sl, leverage)
                                if pos["total_qty"] < 0.001:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — pos too small")
                                    continue

                                # Approved → SEND SIGNAL
                                signal_count += 1
                                await increment_signal_count()

                                log.info(
                                    f"✅ SIGNAL [{strategy}] {sym} {side.upper()} #{signal_count}"
                                    f" | Entry={last:.6f} | TP1={tp1:.6f} | SL={sl:.6f}"
                                )

                                # Telegram message
                                msg = (
                                    f"🎯 <b>[{strategy}] {sym} {side.upper()} #{signal_count}</b>\n"
                                    f"━━━━━━━━━━━━━━━━━\n"
                                    f"Entry: <code>{last:.2f}</code>\n"
                                    f"TP1: <code>{tp1:.2f}</code>\n"
                                    f"TP2: <code>{tp2:.2f}</code>\n"
                                    f"SL: <code>{sl:.2f}</code>\n"
                                    f"Leverage: {leverage}x\n"
                                    f"Score: {score:.2f}\n"
                                    f"Confidence: {confidence}%\n"
                                    f"RRR: {rrr1:.2f}\n"
                                    f"Time: {datetime.utcnow().strftime('%H:%M UTC')}\n"
                                )
                                await send_telegram(msg)

                                await set_cooldown(sym, strategy)
                                break

                            except Exception as e:
                                log.error(f"Strategy {strategy} error {sym}: {e}")
                                continue

                    except Exception as e:
                        log.error(f"Symbol loop error {sym}: {e}")
                        continue

                await asyncio.sleep(3)

            except Exception as e:
                log.error(f"Bot loop iteration error: {e}")
                await asyncio.sleep(10)

    except Exception as e:
        log.exception(f"BOT FATAL ERROR: {e}")


# ---------------------------------------------------------
# DEPLOY STARTUP TELEGRAM
# ---------------------------------------------------------
async def send_deploy_telegram():
    await asyncio.sleep(5)
    try:
        msg = (
            f"🚀 <b>Bot Deployed</b>\n"
            f"Env: production\n"
            f"Pairs: {len(CFG['pairs'])}\n"
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        await send_telegram(msg)
        log.info("📨 Deploy message sent.")
    except Exception as e:
        log.error(f"Deploy telegram error: {e}")


# ---------------------------------------------------------
# FASTAPI LIFESPAN
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ws_task, bot_task, ping_task, deploy_notify_task

    log.info("🚀 Starting services...")

    ws = WS()
    ws_task = asyncio.create_task(ws.run())
    bot_task = asyncio.create_task(bot_loop())
    ping_task = asyncio.create_task(keep_alive())
    deploy_notify_task = asyncio.create_task(send_deploy_telegram())

    try:
        yield
    finally:
        log.info("🛑 Shutting down...")
        for t in [ws_task, bot_task, ping_task, deploy_notify_task]:
            if t:
                t.cancel()
        await asyncio.gather(*[t for t in [ws_task, bot_task, ping_task, deploy_notify_task] if t], return_exceptions=True)
        await cleanup()
        log.info("✓ Shutdown complete")


# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "running", "pairs": len(CFG["pairs"])}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))