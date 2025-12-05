# main.py — FINAL PATCHED VERSION (BLOCK LOG FIXED)

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

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

bot_task = None
ping_task = None
ws_task = None
deploy_notify_task = None


# ---------------------------------------------------------------------
# KEEP ALIVE
# ---------------------------------------------------------------------
async def keep_alive():
    port = int(os.getenv("PORT", 8080))
    url = f"http://0.0.0.0:{port}/health"

    while True:
        try:
            await asyncio.sleep(60)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        log.info("✓ Health ping OK")
        except Exception as e:
            log.error(f"Keep-alive: {e}")


# ---------------------------------------------------------------------
# BOT LOOP — FULL BLOCK LOG VERSION
# ---------------------------------------------------------------------
async def bot_loop():
    log.info("🤖 Starting CoinDCX signal generator...")

    try:
        ex = Exchange()
        signal_count = 0
        await asyncio.sleep(8)

        log.info(f"✓ Bot initialized - Monitoring {len(CFG['pairs'])} pairs")

        while True:
            try:
                limit_ok, _ = await check_daily_signal_limit()
                if not limit_ok:
                    log.warning("⚠️ Daily signal limit reached (30) — sleeping 1h")
                    await asyncio.sleep(3600)
                    continue

                # ========== MAIN PAIR LOOP ==============
                for sym in CFG["pairs"]:
                    for strategy in ["QUICK", "MID", "TREND"]:

                        # cooldown
                        ok, cd_reason = await check_cooldown(sym, strategy)
                        if not ok:
                            log.debug(f"⏳ Cooldown [{strategy}] {sym}: {cd_reason}")
                            continue

                        # heuristic score
                        try:
                            signal = await calculate_advanced_score(sym, strategy)
                        except Exception as e:
                            log.error(f"Score error {sym} [{strategy}]: {e}")
                            continue

                        if not signal:
                            continue
                        if signal["side"] == "none":
                            continue

                        side = signal["side"]
                        score = signal["score"]
                        last = signal["last"]

                        # ML/AI check
                        ai = await ai_review_ensemble(sym, side, score, strategy)
                        if not ai["allow"]:
                            log.info(
                                f"🚫 BLOCKED [{strategy}] {sym} — {ai['reason']} | "
                                f"Conf={ai['confidence']} | Score={score:.2f}"
                            )
                            continue

                        confidence = ai["confidence"]

                        # TP/SL
                        try:
                            tp1, tp2, sl, leverage, liq_price = await calc_tp1_tp2_sl_liq(
                                sym, side, last, confidence, strategy
                            )
                        except Exception as e:
                            log.error(f"TP/SL error {sym}: {e}")
                            continue

                        # RRR
                        risk = abs(last - sl)
                        reward = abs(tp1 - last)
                        rrr = reward / risk if risk > 0 else 0

                        if rrr < 1.0:
                            log.info(f"🚫 BLOCKED [{strategy}] {sym} — RRR low ({rrr:.2f})")
                            continue

                        # ORDERFLOW — FIXED IMPORT
                        from helpers import orderflow_analysis
                        flow = await orderflow_analysis(sym)

                        if not flow:
                            log.info(f"🚫 BLOCKED [{strategy}] {sym} — no-flow")
                            continue

                        if flow["spread"] > 0.25:
                            log.info(f"🚫 BLOCKED [{strategy}] {sym} — spread {flow['spread']:.3f}%")
                            continue

                        if flow["depth_usd"] < 300000:
                            log.info(
                                f"🚫 BLOCKED [{strategy}] {sym} — depth low ({flow['depth_usd']})"
                            )
                            continue

                        # position size
                        pos = position_size_iceberg(CFG["equity"], last, sl, leverage)
                        if pos["total_qty"] < 0.001:
                            log.info(f"🚫 BLOCKED [{strategy}] {sym} — pos too small")
                            continue

                        # ---------- SIGNAL APPROVED ----------
                        signal_count += 1
                        await increment_signal_count()

                        log.info(
                            f"✅ SIGNAL [{strategy}] {sym} {side.upper()} #{signal_count} "
                            f"Entry={last:.6f} | TP1={tp1:.6f} | SL={sl:.6f}"
                        )

                        # telegram
                        msg = (
                            f"🎯 <b>[{strategy}] {sym} {side.upper()} #{signal_count}</b>\n"
                            f"Entry: <code>{last:.2f}</code>\n"
                            f"TP1: <code>{tp1:.2f}</code>\n"
                            f"TP2: <code>{tp2:.2f}</code>\n"
                            f"SL: <code>{sl:.2f}</code>\n"
                            f"Lev: {leverage}x\n"
                            f"Score: {score:.2f}\n"
                            f"Conf: {confidence}%\n"
                            f"RRR: {rrr:.2f}\n"
                        )

                        await send_telegram(msg)
                        await set_cooldown(sym, strategy)
                        break

                await asyncio.sleep(3)

            except Exception as e:
                log.error(f"Bot loop error: {e}")
                await asyncio.sleep(5)

    except Exception as e:
        log.exception(f"FATAL BOT CRASH: {e}")


# ---------------------------------------------------------------------
# DEPLOY NOTIFICATION
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# LIFESPAN MANAGER
# ---------------------------------------------------------------------
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
        log.info("🛑 Shutting down tasks...")
        for t in [ws_task, bot_task, ping_task, deploy_notify_task]:
            if t:
                t.cancel()
        await asyncio.gather(*[t for t in [ws_task, bot_task, ping_task, deploy_notify_task] if t], return_exceptions=True)
        await cleanup()
        log.info("✓ Shutdown complete")


# ---------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "running", "pairs": len(CFG["pairs"])}

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------
# RUN UVICORN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))