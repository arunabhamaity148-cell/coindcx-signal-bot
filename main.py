# main.py — uses helpers.py, pure-logic signals, telegram notify
import os, asyncio, logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
import uvicorn

from helpers import WS, calc_score, calc_tp1_tp2_sl_liq, position_size_iceberg, send_signal_telegram, send_telegram, check_cooldown, set_cooldown, increment_signal_count, check_daily_signal_limit, cleanup, CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

bot_task = None
ws_task = None
ping_task = None
deploy_notify_task = None

async def keep_alive():
    port = int(os.getenv("PORT", 8080))
    url = f"http://0.0.0.0:{port}/health"
    import aiohttp
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        log.debug("Health ping ok")
        except Exception:
            pass
        await asyncio.sleep(60)

async def bot_loop():
    log.info("🤖 Starting Signal Bot (Aggressive Mode)...")
    await asyncio.sleep(5)
    await send_telegram("🚀 <b>Bot Deployed — Aggressive Logic (No ML)</b>")

    while True:
        try:
            limit_ok, reason = await check_daily_signal_limit()
            if not limit_ok:
                log.warning(f"Daily limit reached: {reason}")
                await asyncio.sleep(3600)
                continue

            for sym in CFG["top_pairs"]:
                try:
                    cd_ok, cd_reason = await check_cooldown(sym, "GLOBAL")
                    if not cd_ok:
                        log.debug(f"{sym} cooldown: {cd_reason}")
                        continue

                    sc = await calc_score(sym)
                    if sc["side"] == "none":
                        log.debug(f"{sym} blocked: {sc['reason']}")
                        continue

                    side = sc["side"]
                    score = sc["score"]
                    last = sc["last"]

                    # TP/SL calculation & liq distance
                    tp1, tp2, sl, lev, liq_price, liq_dist_pct = await calc_tp1_tp2_sl_liq(sym, side, last, confidence=60, strategy="QUICK")

                    # iceberg sizing
                    iceberg = position_size_iceberg(CFG["equity"], last, sl, lev)
                    if iceberg["total_qty"] < 0.0005:
                        log.info(f"{sym} BLOCKED: position too small ({iceberg['total_qty']})")
                        continue

                    # liq distance alarm: if too close, warn and skip sending normal signal
                    if liq_dist_pct < float(os.getenv("LIQ_ALERT_PCT", CFG.get("liq_alert_pct", 0.7))):
                        log.warning(f"LIQ CLOSE {sym}: {liq_dist_pct:.2f}% — skipping signal")
                        await send_telegram(f"⚠️ LIQ CLOSE ALERT: {sym} | Entry={last} | LiqDist={liq_dist_pct:.2f}%")
                        # you can choose to continue or still send with warning; we skip to be safe
                        continue

                    # final signal send
                    await increment_signal_count()
                    await send_signal_telegram(sym, "HYBRID", side, last, tp1, tp2, sl, lev, iceberg_info=iceberg, liq_dist_pct=liq_dist_pct, extra=f"Score:{score}")
                    await set_cooldown(sym, "GLOBAL")
                    log.info(f"✅ SIGNAL {sym} {side} Score={score} Lev={lev} LiqDist={liq_dist_pct:.2f}%")
                    # aggressive mode cadence: small sleep between symbols
                    await asyncio.sleep(0.6)

                except Exception as e:
                    log.error(f"Symbol loop error {sym}: {e}")
                    continue

            await asyncio.sleep(1)

        except Exception as e:
            log.error(f"Bot loop error: {e}")
            await asyncio.sleep(5)

async def send_deploy_notify():
    await asyncio.sleep(3)
    try:
        await send_telegram(f"🚀 Bot active — pairs: {len(CFG['top_pairs'])} — Mode: Aggressive (No ML) — Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    except Exception as e:
        log.error(f"Deploy notify error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ws_task, bot_task, ping_task, deploy_notify_task
    log.info("Starting services...")
    ws = WS()
    ws_task = asyncio.create_task(ws.run())
    bot_task = asyncio.create_task(bot_loop())
    ping_task = asyncio.create_task(keep_alive())
    deploy_notify_task = asyncio.create_task(send_deploy_notify())
    try:
        yield
    finally:
        log.info("Shutting down...")
        for t in [ws_task, bot_task, ping_task, deploy_notify_task]:
            if t:
                t.cancel()
        await asyncio.gather(*[t for t in [ws_task, bot_task, ping_task, deploy_notify_task] if t], return_exceptions=True)
        await cleanup()
        log.info("Shutdown complete")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status":"running","pairs":len(CFG["top_pairs"])}

@app.get("/health")
def health():
    return {"status":"ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))