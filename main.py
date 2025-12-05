# ================================================================
# main.py — FULL BOT ENGINE (Railway Production Ready)
# ================================================================

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI
import uvicorn

from helpers import redis, PAIRS, get_last_price
from scorer import compute_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# -----------------------------------------------------------
# ENV
# -----------------------------------------------------------
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 10))
COOLDOWN_MIN = 30

cooldown = {}
ACTIVE_ORDERS = {}


# -----------------------------------------------------------
# TELEGRAM SENDER
# -----------------------------------------------------------
async def send_telegram(msg):
    if not TG_TOKEN or not TG_CHAT:
        return

    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"}

    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload, timeout=10)
    except Exception as e:
        log.error(f"Telegram error: {e}")


# -----------------------------------------------------------
# COOLDOWN CHECK
# -----------------------------------------------------------
def cooldown_ok(sym, strat):
    key = f"{sym}:{strat}"
    if key not in cooldown:
        return True
    diff = (datetime.utcnow() - cooldown[key]).total_seconds() / 60
    return diff >= COOLDOWN_MIN


def set_cooldown(sym, strat):
    cooldown[f"{sym}:{strat}"] = datetime.utcnow()


# -----------------------------------------------------------
# SIGNAL FORMATTER
# -----------------------------------------------------------
def format_signal(sig):
    sym = sig["symbol"]
    side = sig["side"]
    score = sig["score"]
    last = sig["last"]
    strat = sig["strategy"]
    passed = " | ".join(sig["passed"])

    return (
        f"🎯 <b>{sym} {side.upper()} — [{strat}]</b>\n"
        f"💰 Price: <code>{last:.6f}</code>\n"
        f"📊 Score: <b>{score:.1f}</b>\n"
        f"🔍 Passed: {passed}\n"
        f"⏳ Cooldown: {COOLDOWN_MIN}m\n"
    )


# -----------------------------------------------------------
# MAIN SCANNER LOOP
# -----------------------------------------------------------
async def scanner():
    log.info("🔍 Scanner started")
    
    # Test Redis connection
    try:
        await redis.ping()
        log.info("✅ Redis connected successfully")
    except Exception as e:
        log.error(f"❌ Redis connection failed: {e}")
        await send_telegram("⚠️ <b>Bot Started but Redis NOT connected!</b>\nPlease check Railway Redis service.")
        return

    await send_telegram("🚀 <b>Binance Scanner LIVE (v1.0)</b>\n📡 Waiting for real-time trades...")

    while True:
        try:
            results = []

            for sym in PAIRS:
                for strat in ["QUICK", "MID", "TREND"]:
                    if not cooldown_ok(sym, strat):
                        continue

                    sig = await compute_signal(sym, strat)
                    if sig:
                        results.append(sig)

            results.sort(key=lambda x: x["score"], reverse=True)

            if results:
                best = results[:3]

                for sig in best:
                    msg = format_signal(sig)
                    await send_telegram(msg)

                    ACTIVE_ORDERS[f"{sig['symbol']}:{sig['strategy']}"] = sig
                    set_cooldown(sig["symbol"], sig["strategy"])

                    log.info(f"✔ SIGNAL: {sig['symbol']} {sig['strategy']} score={sig['score']:.1f}")

            else:
                log.info("📊 No valid signals")

            await asyncio.sleep(SCAN_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Scanner error: {e}")
            await asyncio.sleep(5)


# -----------------------------------------------------------
# FASTAPI + LIFESPAN (Railway Compatible)
# -----------------------------------------------------------
scan_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scan_task
    log.info("🚀 App starting (Production Bot v1.0)")
    log.info(f"✓ Loaded {len(PAIRS)} pairs")
    
    scan_task = asyncio.create_task(scanner())
    
    yield
    
    if scan_task:
        scan_task.cancel()
        try:
            await scan_task
        except asyncio.CancelledError:
            pass
    
    await redis.close()
    log.info("🛑 App shutdown complete")


app = FastAPI(lifespan=lifespan)


# -----------------------------------------------------------
# API ROUTES
# -----------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "running",
        "pairs": len(PAIRS),
        "cooldown": {k: str(v) for k, v in cooldown.items()},
        "active_orders": ACTIVE_ORDERS,
        "time": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health():
    try:
        await redis.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    return {
        "status": "ok",
        "redis": redis_status,
        "pairs": len(PAIRS)
    }


# -----------------------------------------------------------
# RUN SERVER
# -----------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))