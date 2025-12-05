# ================================================================
# main.py — FULL BOT ENGINE (Scanner + Telegram + Cooldown)
# ================================================================

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta

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
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 10))   # default 10 sec
COOLDOWN_MIN = 30

cooldown = {}      # "BTCUSDT:QUICK": datetime
ACTIVE_ORDERS = {} # store active signal positions


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

    await send_telegram("🚀 <b>Binance Scanner LIVE (v1.0)</b>\n📡 Waiting for real-time trades...")

    while True:
        try:
            results = []

            for sym in PAIRS:
                # evaluate 3 strategies
                for strat in ["QUICK", "MID", "TREND"]:
                    if not cooldown_ok(sym, strat):
                        continue

                    sig = await compute_signal(sym, strat)
                    if sig:
                        results.append(sig)

            # sort by score
            results.sort(key=lambda x: x["score"], reverse=True)

            if results:
                best = results[:3]  # send top 3

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
# FASTAPI + LIFESPAN
# -----------------------------------------------------------
app = FastAPI()
scan_task = None

@app.on_event("startup")
async def start_bot():
    global scan_task
    log.info("🚀 App starting (Production Bot v1.0)")
    log.info(f"✓ Loaded {len(PAIRS)} pairs")

    scan_task = asyncio.create_task(scanner())

@app.on_event("shutdown")
async def close_bot():
    if scan_task:
        scan_task.cancel()


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


# -----------------------------------------------------------
# RUN SERVER
# -----------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))