# ============================================================
# main.py — HYBRID SCANNER + TELEGRAM BOT (BINANCE ONLY MODE)
# ============================================================

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from scorer import compute_score
from helpers import (
    redis_client,
    calc_tp_sl,
    iceberg_size,
    close_redis,
)

# ----------------------------
# CONFIG
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 30))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", 30))

PAIR_LIST = [
    "BTCUSDT","ETHUSDT","SOLUSDT","MATICUSDT","XRPUSDT",
    "BNBUSDT","DOGEUSDT","AVAXUSDT","ADAUSDT","DOTUSDT"
]

cooldown_map = {}
open_count = {"QUICK": 0, "MID": 0, "TREND": 0}
MAX_CONCURRENT = 3


# ----------------------------
# TELEGRAM
# ----------------------------
async def send_telegram(msg: str):
    if TG_TOKEN == "" or TG_CHAT == "":
        return
    import aiohttp
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as s:
        try:
            await s.post(url, json={
                "chat_id": TG_CHAT,
                "text": msg,
                "parse_mode": "HTML"
            })
        except:
            pass


# ----------------------------
# COOLDOWN
# ----------------------------
def cd_ok(sym: str, strat: str) -> bool:
    key = f"{sym}:{strat}"
    last = cooldown_map.get(key)
    if last is None:
        return True
    mins = (datetime.utcnow() - last).total_seconds() / 60
    return mins >= COOLDOWN_MIN


def set_cd(sym: str, strat: str):
    cooldown_map[f"{sym}:{strat}"] = datetime.utcnow()


# ----------------------------
# SCAN LOOP
# ----------------------------
async def scan_loop():
    await send_telegram("🚀 <b>Binance Premium Scalper LIVE</b>\nMode: Balanced | Cooldown: 30m")
    log.info("Scanner started...")

    await asyncio.sleep(5)

    scan_id = 0

    while True:
        try:
            scan_id += 1
            log.info(f"🔍 Scan #{scan_id} running...")

            all_sigs = []

            # evaluate each pair in each strategy
            for sym in PAIR_LIST:
                for strat in ["QUICK", "MID", "TREND"]:

                    # cooldown
                    if not cd_ok(sym, strat):
                        continue

                    # scorer
                    sig = await compute_score(sym, strat)
                    if not sig:
                        continue

                    # check concurrent limit
                    if open_count[strat] >= MAX_CONCURRENT:
                        continue

                    # TP/SL & leverage
                    tp1, tp2, sl, lev, liq_dist = await calc_tp_sl(
                        sig["entry"], sig["side"], sig["strategy_cfg"]
                    )

                    iceberg = iceberg_size(30000, sig["entry"], sl, lev)
                    if iceberg["total"] <= 0:
                        continue

                    all_sigs.append({
                        **sig,
                        "tp1": tp1,
                        "tp2": tp2,
                        "sl": sl,
                        "lev": lev,
                        "liq": liq_dist,
                        "ice": iceberg,
                        "strategy": strat
                    })

            # sort signals by score
            all_sigs.sort(key=lambda x: x["score"], reverse=True)

            # take top 3 signals
            top_sigs = all_sigs[:3]
            log.info(f"📊 Found {len(all_sigs)} signals, sending {len(top_sigs)}")

            # telegram send
            for s in top_sigs:

                open_count[s["strategy"]] += 1
                set_cd(s["symbol"], s["strategy"])

                msg = (
                    f"⚡ <b>{s['strategy']} | {s['symbol']} | {s['side'].upper()}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"💠 Entry: <code>{s['entry']:.4f}</code>\n"
                    f"🎯 TP1: <code>{s['tp1']:.4f}</code> (60%)\n"
                    f"🎯 TP2: <code>{s['tp2']:.4f}</code> (40%)\n"
                    f"🛑 SL: <code>{s['sl']:.4f}</code>\n"
                    f"⚡ Leverage: <b>{s['lev']}x</b> | Liq-dist: {s['liq']:.1f}%\n"
                    f"📊 Score: {s['score']:.2f}\n"
                    f"✔ Passed: {s['done']}\n"
                    f"📦 Iceberg: {s['ice']['orders']}×{s['ice']['each']:.4f}\n"
                )

                await send_telegram(msg)
                await asyncio.sleep(1)

            await asyncio.sleep(SCAN_INTERVAL)

        except Exception as e:
            log.error(f"scanner err: {e}")
            await asyncio.sleep(5)


# ----------------------------
# FASTAPI LIFESPAN
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 App starting...")
    asyncio.create_task(scan_loop())
    yield
    log.info("⏳ Shutting down...")
    await close_redis()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return {
        "status": "running",
        "pairs": len(PAIR_LIST),
        "cooldown": COOLDOWN_MIN,
        "time": datetime.utcnow().isoformat()
    }


# For Railway
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))