# main.py – Sniper-grade signal engine runner for Railway
# Boot ping, console log, emoji TG, health-check

import asyncio
import logging
import os
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
import aiohttp

from helpers import (
    run_all_modes,
    multi_override_watch,
    format_signal,
    send_telegram,
    log_signal,
)

# -------------------- FASTAPI (health-check) --------------------
app = FastAPI()

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "status": "running",
        "engine": "sniper",
        "ts": datetime.utcnow().isoformat(),
    }

# -------------------- ENV + LOGGING (console + file) --------------------
load_dotenv()

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/signals.log"),
        logging.StreamHandler(),  # Railway console-e show korbe
    ],
)
logger = logging.getLogger("sniper-engine")

PING_URL = os.getenv("PING_URL", "").strip()

# -------------------- TELEGRAM TEST PING --------------------
async def test_telegram():
    txt = "🟢 <b>Sniper bot online</b>\n<code>Score ≥ {}</code>".format(
        os.getenv("SCORE_MIN", 90)
    )
    await send_telegram(txt)
    logger.info(
        "Telegram boot ping sent (score_min=%s)",
        os.getenv("SCORE_MIN", "90"),
    )

# -------------------- KEEPALIVE PING (Railway) --------------------
async def ping_forever():
    if not PING_URL:
        return
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(PING_URL, timeout=10) as r:
                    logger.debug("Ping: %s", r.status)
        except Exception as e:
            logger.warning("Ping error: %s", e)
        await asyncio.sleep(60)

# -------------------- SIGNAL DEDUP --------------------
SENT_KEYS: set = set()
SIGNAL_COUNT: int = 0

def make_key(sig: Dict[str, Any]) -> str:
    return f"{sig['symbol']}|{sig['mode']}|{sig['side']}|{sig['time_utc']}"

def flatten_modes(all_modes: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    for mode in ("quick", "mid", "trend"):
        part = all_modes.get(mode, [])
        if isinstance(part, list):
            signals.extend(part)
        else:
            logger.warning("Mode %s not list: %s", mode, type(part))
    return signals

# -------------------- ENGINE CYCLE --------------------
async def engine_cycle() -> None:
    global SIGNAL_COUNT
    try:
        all_modes = await run_all_modes()
    except Exception as e:
        logger.error("run_all_modes error: %s", e)
        traceback.print_exc()
        return

    summary = {k: len(v) if isinstance(v, list) else 0 for k, v in all_modes.items()}
    logger.info("Cycle summary: %s", summary)

    signals = flatten_modes(all_modes)

    # New signals
    for sig in signals:
        if not sig.get("ok"):
            continue
        key = make_key(sig)
        if key in SENT_KEYS:
            continue
        SENT_KEYS.add(key)
        SIGNAL_COUNT += 1
        msg = format_signal(sig, SIGNAL_COUNT)
        logger.info("New signal %s #%d", sig["symbol"], SIGNAL_COUNT)
        await send_telegram(msg)
        log_signal(sig)

    # Override alerts
    if signals:
        try:
            alerts = await multi_override_watch(signals)
            for a in alerts:
                logger.info("Override: %s", a.replace("\n", " | "))
                await send_telegram(a)
        except Exception as e:
            logger.error("override error: %s", e)
            traceback.print_exc()

# -------------------- ENGINE LOOP --------------------
async def engine_loop() -> None:
    logger.info("===== BOT BOOTING =====")
    await test_telegram()
    while True:
        try:
            await engine_cycle()
        except Exception as e:
            logger.error("Cycle crash: %s", e)
            traceback.print_exc()
        await asyncio.sleep(20)   # প্রতি ২০ সেকে একবার full scan

# -------------------- SAFE RUNNER --------------------
async def safe_runner() -> None:
    while True:
        try:
            tasks = [asyncio.create_task(engine_loop())]
            if PING_URL:
                tasks.append(asyncio.create_task(ping_forever()))
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except Exception as e:
            logger.error("safe_runner crash: %s", e)
            traceback.print_exc()
        logger.info("Restarting in 5s...")
        await asyncio.sleep(5)

# -------------------- WEB SERVER (Railway) --------------------
def start_web():
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    threading.Thread(target=start_web, daemon=True).start()
    try:
        asyncio.run(safe_runner())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
