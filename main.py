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
        "ts": datetime.utcnow().isoformat()
    }

# -------------------- ENV + LOGGING (console + file) --------------------
load_dotenv()

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/signals.log"),
        logging.StreamHandler(),   # Railway console-এ দেখাবে
    ],
)

logger = logging.getLogger("sniper-engine")

PING_URL = os.getenv("PING_URL", "").strip()
SCORE_MIN = os.getenv("SCORE_MIN", "90")

# -------------------- TELEGRAM TEST PING --------------------
async def test_telegram() -> None:
    txt = f"🟢 <b>Sniper bot online</b>\n<code>Score ≥ {SCORE_MIN}</code>"
    try:
        await send_telegram(txt)
        logger.info("Telegram boot ping sent")
    except Exception as e:
        logger.warning("Telegram boot ping failed: %s", e)

# -------------------- KEEPALIVE PING (Railway / UptimeRobot) --------------------
async def ping_forever() -> None:
    if not PING_URL:
        logger.info("PING_URL not set, skipping keepalive pings")
        return
    logger.info("Keepalive pinger started")
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(PING_URL, timeout=10) as r:
                    logger.debug("Ping %s -> %s", PING_URL, r.status)
        except Exception as e:
            logger.warning("Ping error: %s", e)
        await asyncio.sleep(60)

# -------------------- SIGNAL DEDUP --------------------
SENT_KEYS: set = set()
SIGNAL_COUNT: int = 0

def make_key(sig: Dict[str, Any]) -> str:
    return f"{sig['symbol']}|{sig['mode']}|{sig['side']}|{sig['time_utc']}"

def flatten_modes(all_modes: Dict[str, List[Dict]]) -> List[Dict]:
    signals: List[Dict] = []
    for mode in ("quick", "mid", "trend"):
        part = all_modes.get(mode, [])
        if isinstance(part, list):
            signals.extend(part)
        else:
            logger.warning("Mode %s is not list (%s)", mode, type(part))
    return signals

# -------------------- ENGINE CYCLE --------------------
async def engine_cycle() -> None:
    global SIGNAL_COUNT
    try:
        all_modes = await run_all_modes()
    except Exception as e:
        logger.error("run_all_modes error: %s", e)
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
        logger.info("New signal %d: %s", SIGNAL_COUNT, key)

        try:
            await send_telegram(msg)
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)

        try:
            log_signal(sig)
        except Exception as e:
            logger.warning("log_signal failed: %s", e)

    # Override alerts (danger layer)
    if signals:
        try:
            alerts = await multi_override_watch(signals)
            for a in alerts:
                logger.info("Override alert: %s", a.replace("\n", " | "))
                try:
                    await send_telegram(a)
                except Exception as e:
                    logger.warning("Telegram override failed: %s", e)
        except Exception as e:
            logger.error("multi_override_watch error: %s", e)

# -------------------- ENGINE LOOP --------------------
async def engine_loop() -> None:
    logger.info("===== ENGINE STARTED (sniper, balanced mode) =====")
    await test_telegram()
    while True:
        try:
            await engine_cycle()
        except Exception as e:
            logger.error("Engine cycle crash: %s", e)
            traceback.print_exc()
        await asyncio.sleep(20)   # প্রতি ~২০ সেকেন্ডে নতুন scan

# -------------------- SAFE RUNNER --------------------
async def safe_runner() -> None:
    """Engine + ping দুটো একসাথে চালায়; crash হলে আবার শুরু করে."""
    while True:
        try:
            tasks = [asyncio.create_task(engine_loop())]

            if PING_URL:
                tasks.append(asyncio.create_task(ping_forever()))

            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_EXCEPTION
            )

            # কোনো task crash করলে অন্যগুলো cancel
            for t in pending:
                t.cancel()

            for t in done:
                if exc := t.exception():
                    logger.error("Task crashed: %s", exc)
        except Exception as e:
            logger.error("safe_runner crash: %s", e)
            traceback.print_exc()

        logger.info("safe_runner restarting in 5s...")
        await asyncio.sleep(5)

# -------------------- WEB SERVER (Railway) --------------------
def start_web() -> None:
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting FastAPI on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    # Web server আলাদা daemon thread-এ
    threading.Thread(target=start_web, daemon=True).start()

    # Main thread-এ async engine
    try:
        asyncio.run(safe_runner())
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down.")
    except Exception as e:
        logger.error("Fatal crash in main: %s", e)
        traceback.print_exc()