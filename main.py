# main.py
# ASCII ONLY VERSION
# ENGINE LOOP + TELEGRAM + FASTAPI HEALTHCHECK

import asyncio
import logging
import os
import threading
import traceback

from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
import aiohttp

# helpers.py theke ei 3ta import অবশ্যই থাকা লাগবে
from helpers import run_all_modes, multi_override_watch, format_signal

# -------------------------------------------------------------
# FASTAPI APP (Railway health check)
# -------------------------------------------------------------
app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

# -------------------------------------------------------------
# ENV + LOGGING
# -------------------------------------------------------------
load_dotenv()

PING_URL = os.getenv("PING_URL")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("signal-engine")

# -------------------------------------------------------------
# TELEGRAM SENDER (ASYNC)
# -------------------------------------------------------------
async def send_telegram(text: str):
    """
    Simple async Telegram sender.
    Plain-text, no Markdown/HTML to avoid format problem.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram env missing, cannot send message.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload, timeout=10) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Telegram send failed: %s - %s", resp.status, body)
    except Exception as e:
        logger.error("Telegram send exception: %s", str(e))

# -------------------------------------------------------------
# KEEPALIVE PING (OPTIONAL)
# -------------------------------------------------------------
async def ping_forever():
    if not PING_URL:
        return
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                await session.get(PING_URL, timeout=10)
            logger.debug("PING OK")
        except Exception as e:
            logger.warning("PING error: %s", str(e))
        await asyncio.sleep(60)

# -------------------------------------------------------------
# HELPER: FLATTEN MODES
# -------------------------------------------------------------
def flatten_modes(all_modes: dict) -> list:
    """
    all_modes expected like {'quick': [...], 'mid': [...], 'trend': [...]}
    returns single list of signals (each should be a dict)
    """
    if not isinstance(all_modes, dict):
        logger.error("flatten_modes: expected dict, got %s", type(all_modes))
        return []

    signals = []
    for key in ("quick", "mid", "trend"):
        part = all_modes.get(key, [])
        if isinstance(part, list):
            signals.extend(part)
        else:
            logger.warning("flatten_modes: mode %s returned %s (not list)", key, type(part))
    return signals

# -------------------------------------------------------------
# MAIN ENGINE CYCLE
# -------------------------------------------------------------
async def engine_cycle():
    """
    1) run_all_modes -> get signals
    2) flatten -> active_signals list
    3) send BUY signals to Telegram
    4) run multi_override_watch -> EXIT / WARNING alerts to Telegram
    """
    try:
        all_modes = await run_all_modes()
    except Exception as e:
        logger.error("run_all_modes failed: %s", str(e))
        logger.debug(traceback.format_exc())
        return

    # debug: log type info
    try:
        if isinstance(all_modes, dict):
            logger.info(
                "run_all_modes returned types: %s",
                {k: type(v).__name__ for k, v in all_modes.items()}
            )
        else:
            logger.info("run_all_modes returned: %s", type(all_modes).__name__)
    except Exception:
        logger.debug("unable to log run_all_modes shape")

    # flatten signals
    active_signals = flatten_modes(all_modes)

    # BUY SIGNALS -> TELEGRAM
    if active_signals:
        for sig in active_signals:
            try:
                text = format_signal(sig)
                await send_telegram(text)
            except Exception as e:
                logger.error("send BUY signal failed: %s", str(e))

    # OVERRIDE WATCH -> EXIT/WARNING ALERTS
    try:
        alerts = await multi_override_watch(active_signals)
        if alerts:
            for a in alerts:
                await send_telegram(a)
                logger.info("OVERRIDE ALERT SENT: %s", a)
    except Exception as e:
        logger.error("multi_override_watch failed: %s", str(e))
        logger.error(traceback.format_exc())

# -------------------------------------------------------------
# CONTINUOUS ENGINE LOOP
# -------------------------------------------------------------
async def engine_loop():
    logger.info("ENGINE STARTED - BALANCED MODE (10-15 alerts/day)")
    while True:
        try:
            await engine_cycle()
        except Exception as e:
            logger.error("engine_cycle outer exception: %s", str(e))
            logger.error(traceback.format_exc())
        # প্রতি ২০ সেকেন্ডে নতুন scan
        await asyncio.sleep(20)

# -------------------------------------------------------------
# SAFE RUNNER
# -------------------------------------------------------------
async def safe_runner():
    while True:
        try:
            tasks = [asyncio.create_task(engine_loop())]
            if PING_URL:
                tasks.append(asyncio.create_task(ping_forever()))
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except Exception as e:
            logger.error("safe_runner caught: %s", str(e))
            logger.error(traceback.format_exc())
        await asyncio.sleep(5)

# -------------------------------------------------------------
# START WEB SERVER (RAILWAY)
# -------------------------------------------------------------
def start_web():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# -------------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------------
if __name__ == "__main__":
    # web server background thread
    thr = threading.Thread(target=start_web, daemon=True)
    thr.start()

    try:
        asyncio.run(safe_runner())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting.")
    except Exception as e:
        logger.error("Unhandled main exception: %s", str(e))
        logger.error(traceback.format_exc())