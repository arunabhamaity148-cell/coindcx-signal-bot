# main.py
# FIXED: ensure run_all_modes result flattened into active_signals list
# ASCII ONLY

import asyncio
import logging
import os
import threading
import traceback
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn

# helpers must provide:
# async def run_all_modes() -> dict  (with keys quick/mid/trend returning lists)
# async def multi_override_watch(active_signals: list) -> list
from helpers import run_all_modes, multi_override_watch, format_signal

# fastapi app for Railway health
app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

# env
load_dotenv()
PING_URL = os.getenv("PING_URL")

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("signal-engine")

# keepalive ping (optional)
async def ping_forever():
    if not PING_URL:
        return
    import aiohttp
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                await session.get(PING_URL, timeout=10)
            logger.debug("PING OK")
        except Exception as e:
            logger.warning("PING error: %s", str(e))
        await asyncio.sleep(60)

# helper to flatten signals safely
def flatten_modes(all_modes: dict) -> list:
    """
    all_modes expected like {'quick': [...], 'mid': [...], 'trend': [...]}
    returns single list of signals (each should be a dict)
    """
    if not isinstance(all_modes, dict):
        # unexpected type — return empty list but log
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

# main engine cycle
async def engine_cycle():
    """
    Single cycle: fetch modes -> flatten -> send to override watcher
    """
    try:
        all_modes = await run_all_modes()
    except Exception as e:
        logger.error("run_all_modes failed: %s", str(e))
        logger.debug(traceback.format_exc())
        return

    # debug: log returned type/summary
    try:
        logger.info("run_all_modes returned types: %s", {k: type(v).__name__ for k, v in all_modes.items()} if isinstance(all_modes, dict) else str(type(all_modes)))
    except Exception:
        logger.debug("unable to log run_all_modes shape")

    # flatten into list of signals
    active_signals = flatten_modes(all_modes)

    # debug inspect first few
    if active_signals:
        try:
            sample = active_signals[:3]
            logger.info("active_signals count=%d sample_keys=%s", len(active_signals), [list(s.keys()) if isinstance(s, dict) else str(type(s)) for s in sample])
        except Exception:
            logger.debug("could not inspect active_signals sample")

    # call override watcher with correct type
    try:
        alerts = await multi_override_watch(active_signals)
        # alerts expected list of formatted messages (strings)
        if alerts:
            for a in alerts:
                # we don't send telegram here to avoid duplicate responsibilities;
                # but if you want, call your tg sender or format_signal
                logger.info("OVERRIDE ALERT: %s", a)
    except Exception as e:
        logger.error("multi_override_watch failed: %s", str(e))
        logger.error(traceback.format_exc())

# continuous engine loop
async def engine_loop():
    logger.info("ENGINE STARTED - BALANCED MODE (10-15 alerts/day)")
    while True:
        try:
            await engine_cycle()
        except Exception as e:
            logger.error("engine_cycle outer exception: %s", str(e))
            logger.error(traceback.format_exc())
        await asyncio.sleep(20)

# safe runner to keep process alive
async def safe_runner():
    while True:
        try:
            # start uvicorn done separately in thread; run engine + ping here
            tasks = [asyncio.create_task(engine_loop())]
            if PING_URL:
                tasks.append(asyncio.create_task(ping_forever()))
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except Exception as e:
            logger.error("safe_runner caught: %s", str(e))
            logger.error(traceback.format_exc())
        await asyncio.sleep(5)

# start web server (Railway PORT)
def start_web():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    # run webserver in background thread
    thr = threading.Thread(target=start_web, daemon=True)
    thr.start()

    try:
        asyncio.run(safe_runner())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting.")
    except Exception as e:
        logger.error("Unhandled main exception: %s", str(e))
        logger.error(traceback.format_exc())