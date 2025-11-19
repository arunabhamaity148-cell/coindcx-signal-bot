# main.py (FINAL FIXED VERSION)

import asyncio
import logging
import os
import threading
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn

# Import helpers
from helpers import run_all_modes, multi_override_watch

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}


# ENV
load_dotenv()
PING_URL = os.getenv("PING_URL")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("signal-engine")


# --- KEEP ALIVE PINGER ---
async def ping_forever():
    if not PING_URL:
        return
    import aiohttp
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                await session.get(PING_URL)
            logger.info("PING OK")
        except Exception as e:
            logger.error(f"PING ERROR: {e}")
        await asyncio.sleep(60)


# --- ENGINE ---
async def engine():
    logger.info("ENGINE STARTED - BALANCED MODE (10-15 alerts/day)")

    # run_all_modes returns active_signals list
    active_signals = await run_all_modes()

    # override watcher receives list
    await multi_override_watch(active_signals)


# --- SAFE RUNNER ---
async def safe_runner():
    while True:
        try:
            await asyncio.gather(
                engine(),
                ping_forever()
            )
        except Exception as e:
            logger.error(f"safe_runner caught → {e}")
        await asyncio.sleep(5)


# --- START FASTAPI IN SEPARATE THREAD ---
def start_web():
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))


if __name__ == "__main__":
    t = threading.Thread(target=start_web)
    t.daemon = True
    t.start()

    asyncio.run(safe_runner())