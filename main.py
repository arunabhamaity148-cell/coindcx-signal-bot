# main.py
# ASCII SAFE VERSION — RAILWAY SAFE

import asyncio
import logging
import os
import threading
import time
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn

# Import helpers
from helpers import (
    run_all_modes,
    format_signal,
    multi_override_watch
)

# -------------------------
# FastAPI App (Ping Protect)
# -------------------------
app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

# -------------------------
# ENV
# -------------------------
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PING_URL = os.getenv("PING_URL")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("coindcx-signal-engine")

# -------------------------
# PINGER — Keep bot alive
# -------------------------
async def ping_forever():
    if not PING_URL:
        return
    import aiohttp
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                await session.get(PING_URL)
            logger.info("PING → OK")
        except Exception as e:
            logger.error(f"PING ERROR → {e}")
        await asyncio.sleep(60)

# -------------------------
# MAIN ENGINE
# -------------------------
async def engine():
    logger.info("ENGINE STARTED — BALANCED MODE (10–15 alerts/day)")
    await asyncio.gather(
        run_all_modes(),          # quick + mid + trend
        multi_override_watch(),   # override safeguard
        ping_forever()            # keep-alive
    )

# -------------------------
# SAFE RUNNER (NO CRASH)
# -------------------------
async def safe_runner():
    while True:
        try:
            await engine()
            logger.warning("Engine returned unexpectedly. Restarting in 5 sec.")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"safe_runner caught → {e}")
            await asyncio.sleep(5)

# -------------------------
# Start FastAPI server (thread)
# -------------------------
def start_web():
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    # Start FastAPI in a thread
    server_thread = threading.Thread(target=start_web)
    server_thread.daemon = True
    server_thread.start()

    # Start engine
    try:
        asyncio.run(safe_runner())
    except KeyboardInterrupt:
        pass