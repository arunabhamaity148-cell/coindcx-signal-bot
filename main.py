# main.py
# ASCII SAFE VERSION ONLY - FINAL 24/7 RAILWAY VERSION WITH SAFE RUNNER

import asyncio
import logging
import time
import os
from typing import List, Dict, Any
import aiohttp
from dotenv import load_dotenv

# FastAPI server to keep Railway container alive
from fastapi import FastAPI
import uvicorn

import threading

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PING_URL = os.getenv("PING_URL")

# Helpers import
from helpers import run_all_modes, format_signal, multi_override_watch

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# HTTP session cache
_http_session = None

async def get_session():
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    return _http_session

async def close_session():
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None

# Telegram sender
async def tg_send(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram token or chat id not set.")
        return
    url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        sess = await get_session()
        await sess.post(url, data=payload, timeout=10)
    except Exception as e:
        logger.error("Telegram send error: " + str(e))

# Spam protection (5 min per pair)
LAST_ALERT = {}

def can_alert(symbol: str) -> bool:
    now = int(time.time())
    if symbol not in LAST_ALERT:
        return True
    return (now - LAST_ALERT[symbol]) > 300

def mark_alert(symbol: str):
    LAST_ALERT[symbol] = int(time.time())

# Send formatted signals
async def send_signals(signals: List[Dict[str, Any]]):
    for sig in signals:
        try:
            if can_alert(sig["symbol"]):
                await tg_send(format_signal(sig))
                mark_alert(sig["symbol"])
                await asyncio.sleep(1)
        except Exception as e:
            logger.error("Error sending signal for " + sig.get("symbol", "unknown") + ": " + str(e))

# Keepalive ping task to prevent Railway sleep
async def keepalive_ping():
    if not PING_URL:
        logger.info("PING_URL not set. Keepalive disabled.")
        return
    logger.info("Keepalive ping active.")
    while True:
        try:
            sess = await get_session()
            await sess.get(PING_URL, timeout=10)
        except:
            pass
        await asyncio.sleep(60)

# Main trading engine
async def engine():
    logger.info("ENGINE STARTED - BALANCED MODE (10-15 alerts/day)")
    while True:
        try:
            all_modes = await run_all_modes()
            signals = all_modes.get("quick", []) + all_modes.get("mid", []) + all_modes.get("trend", [])

            if len(signals) > 0:
                await send_signals(signals)

                # Override danger watcher (uses helpers.multi_override_watch)
                alerts = await multi_override_watch(signals)
                for a in alerts:
                    await tg_send(a)

        except Exception as e:
            logger.error("Engine cycle error: " + str(e))

        await asyncio.sleep(20)

# Start web server function (uvicorn)
def start_web():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Async starter that runs engine and optional ping
async def start_all():
    task_engine = asyncio.create_task(engine())
    task_ping = None

    if PING_URL:
        task_ping = asyncio.create_task(keepalive_ping())

    if task_ping:
        await asyncio.gather(task_engine, task_ping)
    else:
        await asyncio.gather(task_engine)

# safe_runner wrapper - keep process alive if engine crashes
async def safe_runner():
    while True:
        try:
            await start_all()
            # if start_all returns (unexpected), sleep and restart
            logger.warning("start_all() returned unexpectedly. Restarting in 5s.")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error("safe_runner caught exception: " + str(e))
            # wait a bit before restart to avoid crash-loop
            await asyncio.sleep(5)

if __name__ == "__main__":
    # start UVicorn server thread
    server_thread = threading.Thread(target=start_web)
    server_thread.daemon = True
    server_thread.start()

    try:
        asyncio.run(safe_runner())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Fatal error: " + str(e))