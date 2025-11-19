# main.py
# ASCII SAFE VERSION ONLY - FINAL 24/7 RAILWAY VERSION WITH SAFE RUNNER + PORT FIX

import asyncio
import logging
import time
import os
from typing import List, Dict, Any
import aiohttp
from dotenv import load_dotenv

from fastapi import FastAPI
import uvicorn
import threading

# -----------------------------------------------------------
# FASTAPI SERVER TO KEEP RAILWAY CONTAINER RUNNING
# -----------------------------------------------------------

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

# -----------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PING_URL = os.getenv("PING_URL")

# -----------------------------------------------------------
# IMPORT HELPERS
# -----------------------------------------------------------

from helpers import run_all_modes, format_signal, multi_override_watch

# -----------------------------------------------------------
# LOGGING
# -----------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# GLOBAL HTTP SESSION
# -----------------------------------------------------------

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

# -----------------------------------------------------------
# TELEGRAM MESSAGE SENDER
# -----------------------------------------------------------

async def tg_send(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram token or chat ID missing.")
        return

    url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}

    try:
        sess = await get_session()
        await sess.post(url, data=payload, timeout=10)
    except Exception as e:
        logger.error("Telegram send error: " + str(e))

# -----------------------------------------------------------
# SPAM PROTECTION (5 MIN)
# -----------------------------------------------------------

LAST_ALERT = {}

def can_alert(symbol: str) -> bool:
    now = int(time.time())
    if symbol not in LAST_ALERT:
        return True
    return (now - LAST_ALERT[symbol]) > 300

def mark_alert(symbol: str):
    LAST_ALERT[symbol] = int(time.time())

# -----------------------------------------------------------
# SEND SIGNALS
# -----------------------------------------------------------

async def send_signals(signals: List[Dict[str, Any]]):
    for sig in signals:
        try:
            if can_alert(sig["symbol"]):
                await tg_send(format_signal(sig))
                mark_alert(sig["symbol"])
                await asyncio.sleep(1)
        except Exception as e:
            logger.error("Error sending signal: " + str(e))

# -----------------------------------------------------------
# KEEPALIVE PING
# -----------------------------------------------------------

async def keepalive_ping():
    if not PING_URL:
        logger.info("PING_URL not set. Disable keepalive.")
        return

    logger.info("Keepalive ping active.")
    while True:
        try:
            sess = await get_session()
            await sess.get(PING_URL, timeout=10)
        except:
            pass
        await asyncio.sleep(60)

# -----------------------------------------------------------
# TRADING ENGINE LOOP
# -----------------------------------------------------------

async def engine():
    logger.info("ENGINE STARTED - BALANCED MODE (10-15 alerts/day)")
    while True:
        try:
            all_modes = await run_all_modes()

            signals = (
                all_modes.get("quick", []) +
                all_modes.get("mid", []) +
                all_modes.get("trend", [])
            )

            if len(signals) > 0:
                await send_signals(signals)

                alerts = await multi_override_watch(signals)
                for a in alerts:
                    await tg_send(a)

        except Exception as e:
            logger.error("Engine error: " + str(e))

        await asyncio.sleep(20)

# -----------------------------------------------------------
# START WEB SERVER (PORT FIX FOR RAILWAY)
# -----------------------------------------------------------

def start_web():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# -----------------------------------------------------------
# START EVERYTHING
# -----------------------------------------------------------

async def start_all():
    t_engine = asyncio.create_task(engine())
    t_ping = None

    if PING_URL:
        t_ping = asyncio.create_task(keepalive_ping())

    if t_ping:
        await asyncio.gather(t_engine, t_ping)
    else:
        await asyncio.gather(t_engine)

# -----------------------------------------------------------
# SAFE RUNNER (AUTO RESTART IF ENGINE RETURNS)
# -----------------------------------------------------------

async def safe_runner():
    while True:
        try:
            await start_all()
            logger.warning("start_all returned unexpectedly. Restarting.")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error("safe_runner exception: " + str(e))
            await asyncio.sleep(5)

# -----------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------

if __name__ == "__main__":

    server_thread = threading.Thread(target=start_web)
    server_thread.daemon = True
    server_thread.start()

    try:
        asyncio.run(safe_runner())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Fatal error: " + str(e))