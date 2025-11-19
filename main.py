# main.py
# ASCII SAFE VERSION ONLY

import asyncio
import logging
import time
import os
from typing import List, Dict, Any
import aiohttp
from dotenv import load_dotenv

# Load env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PING_URL = os.getenv("PING_URL")

from helpers import run_all_modes, format_signal, multi_override_watch

# Logging
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
        logger.error("Telegram environment variables not set.")
        return
    url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        sess = await get_session()
        async with sess.post(url, data=payload, timeout=10):
            pass
    except Exception as e:
        logger.error("Telegram send error: " + str(e))

# Spam protection
LAST_ALERT = {}

def can_alert(symbol: str) -> bool:
    now = int(time.time())
    if symbol not in LAST_ALERT:
        return True
    return (now - LAST_ALERT[symbol]) > 300

def mark_alert(symbol: str):
    LAST_ALERT[symbol] = int(time.time())

# Send multiple signals
async def send_signals(signals: List[Dict[str, Any]]):
    for sig in signals:
        if can_alert(sig["symbol"]):
            await tg_send(format_signal(sig))
            mark_alert(sig["symbol"])
            await asyncio.sleep(1)

# Keepalive ping
async def keepalive_ping():
    if not PING_URL:
        logger.info("PING_URL not set. Keepalive disabled.")
        return
    while True:
        try:
            sess = await get_session()
            await sess.get(PING_URL, timeout=10)
        except:
            pass
        await asyncio.sleep(60)

# Engine loop
async def engine():
    logger.info("Engine started")
    while True:
        try:
            all_modes = await run_all_modes()
            signals = all_modes.get("quick", []) + all_modes.get("mid", []) + all_modes.get("trend", [])
            if len(signals) > 0:
                await send_signals(signals)
                alerts = await multi_override_watch(signals)
                for al in alerts:
                    await tg_send(al)
        except Exception as e:
            logger.error("Engine error: " + str(e))
        await asyncio.sleep(20)

# Start both engine + ping
if __name__ == "__main__":
    async def start_all():
        t1 = asyncio.create_task(engine())
        if PING_URL:
            t2 = asyncio.create_task(keepalive_ping())
            await asyncio.gather(t1, t2)
        else:
            await asyncio.gather(t1)

    try:
        asyncio.run(start_all())
    except KeyboardInterrupt:
        pass