# main.py
# -------------------------------------------------------------
# FINAL ASCII-SAFE MAIN ENGINE WITH KEEPALIVE PING
# TELEGRAM SENDER - MODE SCAN - OVERRIDE GUARDIAN - KEEPALIVE
# -------------------------------------------------------------

import asyncio
import logging
import time
import os
from typing import List, Dict, Any
import aiohttp
from dotenv import load_dotenv

# load env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PING_URL = os.getenv("PING_URL")  # e.g. https://yourapp.up.railway.app/

# import helpers (assumes helpers.py is in same folder)
from helpers import run_all_modes, format_signal, multi_override_watch

# -------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# VALIDATE ENV
# -------------------------------------------------------------
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env. Telegram send will fail until set.")

# -------------------------------------------------------------
# HTTP SESSION (reuse)
# -------------------------------------------------------------
_http_session: aiohttp.ClientSession = None

async def get_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    return _http_session

async def close_session():
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None

# -------------------------------------------------------------
# TELEGRAM SENDER
# -------------------------------------------------------------
async def tg_send(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials missing. Skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        sess = await get_session()
        async with sess.post(url, data=payload, timeout=10) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Telegram send failed status={resp.status} text={text}")
    except Exception as e:
        logger.error(f"Telegram send exception: {e}")

# -------------------------------------------------------------
# SPAM PROTECTION (5 MIN PER PAIR)
# -------------------------------------------------------------
LAST_ALERT: Dict[str, int] = {}

def can_alert(symbol: str) -> bool:
    now = int(time.time())
    if symbol not in LAST_ALERT:
        return True
    return (now - LAST_ALERT[symbol]) > 300

def mark_alert(symbol: str):
    LAST_ALERT[symbol] = int(time.time())

# -------------------------------------------------------------
# SEND SIGNALS ONE BY ONE (SAFE)
# -------------------------------------------------------------
async def send_signals(signals: List[Dict[str, Any]]):
    for sig in signals:
        try:
            if not can_alert(sig["symbol"]):
                continue
            msg = format_signal(sig)
            await tg_send(msg)
            mark_alert(sig["symbol"])
            await asyncio.sleep(1)  # pacing to avoid rate limits
        except Exception as e:
            logger.error(f"Error sending signal for {sig.get('symbol')}: {e}")

# -------------------------------------------------------------
# KEEPALIVE PING (prevents Railway/Render sleep)
# -------------------------------------------------------------
async def keepalive_ping():
    if not PING_URL:
        logger.info("PING_URL not set. Keepalive ping disabled.")
        return
    logger.info("Keepalive ping task started.")
    while True:
        try:
            sess = await get_session()
            async with sess.get(PING_URL, timeout=10) as resp:
                logger.debug(f"Ping to {PING_URL} returned {resp.status}")
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
        await asyncio.sleep(60)  # ping every 60 seconds

# -------------------------------------------------------------
# MAIN ENGINE LOOP
# -------------------------------------------------------------
async def engine():
    logger.info("ENGINE STARTED - BALANCED MODE ACTIVE (10-15 alerts per day)")

    try:
        while True:
            try:
                # SCAN ALL MODES (quick, mid, trend)
                all_modes = await run_all_modes()

                # MERGE SIGNALS
                signals = (
                    all_modes.get("quick", []) +
                    all_modes.get("mid", []) +
                    all_modes.get("trend", [])
                )

                # SEND SIGNALS
                if len(signals) > 0:
                    await send_signals(signals)

                    # OVERRIDE WATCH
                    danger = await multi_override_watch(signals)
                    for d in danger:
                        await tg_send(d)

            except Exception as e:
                logger.error(f"ENGINE cycle error: {e}")

            # WAIT BEFORE NEXT SCAN
            await asyncio.sleep(20)

    finally:
        # cleanup session on stop
        await close_session()

# -------------------------------------------------------------
# RUN BOTH ENGINE + PING IN PARALLEL
# -------------------------------------------------------------
if __name__ == "__main__":
    async def start_all():
        tasks = []
        tasks.append(asyncio.create_task(engine()))
        # start keepalive ping if PING_URL provided
        if PING_URL:
            tasks.append(asyncio.create_task(keepalive_ping()))
        await asyncio.gather(*tasks)

    try:
        asyncio.run(start_all())
    except KeyboardInterrupt:
        logger.info("Received exit. Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")