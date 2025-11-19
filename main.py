# main.py
# ASCII SAFE - SINGLE EVENT LOOP - UVICORN ASYNC SERVER
# FINAL STABLE 24/7 RAILWAY VERSION

import asyncio
import logging
import time
import os
from typing import List, Dict, Any
import aiohttp
from dotenv import load_dotenv

# FastAPI + Uvicorn async server
from fastapi import FastAPI
from uvicorn import Config as UvicornConfig, Server as UvicornServer

# Load environment
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PING_URL = os.getenv("PING_URL")

# FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

# Import helpers (ensure helpers.py present)
from helpers import run_all_modes, format_signal, multi_override_watch

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global aiohttp session
_http_session: aiohttp.ClientSession | None = None

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

# Telegram sender
async def tg_send(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials missing.")
        return
    url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        sess = await get_session()
        async with sess.post(url, data=payload, timeout=10) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Telegram send failed status={resp.status} text={text}")
    except Exception as e:
        logger.error("Telegram send exception: " + str(e))

# Spam protection (5 minutes per symbol)
LAST_ALERT: Dict[str, int] = {}

def can_alert(symbol: str) -> bool:
    now = int(time.time())
    if symbol not in LAST_ALERT:
        return True
    return (now - LAST_ALERT[symbol]) > 300

def mark_alert(symbol: str):
    LAST_ALERT[symbol] = int(time.time())

# Send signals sequentially with pacing
async def send_signals(signals: List[Dict[str, Any]]):
    for sig in signals:
        try:
            if can_alert(sig["symbol"]):
                await tg_send(format_signal(sig))
                mark_alert(sig["symbol"])
                await asyncio.sleep(1)
        except Exception as e:
            logger.error("Error sending signal for %s: %s", sig.get("symbol", "unknown"), str(e))

# Keepalive ping (optional)
async def keepalive_ping():
    if not PING_URL:
        logger.info("PING_URL not set. Keepalive disabled.")
        return
    logger.info("Keepalive ping active.")
    while True:
        try:
            sess = await get_session()
            async with sess.get(PING_URL, timeout=10):
                pass
        except Exception as e:
            logger.debug("Keepalive ping failed: %s", str(e))
        await asyncio.sleep(60)

# Main trading engine loop
async def engine_loop():
    logger.info("ENGINE STARTED - BALANCED MODE (10-15 alerts/day)")
    while True:
        try:
            all_modes = await run_all_modes()
            signals = (all_modes.get("quick", []) + all_modes.get("mid", []) + all_modes.get("trend", []))
            if signals:
                await send_signals(signals)
                alerts = await multi_override_watch(signals)
                for a in alerts:
                    await tg_send(a)
        except Exception as e:
            logger.error("Engine cycle exception: %s", str(e))
        await asyncio.sleep(20)

# Uvicorn server run as an async coroutine
async def run_uvicorn_server():
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    config = UvicornConfig(app=app, host="0.0.0.0", port=port, log_level="info", loop="asyncio")
    server = UvicornServer(config)
    logger.info("Starting uvicorn server on port %d", port)
    # serve() is a coroutine that runs until shutdown
    await server.serve()

# Coordinator that starts uvicorn + engine + ping in same loop
async def start_all_tasks():
    uvicorn_task = asyncio.create_task(run_uvicorn_server())
    engine_task = asyncio.create_task(engine_loop())
    ping_task = None
    if PING_URL:
        ping_task = asyncio.create_task(keepalive_ping())

    tasks = [uvicorn_task, engine_task]
    if ping_task:
        tasks.append(ping_task)

    # Wait until one of the tasks finishes or raises
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # If any task raised an exception, log it and cancel others
    for d in done:
        if d.exception():
            logger.error("Task exception: %s", str(d.exception()))
    for p in pending:
        p.cancel()
    # allow short delay before returning to caller
    await asyncio.sleep(1)

# Safe runner to restart start_all_tasks if it returns unexpectedly
async def safe_runner():
    while True:
        try:
            await start_all_tasks()
            logger.warning("start_all_tasks returned unexpectedly. Restarting in 5s.")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error("safe_runner caught exception: %s", str(e))
            await asyncio.sleep(5)

# Graceful shutdown handler
def _handle_exit():
    logger.info("Shutdown requested, closing sessions.")
    # we do not call loop.stop() here; safe_runner will exit naturally on cancellation

# Main entry
if __name__ == "__main__":
    # attach simple signal handlers for graceful shutdown
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.add_signal_handler  # type: ignore
    except Exception:
        pass

    try:
        loop.run_until_complete(safe_runner())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting.")
    except Exception as e:
        logger.error("Unhandled fatal exception in main: %s", str(e))
    finally:
        # cleanup aiohttp session
        try:
            loop.run_until_complete(close_session())
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass