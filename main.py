# main.py
# ENGINE + TELEGRAM SENDER (ASCII ONLY)

import asyncio
import logging
import os
import threading
import traceback

from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
import aiohttp

# helpers must provide:
# async def run_all_modes() -> dict  (quick/mid/trend -> list of signals)
# async def multi_override_watch(active_signals: list) -> list[str]
# def format_signal(sig: dict) -> str
from helpers import run_all_modes, multi_override_watch, format_signal

# -------------------------------------------------
# FASTAPI APP (HEALTH CHECK FOR RAILWAY)
# -------------------------------------------------
app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

# -------------------------------------------------
# ENV + LOGGING
# -------------------------------------------------
load_dotenv()

PING_URL = os.getenv("PING_URL")

# try both names so old .env ও কাজ করবে
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("signal-engine")

# -------------------------------------------------
# TELEGRAM SENDER
# -------------------------------------------------
async def send_telegram(text: str) -> None:
    """Send plain text message to Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram token/chat_id missing, skip send.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        # parse_mode deliberately omitted (emoji-safe, any text ok)
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload, timeout=10) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("Telegram send failed: %s %s", resp.status, body)
                else:
                    logger.debug("Telegram send ok.")
    except Exception as e:
        logger.error("Telegram send exception: %s", str(e))

# -------------------------------------------------
# KEEPALIVE PING (OPTIONAL)
# -------------------------------------------------
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

# -------------------------------------------------
# FLATTEN MODES -> ONE LIST OF SIGNALS
# -------------------------------------------------
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
            logger.warning(
                "flatten_modes: mode %s returned %s (not list)",
                key,
                type(part),
            )
    return signals

# -------------------------------------------------
# ONE ENGINE CYCLE
# -------------------------------------------------
async def engine_cycle():
    """
    1) run_all_modes -> all_modes
    2) flatten -> active_signals
    3) সব BUY signal Telegram এ পাঠাও
    4) active_signals দিয়ে multi_override_watch -> EXIT alert গুলোও Telegram এ পাঠাও
    """
    try:
        all_modes = await run_all_modes()
    except Exception as e:
        logger.error("run_all_modes failed: %s", str(e))
        logger.debug(traceback.format_exc())
        return

    # debug: shape log
    try:
        if isinstance(all_modes, dict):
            shape = {k: type(v).__name__ for k, v in all_modes.items()}
        else:
            shape = str(type(all_modes))
        logger.info("run_all_modes returned types: %s", shape)
    except Exception:
        logger.debug("unable to log run_all_modes shape")

    # 1) flatten
    active_signals = flatten_modes(all_modes)

    # debug sample
    if active_signals:
        try:
            sample = active_signals[:3]
            sample_keys = [
                list(s.keys()) if isinstance(s, dict) else str(type(s))
                for s in sample
            ]
            logger.info(
                "active_signals count=%d sample_keys=%s",
                len(active_signals),
                sample_keys,
            )
        except Exception:
            logger.debug("could not inspect active_signals sample")

    # 2) BUY SIGNALS -> TELEGRAM
    for sig in active_signals:
        if not isinstance(sig, dict):
            continue
        if not sig.get("ok"):
            continue

        try:
            text = format_signal(sig)
            await send_telegram(text)
            logger.info(
                "TELEGRAM BUY SENT: mode=%s symbol=%s entry=%s score=%s",
                sig.get("mode"),
                sig.get("symbol"),
                sig.get("entry"),
                sig.get("score"),
            )
        except Exception as e:
            logger.error("failed to send BUY signal: %s", str(e))

    # 3) OVERRIDE / EXIT ALERTS
    if active_signals:
        try:
            alerts = await multi_override_watch(active_signals)
            if alerts:
                logger.info("multi_override_watch returned %d alerts", len(alerts))
                for a in alerts:
                    # log + send
                    logger.info("OVERRIDE ALERT: %s", a)
                    await send_telegram(a)
        except Exception as e:
            logger.error("multi_override_watch failed: %s", str(e))
            logger.error(traceback.format_exc())

# -------------------------------------------------
# CONTINUOUS ENGINE LOOP
# -------------------------------------------------
SCAN_INTERVAL = 20  # seconds, Balanced mode

async def engine_loop():
    logger.info("ENGINE STARTED - BALANCED MODE (10-15 alerts/day)")
    while True:
        try:
            await engine_cycle()
        except Exception as e:
            logger.error("engine_cycle outer exception: %s", str(e))
            logger.error(traceback.format_exc())
        await asyncio.sleep(SCAN_INTERVAL)

# -------------------------------------------------
# SAFE RUNNER
# -------------------------------------------------
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

# -------------------------------------------------
# START WEB SERVER (RAILWAY)
# -------------------------------------------------
def start_web():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    # web server in background thread
    thr = threading.Thread(target=start_web, daemon=True)
    thr.start()

    try:
        asyncio.run(safe_runner())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting.")
    except Exception as e:
        logger.error("Unhandled main exception: %s", str(e))
        logger.error(traceback.format_exc())