# main.py
# ASCII-only, async signal engine runner for Railway

import asyncio
import logging
import os
import threading
import traceback
from typing import Any, Dict, List

from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
import aiohttp

from helpers import run_all_modes, multi_override_watch, format_signal

# -------------------------------------------------------------
# FASTAPI APP (for Railway health check)
# -------------------------------------------------------------
app = FastAPI()

@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "running"}

# -------------------------------------------------------------
# ENV + LOGGING
# -------------------------------------------------------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
PING_URL = os.getenv("PING_URL", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("signal-engine")

# -------------------------------------------------------------
# TELEGRAM SENDER
# -------------------------------------------------------------
async def send_telegram(text: str) -> None:
    """
    Send a message to Telegram chat using bot token + chat id from env.
    If not configured, just log the message.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured, message skipped:\n%s", text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",  # safe for plain text too
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as resp:
                _ = await resp.text()
                if resp.status != 200:
                    logger.warning("Telegram send failed: status=%s body=%s", resp.status, _)
    except Exception as e:
        logger.error("send_telegram error: %s", e)

# -------------------------------------------------------------
# KEEPALIVE PING
# -------------------------------------------------------------
async def ping_forever() -> None:
    """
    Periodically hit PING_URL to keep Railway dyno alive.
    """
    if not PING_URL:
        logger.info("PING_URL not set, ping_forever disabled.")
        return

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(PING_URL, timeout=10) as resp:
                    _ = await resp.text()
                    logger.debug("PING: status=%s", resp.status)
        except Exception as e:
            logger.warning("PING error: %s", e)
        await asyncio.sleep(60)

# -------------------------------------------------------------
# SIGNAL TRACKING (TO AVOID RESENDING SAME SIGNAL)
# -------------------------------------------------------------
SENT_KEYS: set = set()
SIGNAL_COUNT_TODAY: int = 0

def make_signal_key(sig: Dict[str, Any]) -> str:
    """
    Create a unique key for a signal to avoid duplicates.
    """
    symbol = sig.get("symbol", "")
    mode = sig.get("mode", "")
    side = sig.get("side", "BUY")
    time_utc = sig.get("time_utc", "")
    return f"{symbol}|{mode}|{side}|{time_utc}"

def flatten_modes(all_modes: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    all_modes expected like {'quick': [...], 'mid': [...], 'trend': [...]}
    returns single list of signals (each should be a dict)
    """
    if not isinstance(all_modes, dict):
        logger.error("flatten_modes: expected dict, got %s", type(all_modes))
        return []

    signals: List[Dict[str, Any]] = []
    for key in ("quick", "mid", "trend"):
        part = all_modes.get(key, [])
        if isinstance(part, list):
            signals.extend(part)
        else:
            logger.warning("flatten_modes: mode %s returned %s (not list)", key, type(part))
    return signals

# -------------------------------------------------------------
# SINGLE ENGINE CYCLE
# -------------------------------------------------------------
async def engine_cycle() -> None:
    global SIGNAL_COUNT_TODAY

    try:
        all_modes = await run_all_modes()
    except Exception as e:
        logger.error("run_all_modes failed: %s", e)
        logger.debug(traceback.format_exc())
        return

    # Debug summary
    try:
        if isinstance(all_modes, dict):
            summary = {k: len(v) if isinstance(v, list) else 0 for k, v in all_modes.items()}
            logger.info("run_all_modes summary: %s", summary)
        else:
            logger.info("run_all_modes returned type: %s", type(all_modes))
    except Exception:
        logger.debug("Could not log run_all_modes summary")

    # Flatten signals
    active_signals = flatten_modes(all_modes)

    # 1) NEW SIGNALS -> TELEGRAM
    if active_signals:
        for sig in active_signals:
            if not sig.get("ok"):
                continue
            key = make_signal_key(sig)
            if key in SENT_KEYS:
                # already sent this exact signal
                continue
            # mark as sent
            SENT_KEYS.add(key)
            SIGNAL_COUNT_TODAY += 1
            # format message
            msg = format_signal(sig, SIGNAL_COUNT_TODAY)
            logger.info("New signal: %s mode=%s score=%s", sig.get("symbol"), sig.get("mode"), sig.get("score"))
            await send_telegram(msg)

    # 2) OVERRIDE / DANGER ALERTS
    try:
        if active_signals:
            alerts = await multi_override_watch(active_signals)
            if alerts:
                for a in alerts:
                    logger.info("OVERRIDE ALERT: %s", a.replace("\n", " | "))
                    await send_telegram(a)
    except Exception as e:
        logger.error("multi_override_watch failed: %s", e)
        logger.error(traceback.format_exc())

# -------------------------------------------------------------
# CONTINUOUS ENGINE LOOP
# -------------------------------------------------------------
async def engine_loop() -> None:
    logger.info("ENGINE STARTED - BALANCED MODE (10-15 alerts/day target)")
    while True:
        try:
            await engine_cycle()
        except Exception as e:
            logger.error("engine_cycle outer exception: %s", e)
            logger.error(traceback.format_exc())
        # wait between scans (20s is a good starting point)
        await asyncio.sleep(20)

# -------------------------------------------------------------
# SAFE RUNNER (RESTART ON CRASH)
# -------------------------------------------------------------
async def safe_runner() -> None:
    """
    Keep engine_loop (and ping_forever) running.
    If anything crashes, wait a bit and restart.
    """
    while True:
        try:
            tasks: List[asyncio.Task] = [asyncio.create_task(engine_loop())]
            if PING_URL:
                tasks.append(asyncio.create_task(ping_forever()))
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except Exception as e:
            logger.error("safe_runner caught: %s", e)
            logger.error(traceback.format_exc())
        logger.info("safe_runner: restarting in 5 seconds...")
        await asyncio.sleep(5)

# -------------------------------------------------------------
# START WEB SERVER (Railway)
# -------------------------------------------------------------
def start_web() -> None:
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

# -------------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------------
if __name__ == "__main__":
    # Start FastAPI / Uvicorn in background thread
    server_thread = threading.Thread(target=start_web, daemon=True)
    server_thread.start()

    try:
        asyncio.run(safe_runner())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting.")
    except Exception as e:
        logger.error("Unhandled main exception: %s", e)
        logger.error(traceback.format_exc())