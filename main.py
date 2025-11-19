# main.py
# -------------------------------------------------------------
# ASCII SAFE VERSION - FULL ENGINE
# TELEGRAM SENDER - MODE SCAN - OVERRIDE GUARDIAN
# -------------------------------------------------------------

import asyncio
import logging
import time
from typing import List, Dict, Any
import aiohttp

# import helpers
from helpers import run_all_modes, format_signal, multi_override_watch

# -------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# TELEGRAM CONFIG
# -------------------------------------------------------------
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

async def tg_send(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, data=payload)
    except Exception as e:
        logging.error(f"Telegram send failed: {e}")

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
        if not can_alert(sig["symbol"]):
            continue
        msg = format_signal(sig)
        await tg_send(msg)
        mark_alert(sig["symbol"])
        await asyncio.sleep(1)

# -------------------------------------------------------------
# MAIN ENGINE LOOP
# -------------------------------------------------------------
async def engine():
    logging.info("ENGINE STARTED - BALANCED MODE ACTIVE (10-15 alerts per day)")

    while True:
        try:
            # SCAN ALL MODES
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
            logging.error(f"ENGINE ERROR: {e}")

        # WAIT BEFORE NEXT SCAN
        await asyncio.sleep(20)

# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(engine())