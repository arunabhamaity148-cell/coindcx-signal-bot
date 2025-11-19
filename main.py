main.py (Part 1 of 4)

-------------------------------------------------------------

MAIN LOOP • TELEGRAM SEND • MODE ROTATION • ASYNC ENGINE

-------------------------------------------------------------

import asyncio import logging import time from typing import List, Dict, Any

from helpers import run_all_modes, format_signal, multi_override_watch

-------------------------------------------------------------

LOGGING

-------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") logger = logging.getLogger(name)

-------------------------------------------------------------

TELEGRAM SENDER (HTTP POST)

-------------------------------------------------------------

import aiohttp

TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN" TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

async def tg_send(msg: str): url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg} try: async with aiohttp.ClientSession() as session: async with session.post(url, data=payload) as resp: return await resp.text() except Exception as e: logger.error(f"Telegram send failed: {e}")

-------------------------------------------------------------

MODE ROTATION (Quick → Mid → Trend)

-------------------------------------------------------------

MODES = ["quick", "mid", "trend"] MODE_INDEX = 0

def next_mode() -> str: global MODE_INDEX mode = MODES[MODE_INDEX] MODE_INDEX = (MODE_INDEX + 1) % len(MODES) return mode

-------------------------------------------------------------

SCAN + SEND SIGNALS

-------------------------------------------------------------

async def scan_and_alert(): logger.info("Running mode scan...")

all_modes = await run_all_modes()
alerts = []

for mode, results in all_modes.items():
    for sig in results:
        msg = format_signal(sig)
        await tg_send(msg)
        alerts.append(sig)

return alerts

-------------------------------------------------------------

MAIN ENGINE

-------------------------------------------------------------

async def engine(): logger.info("STARTING LIVE ENGINE...")

while True:
    try:
        active_signals = await scan_and_alert()

        # guardian override
        if len(active_signals) > 0:
            logger.info("Running danger override check...")
            danger_msgs = await multi_override_watch(active_signals)
            for d in danger_msgs:
                await tg_send(d)

    except Exception as e:
        logger.error(f"ENGINE ERROR: {e}")

    await asyncio.sleep(20)  # next cycle

-------------------------------------------------------------

RUN APP

-------------------------------------------------------------

if name == "main": asyncio.run(engine())
main.py (Part 2 of 2)

-------------------------------------------------------------

CLEAN BATCH SENDING • SPAM PROTECTION • MODE TIMING CONTROL

ADVANCED ENGINE LOOP (OPTIMIZED FOR 10–15 ALERTS/DAY)

-------------------------------------------------------------

import asyncio import logging import time from typing import List, Dict, Any

from helpers import run_all_modes, format_signal, multi_override_watch import aiohttp

-------------------------------------------------------------

TELEGRAM CONFIG

-------------------------------------------------------------

TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN" TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

async def tg_send(msg: str): url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg} try: async with aiohttp.ClientSession() as session: await session.post(url, data=payload) except Exception as e: logging.error(f"Telegram error: {e}")

-------------------------------------------------------------

SPAM PROTECTION (same pair within 5 minutes)

-------------------------------------------------------------

LAST_ALERT: Dict[str, int] = {}

def can_alert(symbol: str) -> bool: now = int(time.time()) if symbol not in LAST_ALERT: return True return (now - LAST_ALERT[symbol]) > 300

def mark_alert(symbol: str): LAST_ALERT[symbol] = int(time.time())

-------------------------------------------------------------

BATCH FORMATTED SIGNAL SENDER

-------------------------------------------------------------

async def send_signals(signals: List[Dict[str, Any]]): for sig in signals: if not can_alert(sig['symbol']): continue msg = format_signal(sig) await tg_send(msg) mark_alert(sig['symbol']) await asyncio.sleep(1)  # smooth pacing

-------------------------------------------------------------

ADVANCED ENGINE – 20 SEC LOOP • 10–15 ALERT/DAY

-------------------------------------------------------------

async def engine(): logging.info("🚀 Engine Running — Balanced Mode (10–15 alerts/day)")

while True:
    try:
        all_modes = await run_all_modes()

        # merge signals
        signals = (
            all_modes.get("quick", []) +
            all_modes.get("mid", []) +
            all_modes.get("trend", [])
        )

        # send signals
        if len(signals) > 0:
            await send_signals(signals)

            # override danger watch
            danger = await multi_override_watch(signals)
            for d in danger:
                await tg_send(d)

    except Exception as e:
        logging.error(f"ENGINE ERROR: {e}")

    await asyncio.sleep(20)

-------------------------------------------------------------

RUN

-------------------------------------------------------------

if name == "main": asyncio.run(engine())