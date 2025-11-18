# main.py
"""
Main runner for coindcx-signal-bot using helpers.py

Usage:
 - put TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in Railway environment or .env (optional)
 - set NOTIFY_ONLY env var to "True" during testing (default True in helpers)
 - adjust MIN_SIGNAL_SCORE / MODE_THRESHOLDS etc via env or in helpers.py
 - run with python main.py
"""

import os
import time
import signal
import logging
from typing import List

# optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import ccxt
from helpers import (
    load_coins,
    get_exchange,
    scan_symbol,
    emit_signal,
    load_cooldown_map,
    persist_cooldown_map,
)

# ---------------- logging ----------------
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

# ---------------- config (change with env) ----------------
EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "binance")
API_KEY = os.getenv("BINANCE_API_KEY", "") or None
API_SECRET = os.getenv("BINANCE_API_SECRET", "") or None

TIMEFRAME = os.getenv("TIMEFRAME", "1m")
LOOP_SLEEP_SECONDS = int(os.getenv("LOOP_SLEEP_SECONDS", 5))  # seconds between scanning batches
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", 3))  # safety: max signals per loop
SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", 12))  # how many coins per scanning iteration
MODE_ORDER = os.getenv("MODE_ORDER", "quick,mid,trend").split(",")  # modes priority

COOLDOWN_PERSIST_PATH = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")

# graceful stop flag
STOP = False

def sigint_handler(signum, frame):
    global STOP
    logger.info("Signal received, stopping loops...")
    STOP = True

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)

def create_exchange():
    logger.info("Creating exchange: %s (api set: %s)", EXCHANGE_NAME, bool(API_KEY and API_SECRET))
    ex = get_exchange(EXCHANGE_NAME, API_KEY, API_SECRET)
    return ex

def run_main_loop():
    # load persistent cooldowns (if any)
    load_cooldown_map(COOLDOWN_PERSIST_PATH)

    exchange = create_exchange()
    coins = load_coins("coins.csv")
    if not coins:
        logger.warning("No coins loaded from coins.csv. Exiting.")
        return

    # normalize coin list to pairs later
    coins = [c.strip().upper() for c in coins if c.strip()]
    logger.info("Loaded %d coins", len(coins))

    loop_counter = 0
    try:
        while not STOP:
            loop_counter += 1
            logger.info("Loop #%d start - will scan up to %d coins (batch)", loop_counter, SCAN_BATCH_SIZE)

            # slice coins per loop (rotate to avoid always start)
            start_idx = (loop_counter - 1) * SCAN_BATCH_SIZE % max(1, len(coins))
            batch = []
            for i in range(SCAN_BATCH_SIZE):
                batch.append(coins[(start_idx + i) % len(coins)])

            emitted = 0
            for symbol in batch:
                if STOP:
                    break

                # Try each mode in MODE_ORDER. If emit, break to respect MAX_EMITS_PER_LOOP
                for mode in MODE_ORDER:
                    # get indicator + score dict for symbol
                    info = scan_symbol(exchange, symbol, mode=mode, timeframe=TIMEFRAME)
                    # info contains 'score','entry','atr', etc.
                    # Ensure we only emit up to MAX_EMITS_PER_LOOP
                    if info.get("score", 0) <= 0:
                        # skip if no data / bad
                        continue

                    # Attempt to emit signal for this mode
                    success = emit_signal(exchange, symbol, mode, info, size_quote=float(os.getenv("DEFAULT_SIZE_QUOTE", 10.0)), lev=int(os.getenv("DEFAULT_LEV", 1)))
                    if success:
                        emitted += 1
                        logger.info("Emitted signal for %s mode=%s (loop emitted %d)", symbol, mode, emitted)
                        # if we emitted, break mode loop for this symbol (no multi-mode same symbol)
                        break

                    # else continue to next mode
                # stop if emitted too many signals this loop
                if emitted >= MAX_EMITS_PER_LOOP:
                    logger.info("Reached MAX_EMITS_PER_LOOP=%d; pausing scanning this loop", MAX_EMITS_PER_LOOP)
                    break

            logger.info("Loop #%d done: scanned=%d emitted=%d elapsed (sleeping %ss)", loop_counter, len(batch), emitted, LOOP_SLEEP_SECONDS)
            # persist cooldown map periodically
            try:
                persist_cooldown_map(COOLDOWN_PERSIST_PATH)
            except Exception:
                logger.exception("Failed persisting cooldown map")

            # Sleep before next loop (allow small jitter)
            for _ in range(0, LOOP_SLEEP_SECONDS):
                if STOP:
                    break
                time.sleep(1)

    finally:
        logger.info("Shutting down main loop, persisting state...")
        try:
            persist_cooldown_map(COOLDOWN_PERSIST_PATH)
        except Exception:
            logger.exception("Persist failed on shutdown")

if __name__ == "__main__":
    run_main_loop()