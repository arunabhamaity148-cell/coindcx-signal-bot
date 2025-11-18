# main.py — BUY+SELL ready runner (paste into repo)
from __future__ import annotations
import os
import time
import signal
import logging
import csv
from typing import List

# optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# import helpers (expects the helpers.py you pasted earlier)
from helpers import (
    create_exchange_instance,
    load_cooldown_map,
    persist_cooldown_map,
    scan_and_maybe_emit,
    resolve_symbol_if_needed,
)

# ---------------- logging ----------------
logger = logging.getLogger("main")
logger.setLevel(os.getenv("MAIN_LOG_LEVEL", "INFO"))
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

# ---------------- config (env / defaults) ----------------
EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "binance")
API_KEY = os.getenv("EXCHANGE_API_KEY", "") or None
API_SECRET = os.getenv("EXCHANGE_API_SECRET", "") or None

COINS_CSV = os.getenv("COINS_CSV", "coins.csv")
SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "3"))
TIMEFRAME = os.getenv("TIMEFRAME", "1m")

MODES = os.getenv("MODES", "quick,mid,trend").split(",")  # order of priority
SIDES = ["buy", "sell"]

COOLDOWN_PERSIST_PATH = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")

# graceful stop
STOP = False
def _signal_handler(sig, frame):
    global STOP
    STOP = True
    logger.info("Shutdown signal received")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------- load coins ----------------
def load_coins(path: str = "coins.csv") -> List[str]:
    out: List[str] = []
    try:
        with open(path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row: continue
                sym = row[0].strip()
                if not sym: continue
                out.append(sym.upper())
    except FileNotFoundError:
        logger.error("coins.csv not found at %s", path)
    except Exception:
        logger.exception("load_coins failed")
    return out

# ---------------- main loop ----------------
def main():
    global STOP
    # load persisted cooldowns
    try:
        load_cooldown_map(COOLDOWN_PERSIST_PATH)
    except Exception:
        logger.debug("No cooldown map loaded / error ignored")

    # create exchange
    try:
        ex = create_exchange_instance(EXCHANGE_NAME, API_KEY, API_SECRET)
    except Exception as e:
        logger.exception("Failed to create exchange instance: %s", e)
        return

    coins = load_coins(COINS_CSV)
    if not coins:
        logger.error("No coins loaded. Put coin list in %s (first column symbol). Exiting.", COINS_CSV)
        return

    logger.info("Loaded %d coins. Scan batch=%d sleep=%ss max_emits=%d", len(coins), SCAN_BATCH_SIZE, LOOP_SLEEP_SECONDS, MAX_EMITS_PER_LOOP)

    loop_count = 0
    try:
        while not STOP:
            loop_count += 1
            start_idx = ((loop_count - 1) * SCAN_BATCH_SIZE) % max(1, len(coins))
            batch = [coins[(start_idx + i) % len(coins)] for i in range(SCAN_BATCH_SIZE)]
            emitted = 0
            logger.info("Loop %d: scanning %d coins (start_idx=%d)", loop_count, len(batch), start_idx)

            for symbol in batch:
                if STOP:
                    break
                # try modes in order, and each mode try both sides
                for mode in MODES:
                    mode = mode.strip().lower()
                    emitted_in_symbol = False
                    for side in SIDES:
                        if STOP:
                            break
                        try:
                            # optional: resolve symbol for exchange if needed (not strictly required by scan function)
                            # pair = resolve_symbol_if_needed(ex, symbol)
                            ok = scan_and_maybe_emit(ex, symbol, mode=mode, side=side, timeframe=TIMEFRAME)
                            if ok:
                                emitted += 1
                                emitted_in_symbol = True
                                logger.info("Emitted signal for %s %s mode=%s (loop emitted=%d)", symbol, side, mode, emitted)
                                # break sides loop once emitted for this symbol and mode
                                break
                        except Exception:
                            logger.exception("Error scanning %s %s %s", symbol, mode, side)
                    if emitted_in_symbol:
                        # don't check further modes for this symbol to avoid spam — move to next symbol
                        break
                    if emitted >= MAX_EMITS_PER_LOOP:
                        break
                if emitted >= MAX_EMITS_PER_LOOP:
                    logger.info("Reached MAX_EMITS_PER_LOOP=%d; pausing scanning this loop", MAX_EMITS_PER_LOOP)
                    break

            # persist cooldowns periodically
            try:
                persist_cooldown_map(COOLDOWN_PERSIST_PATH)
            except Exception:
                logger.exception("Failed to persist cooldown map")

            logger.info("Loop %d done: scanned=%d emitted=%d sleeping=%ss", loop_count, len(batch), emitted, LOOP_SLEEP_SECONDS)

            # sleep small intervals for responsive shutdown
            slept = 0.0
            while slept < LOOP_SLEEP_SECONDS and not STOP:
                time.sleep(0.5)
                slept += 0.5

    except Exception:
        logger.exception("Fatal error in main loop")
    finally:
        logger.info("Shutting down main, persisting cooldown map")
        try:
            persist_cooldown_map(COOLDOWN_PERSIST_PATH)
        except Exception:
            logger.exception("Persist on shutdown failed")

if __name__ == "__main__":
    main()