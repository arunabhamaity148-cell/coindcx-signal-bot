# main.py — batch runner for helpers.py
import os, time, signal, logging
from typing import List
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import ccxt
from helpers import (
    create_exchange_instance, load_cooldown_map, persist_cooldown_map,
    scan_and_maybe_emit, resolve_symbol_if_needed
)

# logging
logger = logging.getLogger("main")
logger.setLevel(os.getenv("MAIN_LOG_LEVEL","INFO"))
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

# config
EXCHANGE_NAME = os.getenv("EXCHANGE_NAME","binance")
API_KEY = os.getenv("EXCHANGE_API_KEY","") or None
API_SECRET = os.getenv("EXCHANGE_API_SECRET","") or None
COINS_CSV = os.getenv("COINS_CSV","coins.csv")
SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE","20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS","5"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP","3"))
TIMEFRAME = os.getenv("TIMEFRAME","1m")

STOP = False
def _sig(sig, frame):
    global STOP
    STOP = True
    logger.info("Shutdown signal received")
signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)

def load_coins(path: str = "coins.csv") -> List[str]:
    out = []
    try:
        import csv
        with open(path) as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if not row: continue
                out.append(row[0].strip().upper())
    except Exception:
        logger.exception("load_coins failed")
    return out

def main():
    load_cooldown_map()  # ensure cooldown loaded
    ex = create_exchange_instance(EXCHANGE_NAME, API_KEY, API_SECRET)
    coins = load_coins(COINS_CSV)
    if not coins:
        logger.error("No coins found in coins.csv — exiting")
        return
    logger.info("Loaded %d coins", len(coins))

    loop = 0
    while not STOP:
        loop += 1
        start_idx = ((loop - 1) * SCAN_BATCH_SIZE) % len(coins)
        batch = [coins[(start_idx + i) % len(coins)] for i in range(SCAN_BATCH_SIZE)]
        emitted = 0
        for symbol in batch:
            if STOP: break
            # try modes priority quick->mid->trend
            for mode in ("quick","mid","trend"):
                try:
                    # resolve symbol to exchange market if needed
                    pair = resolve_symbol_if_needed(ex, symbol)
                    ok = scan_and_maybe_emit(ex, symbol, mode=mode, timeframe=TIMEFRAME)
                    if ok:
                        emitted += 1
                        logger.info("Emitted for %s mode=%s (loop emitted=%d)", symbol, mode, emitted)
                        break
                except Exception:
                    logger.exception("Error scanning %s", symbol)
            if emitted >= MAX_EMITS_PER_LOOP:
                logger.info("Reached MAX_EMITS_PER_LOOP=%d; breaking batch", MAX_EMITS_PER_LOOP)
                break
        # persist cooldown
        try:
            persist_cooldown_map()
        except Exception:
            logger.exception("persist cooldown failed")
        logger.info("Loop %d done scanned=%d emitted=%d sleeping=%ss", loop, len(batch), emitted, LOOP_SLEEP_SECONDS)
        # sleep in small steps so ctrl-c faster
        slept = 0.0
        while slept < LOOP_SLEEP_SECONDS and not STOP:
            time.sleep(0.5); slept += 0.5

    logger.info("Exiting main")
    try:
        persist_cooldown_map()
    except Exception:
        pass

if __name__ == "__main__":
    main()