# main.py
# Run: python main.py
# Expects helpers.py (Part1 + Part2) in same repo with functions:
# get_exchange(), load_coins(), run_scan_and_emit(), evaluate_symbol_and_emit()

import os
import time
import signal
import logging
from datetime import datetime
from typing import List

# import helpers (must be the combined helpers.py file)
from helpers import (
    get_exchange,
    load_coins,
    run_scan_and_emit,
    evaluate_symbol_and_emit,
    send_telegram_payload,    # optional: for system alerts
)

# ---------- CONFIG (env / sensible defaults) ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
COINS_CSV = os.getenv("COINS_CSV", "coins.csv")
LOOP_INTERVAL = float(os.getenv("LOOP_INTERVAL", "10"))      # seconds between full scans
BATCH_SLEEP = float(os.getenv("BATCH_SLEEP", "0.4"))         # pause between symbols
MAX_COINS_PER_LOOP = int(os.getenv("MAX_COINS_PER_LOOP", "12"))
MAX_EMIT_PER_LOOP = int(os.getenv("MAX_EMIT_PER_LOOP", "3")) # safety: max signals per loop
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "300"))
PAIR_SUFFIX = os.getenv("PAIR_SUFFIX", "/USDT")
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1").strip() != "0"

# ---------- Logging ----------
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("main")

# ---------- Graceful shutdown ----------
_shutdown = False
def _sig_handler(sig, frame):
    global _shutdown
    logger.info("Received signal %s, shutting down...", sig)
    _shutdown = True

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

# ---------- Main runner ----------
def main():
    logger.info("Starting bot main — exchange=%s timeframe=%s", EXCHANGE_ID, TIMEFRAME)
    # create exchange (public or with API keys from env)
    ex = get_exchange(exchange_id := os.getenv("EXCHANGE_ID", EXCHANGE_ID))
    # load coins
    coins = load_coins(COINS_CSV)
    logger.info("Loaded %d coins from %s", len(coins), COINS_CSV)

    loop_count = 0
    while not _shutdown:
        loop_count += 1
        start = time.time()
        logger.info("Loop #%d start — will scan up to %d coins (max emit %d)", loop_count, MAX_COINS_PER_LOOP, MAX_EMIT_PER_LOOP)

        # reload coins every few loops so we can edit coins.csv without restart
        if loop_count % 6 == 0:
            try:
                coins = load_coins(COINS_CSV)
                logger.info("Reloaded coin list (%d)", len(coins))
            except Exception as e:
                logger.exception("Failed to reload coins.csv: %s", e)

        # take top N
        to_check = coins[:MAX_COINS_PER_LOOP]
        emitted_count = 0

        for symbol in to_check:
            if _shutdown:
                break
            try:
                if emitted_count >= MAX_EMIT_PER_LOOP:
                    logger.info("Reached max_emit_per_loop=%d — skipping remaining symbols this loop", MAX_EMIT_PER_LOOP)
                    break

                # evaluate and emit via helpers.run_scan_and_emit (it handles cooldown + telegram)
                # we call evaluate_symbol_and_emit directly to control per-symbol emitting (safe)
                payload = evaluate_symbol_and_emit(
                    ex,
                    symbol,
                    timeframe=TIMEFRAME,
                    limit=OHLCV_LIMIT,
                    pair_suffix=PAIR_SUFFIX,
                    send_telegram_flag=SEND_TELEGRAM
                )

                if payload:
                    emitted_count += 1
                    logger.info("Signal emitted for %s | mode=%s | score=%.2f | telegram=%s",
                                payload.get("symbol"), payload.get("mode"), payload.get("score"),
                                payload.get("telegram_ok"))
                # safety small sleep between symbol checks
                time.sleep(BATCH_SLEEP)

            except Exception as e:
                logger.exception("Error processing %s: %s", symbol, e)
                # optional: notify owner via telegram about critical errors (best effort)
                try:
                    if SEND_TELEGRAM:
                        send_telegram_payload({
                            "symbol": symbol,
                            "entry": None,
                            "tp": None,
                            "sl": None,
                            "mode": "SYSTEM",
                            "leverage": None,
                            "score": 0,
                            "reasons": [f"Main loop exception: {str(e)[:150]}"]
                        })
                except Exception:
                    logger.exception("Failed to send system alert")

        elapsed = time.time() - start
        sleep_for = max(0.0, LOOP_INTERVAL - elapsed)
        logger.info("Loop #%d done: scanned=%d emitted=%d elapsed=%.2fs sleeping=%.2fs",
                    loop_count, len(to_check), emitted_count, elapsed, sleep_for)

        # graceful sleep to allow KeyboardInterrupt to be caught
        slept = 0.0
        while slept < sleep_for and not _shutdown:
            time.sleep(0.5)
            slept += 0.5

    logger.info("Shutdown complete — exiting main")

if __name__ == "__main__":
    main()