# main.py (UPDATED for 100/100 helpers)
import os
import time
import signal
import logging
from typing import List

# import helpers (single merged helpers.py expected)
from helpers import (
    create_exchange,
    load_coins,
    analyze_and_emit,
)

# config
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("main")

EXCHANGE_ID = os.environ.get("EXCHANGE_ID", "binance")
COINS_FILE = os.environ.get("COINS_CSV", "coins.csv")
LOOP_INTERVAL = float(os.environ.get("LOOP_INTERVAL", "15"))   # seconds
MAX_COINS_PER_LOOP = int(os.environ.get("MAX_COINS_PER_LOOP", "12"))
BALANCE_FOR_RISK = float(os.environ.get("BALANCE_FOR_RISK", "100"))  # used for position sizing
RISK_PCT = float(os.environ.get("RISK_PCT", "1.0"))  # percent per trade

_shutdown = False
def _signal(sig, frame):
    global _shutdown
    logger.info("Shutdown signal received")
    _shutdown = True

signal.signal(signal.SIGINT, _signal)
signal.signal(signal.SIGTERM, _signal)

def main():
    logger.info("Starting upgraded bot main")
    cfg = None
    try:
        cfg = type("C", (), {})()
        cfg.id = EXCHANGE_ID
        cfg.api_key = os.environ.get("EXCHANGE_API_KEY", None)
        cfg.api_secret = os.environ.get("EXCHANGE_API_SECRET", None)
        cfg.enable_rate_limit = True
        cfg.options = {}
        ex = create_exchange(cfg)
    except Exception as e:
        logger.exception("Exchange init failed: %s", e)
        return

    coins = load_coins(COINS_FILE)
    logger.info("Loaded %d coins", len(coins))

    loop = 0
    while not _shutdown:
        loop += 1
        start = time.time()
        logger.info("Loop #%d — scanning up to %d coins", loop, MAX_COINS_PER_LOOP)
        to_check = coins[:MAX_COINS_PER_LOOP]
        emitted = 0
        for sym in to_check:
            if _shutdown:
                break
            try:
                res = analyze_and_emit(
                    ex,
                    sym,
                    timeframe=os.environ.get("TIMEFRAME", "1m"),
                    limit=int(os.environ.get("OHLCV_LIMIT", "300")),
                    pair_suffix=os.environ.get("PAIR_SUFFIX", "/USDT"),
                    balance_for_risk=BALANCE_FOR_RISK,
                    risk_pct=RISK_PCT,
                    send_telegram_flag=os.environ.get("SEND_TELEGRAM", "1") == "1"
                )
                if res:
                    emitted += 1
                    logger.info("Signal: %s %s lev=%sx score=%.2f", res['symbol'], res['mode'], res['leverage'], res['score'])
                # small pause
                time.sleep(0.5)
            except Exception:
                logger.exception("Error processing %s", sym)
        elapsed = time.time() - start
        sleep_for = max(1.0, LOOP_INTERVAL - elapsed)
        logger.info("Loop done: scanned=%d emitted=%d elapsed=%.2fs sleeping=%.2fs", len(to_check), emitted, elapsed, sleep_for)
        slept = 0.0
        while slept < sleep_for and not _shutdown:
            time.sleep(0.5)
            slept += 0.5

    logger.info("Exiting main cleanly")

if __name__ == "__main__":
    main()