# main.py
# Run: python main.py
# Requires: helpers.py (PART1 + PART2), coins.csv, requirements installed

import os
import time
import signal
import logging
from datetime import datetime
from typing import List

# import from your helpers
from helpers import (
    create_exchange,
    load_coins,
    SignalEvaluator,
    CooldownManager,
    CoinLimiter,
    send_telegram_signal,
)

# ---------- Basic logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("coindcx-bot-main")


# ---------- Config (env / defaults) ----------
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID")
COINS_FILE = os.getenv("COINS_FILE", "coins.csv")

# main loop timing (seconds)
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL", "10"))  # how often to scan coins (1m use small)
BATCH_SLEEP = float(os.getenv("BATCH_SLEEP", "0.5"))  # pause between coin checks to avoid rate limits

# maximum coins processed per loop (can tune)
MAX_COINS_PER_LOOP = int(os.getenv("MAX_COINS_PER_LOOP", "12"))

# ---------- Graceful shutdown ----------
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    logger.info("Received shutdown signal: %s", sig)
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------- Main runner ----------
def run_loop():
    logger.info("Starting main loop — Exchange: %s", EXCHANGE_NAME)

    # create exchange connection (helpers.create_exchange should handle keys via env or config)
    ex = create_exchange(EXCHANGE_NAME)

    # cooldown manager and limiter
    cooldown = CooldownManager()
    limiter = CoinLimiter(max_coins=MAX_COINS_PER_LOOP)

    # signal evaluator (will use send_telegram inside)
    evaluator = SignalEvaluator(
        ex,
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat=TELEGRAM_CHAT,
        cooldown=cooldown,
        limiter=limiter,
    )

    # load coin list
    try:
        coins = load_coins(COINS_FILE)
        logger.info("Loaded %d coins", len(coins))
    except Exception as e:
        logger.exception("Failed to load coins file: %s", e)
        coins = []

    loop_count = 0
    while not _shutdown:
        loop_count += 1
        start_ts = time.time()
        logger.info("Loop #%d start — scanning up to %d coins", loop_count, MAX_COINS_PER_LOOP)

        # re-load coins every few loops so you can update file without restart
        if loop_count % 6 == 0:  # every ~6 loops (tunable)
            try:
                coins = load_coins(COINS_FILE)
                logger.info("Reloaded coins list, total %d", len(coins))
            except Exception:
                logger.exception("Reload coins failed")

        # limit coins per loop and respect limiter state
        to_check = coins[:MAX_COINS_PER_LOOP]

        for base_symbol in to_check:
            if _shutdown:
                break
            try:
                logger.debug("Checking %s", base_symbol)
                # evaluate all modes (helpers.SignalEvaluator returns signal dict or None)
                sig = evaluator.evaluate_symbol_all_modes(base_symbol, quote="USDT", timeframe="1m")
                if sig:
                    logger.info("Signal emitted: %s %s %s", sig["mode"], sig["symbol"], sig["side"])
                # small pause to avoid hammering exchange
                time.sleep(BATCH_SLEEP)
            except Exception as e:
                logger.exception("Error evaluating %s: %s", base_symbol, e)
                # if severe error, send Telegram alert to owner (optional)
                try:
                    if TELEGRAM_TOKEN and TELEGRAM_CHAT:
                        send_telegram_signal(
                            TELEGRAM_TOKEN,
                            TELEGRAM_CHAT,
                            {
                                "mode": "SYSTEM",
                                "symbol": base_symbol,
                                "side": "ERROR",
                                "entry": None,
                                "tp": None,
                                "sl": None,
                                "leverage": None,
                                "reason": f"Exception evaluating {base_symbol}: {e}",
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        )
                except Exception:
                    logger.exception("Failed to send error message to Telegram")

        # end of coin loop
        elapsed = time.time() - start_ts
        sleep_for = max(0, LOOP_INTERVAL - elapsed)
        logger.info("Loop #%d finished, elapsed %.2fs, sleeping %.2fs", loop_count, elapsed, sleep_for)
        # graceful sleep with ability to break early
        slept = 0.0
        while slept < sleep_for and not _shutdown:
            time.sleep(0.5)
            slept += 0.5

    logger.info("Main loop exiting cleanly — goodbye.")


if __name__ == "__main__":
    logger.info("coindcx-signal-bot main starting")
    try:
        run_loop()
    except Exception as ex:
        logger.exception("Unhandled exception in main: %s", ex)
    finally:
        logger.info("Shutdown complete")