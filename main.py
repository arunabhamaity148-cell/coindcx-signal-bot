# main.py — FINAL PRO (for your 100/100 helpers.py)

import time
import logging
from datetime import datetime

from helpers import (
    get_exchange,
    analyze_coin,
    format_signal_message,
    send_telegram_message,
    SCAN_BATCH_SIZE,
    LOOP_SLEEP_SECONDS,
    MAX_EMITS_PER_LOOP,
)

# --------------------------
# LOGGING
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOG = logging.getLogger("main")


# --------------------------
# MAIN LOOP
# --------------------------
def main_loop():
    ex = get_exchange()
    LOG.info("🚀 ArunBot PRO Started — Safe Setup 100/100")

    loop_id = 0

    while True:
        try:
            loop_id += 1

            # all USDT pairs
            markets = list(ex.markets.keys())
            symbols = [s for s in markets if s.endswith("/USDT")]

            LOG.info(f"🔄 Loop {loop_id} | Total Symbols: {len(symbols)}")

            # scan in batches
            for idx in range(0, len(symbols), SCAN_BATCH_SIZE):
                batch = symbols[idx:idx + SCAN_BATCH_SIZE]
                emits = 0

                LOG.info(f"📦 Batch {idx//SCAN_BATCH_SIZE + 1} | Size={len(batch)}")

                for sym in batch:
                    sig = analyze_coin(ex, sym)

                    if sig:
                        emits += 1
                        msg = format_signal_message(sig)
                        send_telegram_message(msg)

                        LOG.info(
                            f"📡 EMIT → {sym} | "
                            f"{sig['direction']} | "
                            f"mode={sig['mode']} | "
                            f"score={sig['score']}"
                        )

                        # prevent too many signals
                        if emits >= MAX_EMITS_PER_LOOP:
                            LOG.info("🛑 MAX_EMITS_PER_LOOP reached — stopping batch")
                            break

                LOG.info(
                    f"🧭 Loop {loop_id} Batch result → scanned={len(batch)} | "
                    f"emitted={emits} | sleep={LOOP_SLEEP_SECONDS}s"
                )

                time.sleep(LOOP_SLEEP_SECONDS)

        except Exception as e:
            LOG.error(f"❌ Main loop error: {e}")
            time.sleep(3)


# --------------------------
# ENTRY POINT
# --------------------------
if __name__ == "__main__":
    main_loop()