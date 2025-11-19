import time
import logging
from helpers import (
    get_exchange,
    analyze_coin,
    format_signal_message,
    send_telegram_message,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOG = logging.getLogger("main")

# Scan settings
SCAN_BATCH_SIZE = 20        # একেকবারে কয়টা coin scan করবে
LOOP_SLEEP_SECONDS = 5      # batch scan → sleep → next batch
MAX_EMITS_PER_LOOP = 1      # প্রতি loop এ কয়টা signal allow


def main_loop():
    ex = get_exchange()
    LOG.info("🚀 ArunBot Pro Scanner Started")

    while True:
        try:
            # সব market names load করা
            markets = list(ex.markets.keys())
            symbols = [s for s in markets if s.endswith("/USDT")]

            total = len(symbols)

            # batch-by-batch scan
            for idx in range(0, total, SCAN_BATCH_SIZE):
                batch = symbols[idx:idx + SCAN_BATCH_SIZE]

                emits = 0

                for sym in batch:
                    sig = analyze_coin(ex, sym)

                    if sig:
                        emits += 1
                        msg = format_signal_message(sig)
                        send_telegram_message(msg)

                        LOG.info(
                            f"EMIT → {sym} | mode={sig['mode']} | "
                            f"dir={sig['direction']} | score={sig['score']}"
                        )

                        # এক loop এ একটার বেশি signal allow না
                        if emits >= MAX_EMITS_PER_LOOP:
                            break

                LOG.info(
                    f"Loop | part idx={idx} | scanned={len(batch)} | "
                    f"emitted={emits} | sleep={LOOP_SLEEP_SECONDS}"
                )

                time.sleep(LOOP_SLEEP_SECONDS)

        except Exception as e:
            LOG.error(f"Main loop error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main_loop()