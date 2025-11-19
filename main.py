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
    MAX_EMITS_PER_LOOP
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOG = logging.getLogger("main")


def main_loop():
    ex = get_exchange()
    LOG.info("🚀 ArunBot Pro Scanner Started")

    while True:
        try:
            # load markets
            ex.load_markets()
            symbols = [s for s in ex.markets.keys() if s.endswith("/USDT")]

            emits_this_loop = 0

            # batch scanning
            for idx in range(0, len(symbols), SCAN_BATCH_SIZE):

                batch = symbols[idx:idx + SCAN_BATCH_SIZE]
                total_scanned = len(batch)
                batch_emits = 0

                for sym in batch:
                    if emits_this_loop >= MAX_EMITS_PER_LOOP:
                        break

                    sig = analyze_coin(ex, sym)
                    if sig:
                        batch_emits += 1
                        emits_this_loop += 1
                        msg = format_signal_message(sig)
                        send_telegram_message(msg)

                        LOG.info(
                            f"EMIT → {sig['pair']} | {sig['direction']} | "
                            f"mode={sig['mode']} score={sig['score']}"
                        )

                LOG.info(
                    f"Loop part idx={idx} | scanned={total_scanned} | "
                    f"emitted={batch_emits} | sleeping={LOOP_SLEEP_SECONDS}s"
                )

                time.sleep(LOOP_SLEEP_SECONDS)

                # safety: stop scanning if max emits reached
                if emits_this_loop >= MAX_EMITS_PER_LOOP:
                    break

        except Exception as e:
            LOG.error(f"Main loop error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main_loop()