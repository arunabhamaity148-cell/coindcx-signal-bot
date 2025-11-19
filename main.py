import time
import logging
from datetime import datetime
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

SCAN_BATCH = 20
SLEEP = 5


def main_loop():
    ex = get_exchange()
    LOG.info("🚀 ArunBot Pro Scanner Started")

    while True:
        try:
            markets = list(ex.markets.keys())
            symbols = [s for s in markets if s.endswith("/USDT")]

            for idx in range(0, len(symbols), SCAN_BATCH):
                batch = symbols[idx:idx + SCAN_BATCH]

                emits = 0
                for sym in batch:
                    sig = analyze_coin(ex, sym)
                    if sig:
                        emits += 1
                        msg = format_signal_message(sig)
                        send_telegram_message(msg)
                        LOG.info(f"EMIT → {sym} | mode={sig['mode']} score={sig['score']}")

                LOG.info(f"Loop | scanned={len(batch)} emitted={emits} sleeping={SLEEP}")

                time.sleep(SLEEP)

        except Exception as e:
            LOG.error(f"Main loop error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main_loop()