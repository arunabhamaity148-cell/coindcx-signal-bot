import time
import logging
from helpers import get_exchange, scan_batch

SCAN_BATCH = 20
SLEEP = 5

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
            markets = list(ex.markets.keys())
            symbols = [s for s in markets if s.endswith("/USDT")]

            for idx in range(0, len(symbols), SCAN_BATCH):
                batch = symbols[idx:idx + SCAN_BATCH]

                emits = scan_batch(ex, batch)
                LOG.info(f"Loop part idx={idx} | scanned={len(batch)} emitted={emits} sleeping={SLEEP}")

                time.sleep(SLEEP)

        except Exception as e:
            LOG.error(f"Main loop error: {e}")
            time.sleep(3)

if __name__ == "__main__":
    main_loop()