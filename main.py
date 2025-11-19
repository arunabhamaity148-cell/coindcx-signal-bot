import time
import logging
from helpers import (
    get_exchange,
    analyze_symbol,          # ← FIXED
    format_signal_message,
    send_telegram_message,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOG = logging.getLogger("main")

SCAN_BATCH = 20        # প্রতি লুপে কয়টা কয়েন
SLEEP = 5              # প্রতি ব্যাচের মধ্যে sleep


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
                LOG.info(f"Loop {idx//SCAN_BATCH + 1} | scanning batch size={len(batch)}")

                for sym in batch:
                    sig = analyze_symbol(ex, sym)   # ← FIXED
                    if sig:
                        emits += 1
                        msg = format_signal_message(sig)
                        send_telegram_message(msg)
                        LOG.info(f"EMIT → {sym} | mode={sig['mode']} score={sig['score']}")

                LOG.info(f"Loop done | scanned={len(batch)} emitted={emits} | sleeping {SLEEP}s")
                time.sleep(SLEEP)

        except Exception as e:
            LOG.error(f"Main loop error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main_loop()