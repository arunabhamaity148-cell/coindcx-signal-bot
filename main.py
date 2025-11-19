import time
import logging
import os
import threading
import requests

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
SCAN_BATCH_SIZE = 20
LOOP_SLEEP_SECONDS = 5
MAX_EMITS_PER_LOOP = 1


# ---------------------------------------------------
# 🔥 SELF KEEP-ALIVE (Railway Idle বন্ধ করবে)
# ---------------------------------------------------
def self_keepalive():
    try:
        url = os.getenv("RAILWAY_PUBLIC_URL", "")
        if url:
            requests.get(url, timeout=5)     # hit itself
            LOG.info("🔄 KeepAlive Ping Sent")
    except Exception as e:
        LOG.warning(f"KeepAlive Error: {e}")

    threading.Timer(120, self_keepalive).start()  # every 2 min


# ---------------------------------------------------
# 🔥 MAIN SCANNER LOOP
# ---------------------------------------------------
def main_loop():
    ex = get_exchange()
    LOG.info("🚀 ArunBot Pro Scanner Started")

    while True:
        try:
            markets = list(ex.markets.keys())
            symbols = [s for s in markets if s.endswith("/USDT")]
            total = len(symbols)

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


# ---------------------------------------------------
# 🔥 START
# ---------------------------------------------------
if __name__ == "__main__":
    # keepalive start here
    self_keepalive()

    # main scanner start
    main_loop()