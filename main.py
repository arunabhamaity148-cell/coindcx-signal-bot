import time, logging
from helpers import (
    get_exchange,
    analyze_coin,
    format_signal_message,
    send_telegram_message,
    btc_calm
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
            if not btc_calm(ex):
                LOG.info("⏳ BTC unstable — cooling…")
                time.sleep(8)
                continue

            symbols = [s for s in ex.markets if s.endswith("/USDT")]

            for idx in range(0, len(symbols), SCAN_BATCH):
                batch = symbols[idx:idx+SCAN_BATCH]
                emits = 0

                for sym in batch:
                    sig = analyze_coin(ex, sym)
                    if sig:
                        emits += 1
                        msg = format_signal_message(sig)
                        send_telegram_message(msg)
                        LOG.info(f"EMIT → {sym} | mode={sig['mode']} score={sig['score']}")

                LOG.info(f"Loop {idx//SCAN_BATCH} | scanned={len(batch)} emitted={emits}")
                time.sleep(SLEEP)

        except Exception as e:
            LOG.error(f"Main Error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main_loop()