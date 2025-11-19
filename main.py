# main.py
import time, logging
from datetime import datetime
from helpers import (
    get_exchange,
    analyze_coin,
    format_signal_message,
    send_telegram_message,
    cooldown_key_for,
    _cd,  # cooldown manager instance from helpers
)

LOG = logging.getLogger("main")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

SCAN_BATCH = int(__import__("os").environ.get("SCAN_BATCH_SIZE", "20"))
SLEEP = float(__import__("os").environ.get("LOOP_SLEEP_SECONDS", "5"))
MAX_EMITS = int(__import__("os").environ.get("MAX_EMITS_PER_LOOP", "1"))

def main_loop():
    ex = get_exchange()
    LOG.info("🚀 ArunBot Pro Scanner Started")
    loop_count = 0

    while True:
        loop_count += 1
        start_ts = time.time()
        try:
            markets = list(ex.markets.keys())
            symbols = [s for s in markets if s.endswith("/USDT")]

            total_scanned = 0
            total_emitted = 0

            for idx in range(0, len(symbols), SCAN_BATCH):
                batch = symbols[idx:idx + SCAN_BATCH]
                emits = 0
                for sym in batch:
                    total_scanned += 1
                    sig = analyze_coin(ex, sym)
                    if sig:
                        key = cooldown_key_for(sig)
                        if _cd.is_cooled(key):
                            LOG.debug("Cooled: %s", key)
                            continue
                        if emits >= MAX_EMITS:
                            LOG.debug("Max emits reached for this loop")
                            continue
                        # emit
                        msg = format_signal_message(sig)
                        ok = send_telegram_message(msg)
                        if ok:
                            _cd.set_cooldown(key, sig.get("mode","MID"))
                            emits += 1
                            total_emitted += 1
                            LOG.info("EMIT → %s | mode=%s score=%.1f", sig.get("pair"), sig.get("mode"), sig.get("score"))
                LOG.info("Loop #%d part idx=%d | scanned=%d emitted=%d sleeping=%.1f",
                         loop_count, idx, len(batch), emits, SLEEP)
                time.sleep(SLEEP)

            elapsed = time.time() - start_ts
            LOG.info("LOOP FINISHED #%d | scanned_total=%d emitted_total=%d elapsed=%.2fs",
                     loop_count, total_scanned, total_emitted, elapsed)

        except Exception as e:
            LOG.exception("Main loop error: %s", e)
            time.sleep(3)

if __name__ == "__main__":
    main_loop()