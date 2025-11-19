# main.py — clean version without cooldown
from __future__ import annotations
import os, time, csv, logging

from helpers import (
    analyze_symbol_sync,
    format_signal_message,
    send_telegram_message,
    preview_signal_log,
    get_exchange,   # this exists in your helpers
    SCAN_BATCH_SIZE,
    LOOP_SLEEP_SECONDS,
    MAX_EMITS_PER_LOOP,
)

LOG = logging.getLogger("main")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel("INFO")


def load_coins(path="coins.csv"):
    arr = []
    if not os.path.exists(path):
        LOG.error("coins.csv missing")
        return arr
    with open(path) as f:
        for row in csv.reader(f):
            if row and row[0].upper() != "SYMBOL":
                arr.append(row[0].upper())
    LOG.info("Loaded %d coins", len(arr))
    return arr


def main_loop():
    coins = load_coins()
    if not coins:
        return

    ex = get_exchange()

    loop_id = 0
    batch = SCAN_BATCH_SIZE
    sleep_s = LOOP_SLEEP_SECONDS
    max_emit = MAX_EMITS_PER_LOOP

    LOG.info("Scanner started | batch=%d sleep=%.1f max_emit=%d",
             batch, sleep_s, max_emit)

    while True:
        loop_id += 1
        emitted = 0
        scanned = 0

        for start_idx in range(0, len(coins), batch):
            batch_coins = coins[start_idx:start_idx + batch]
            LOG.info("Loop %d | start_idx=%d | scanning=%d",
                     loop_id, start_idx, len(batch_coins))

            for sym in batch_coins:
                scanned += 1
                sig = analyze_symbol_sync(ex, sym)
                if not sig:
                    continue

                preview_signal_log(sig)

                msg = format_signal_message(sig)
                send_telegram_message(msg, preview=True)

                emitted += 1
                if emitted >= max_emit:
                    break

            LOG.info("Loop %d | batch done | scanned=%d emitted=%d sleeping=%.1f",
                     loop_id, scanned, emitted, sleep_s)

            if emitted >= max_emit:
                break

            time.sleep(sleep_s)

        LOG.info("Loop %d finished | total_scanned=%d total_emitted=%d => sleeping %.1f",
                 loop_id, scanned, emitted, sleep_s)

        time.sleep(sleep_s)


if __name__ == "__main__":
    main_loop()