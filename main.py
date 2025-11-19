# main.py — FINAL run loop + loop logging + emit-limit + cooldown check
import time, logging
from helpers import (
    get_exchange,
    analyze_coin,
    format_signal_message,
    send_telegram_message,
    cooldown_key_for,
    _cd_mgr,
    MAX_EMITS_PER_LOOP,
)
from math import ceil

LOG = logging.getLogger("main")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

SCAN_BATCH = int(20)
SLEEP = float(5.0)

def main_loop():
    ex = get_exchange()
    LOG.info("🚀 ArunBot Pro Scanner Started")
    loop_no = 0
    while True:
        loop_no += 1
        try:
            markets = list(ex.markets.keys())
            symbols = [s for s in markets if s.endswith("/USDT")]
            total = len(symbols)
            if total == 0:
                LOG.warning("No symbols found in exchange.markets")
                time.sleep(10); continue

            # split into parts to show progress
            parts = range(0, total, SCAN_BATCH)
            for part_idx, start in enumerate(parts):
                end = min(start + SCAN_BATCH, total)
                batch = symbols[start:end]
                emits = 0
                LOG.info(f"Loop #{loop_no} part idx={start} | scanned={len(batch)} emitted={emits} sleeping={SLEEP}")

                for sym in batch:
                    if emits >= MAX_EMITS_PER_LOOP:
                        break
                    try:
                        sig = analyze_coin(ex, sym)
                        if not sig:
                            continue
                        key = cooldown_key_for(sig['pair'], sig['mode'])
                        if _cd_mgr.is_cooled(key):
                            LOG.info("Cooldown active skip %s %s", sig['pair'], sig['mode']); continue
                        # set cooldown
                        _cd_mgr.set_cooldown(key, sig['mode'])
                        # send
                        msg = format_signal_message(sig)
                        ok = send_telegram_message(msg)
                        if ok:
                            emits += 1
                            LOG.info(f"EMIT → {sig['pair']} | mode={sig['mode']} score={sig['score']}")
                    except Exception as e:
                        LOG.exception("Error processing %s: %s", sym, e)

                LOG.info(f"Loop #{loop_no} part idx={start} | scanned={len(batch)} emitted={emits} sleeping={SLEEP}")
                time.sleep(SLEEP)

        except Exception as e:
            LOG.exception("Main loop error: %s", e)
            time.sleep(5)

if __name__ == "__main__":
    main_loop()