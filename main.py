# ============================================================
# main.py — FINAL SAFE PRO SCANNER (NO REDIS, SYNC ENGINE)
# Matching helpers.py (final clean version)
# ============================================================

import time
import logging
import helpers

LOG = logging.getLogger("main")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel("INFO")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
COINS = [
    "BTC", "ETH", "XRP", "SOL", "ADA", "LTC", "BNB", "DOGE", "DOT", "MATIC",
    "LINK", "BCH", "ETC", "ATOM", "FIL", "AAVE", "GRT", "SAND", "EGLD", "VET",
    "CRV", "NEAR", "AR", "PEPE", "FTM", "RUNE", "INJ", "KAVA", "OP", "TIA",
    "CHZ", "HBAR", "FLOW", "TRX", "TON", "SXP", "XLM", "RNDR", "SEI", "PYTH",
]

SCAN_DELAY_SEC = 5        # প্রতি loop-এ ৫ সেকেন্ড gap
BTC_CHECK_EVERY = 15      # প্রতি ১৫ সেকেন্ডে BTC stable check

# ------------------------------------------------------------
# BOT LOOP
# ------------------------------------------------------------
def main_loop():

    LOG.info("🚀 ArunBot Pro Scanner Started")

    ex = helpers.get_ex()

    last_btc_check = 0
    btc_ok = True

    while True:
        try:
            now = time.time()

            # --------------------------------------
            # BTC STABILITY CHECK
            # --------------------------------------
            if now - last_btc_check > BTC_CHECK_EVERY:
                btc_ok = helpers.is_btc_stable(ex)
                last_btc_check = now
                if not btc_ok:
                    LOG.info("⚠️ BTC unstable — pausing alerts")

            # --------------------------------------
            # SCAN COINS
            # --------------------------------------
            emits_this_loop = 0

            for sym in COINS:

                if emits_this_loop >= 1:   # max 1 signal per loop
                    break

                # BTC unstable হলে skip
                if not btc_ok:
                    continue

                sig = helpers.analyze(ex, sym)
                if sig:
                    helpers.tg_send(helpers.tg_format(sig))
                    helpers.preview_signal_log(sig)  # console preview
                    emits_this_loop += 1
                    time.sleep(1.2)  # slight pause after signal

                time.sleep(0.3)  # per-symbol pause

            time.sleep(SCAN_DELAY_SEC)

        except KeyboardInterrupt:
            LOG.info("User stopped manually.")
            break

        except Exception as e:
            LOG.error(f"Loop error: {e}")
            time.sleep(2)

# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":
    main_loop()