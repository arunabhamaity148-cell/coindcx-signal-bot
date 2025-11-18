# ==========================================================
# main.py — Works with helpers_part1 + helpers_part2 + helpers_part3
# Stable Version (100/100)
# ==========================================================

import time
import pandas as pd

from helpers_part1 import (
    load_symbols,
    get_exchange,
    fetch_ohlcv_safe,
)

from helpers_part2 import (
    evaluate_signal,
)

from helpers_part3 import (
    send_signal,
)


# ----------------------------------------------------------
# MODE SETTINGS
# ----------------------------------------------------------
MODES = {
    "quick": 20,      # Every 20 seconds
    "mid": 60,        # Every 60 seconds
    "trend": 180      # Every 180 seconds
}


# ----------------------------------------------------------
# RUN A SINGLE MODE
# ----------------------------------------------------------
def run_mode(mode_name):

    print(f"\n=== Running Mode: {mode_name.upper()} ===")

    symbols = load_symbols()
    ex = get_exchange()

    for sym in symbols:

        print(f"\n[Checking] {sym} ({mode_name})")

        # Fetch 1m data
        raw = fetch_ohlcv_safe(ex, sym, timeframe="1m", limit=200)

        if raw is None:
            print("[SKIP] No data returned")
            continue

        df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])

        # Evaluate BUY signal
        data = evaluate_signal(df, side="BUY")

        if data is None:
            print("[NO SIGNAL] Conditions not met")
            continue

        # Accuracy check (filter)
        if data["accuracy"] < 60:
            print(f"[SKIP] Low Accuracy: {data['accuracy']}%")
            continue

        # Send alert to Telegram
        send_signal(sym, mode_name, "BUY", data)

        # Safety delay per coin
        time.sleep(1)

    print(f"=== END {mode_name.upper()} MODE ===\n")


# ----------------------------------------------------------
# MASTER LOOP (24×7)
# ----------------------------------------------------------
def main():
    print("\n🚀 SIGNAL ENGINE STARTED (Stable Version)\n")

    while True:
        for mode_name, delay in MODES.items():
            run_mode(mode_name)
            print(f"⏳ Sleeping {delay} sec...\n")
            time.sleep(delay)


# ----------------------------------------------------------
# START
# ----------------------------------------------------------
if __name__ == "__main__":
    main()