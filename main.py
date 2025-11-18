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


MODES = {
    "quick": 20,
    "mid": 60,
    "trend": 180
}


def run_mode(mode_name):

    print(f"\n=== Running Mode: {mode_name.upper()} ===")

    symbols = load_symbols()
    ex = get_exchange()

    for sym in symbols:

        print(f"\n[Checking] {sym} ({mode_name})")

        raw = fetch_ohlcv_safe(ex, sym, timeframe="1m", limit=200)

        if raw is None:
            print("[SKIP] No data returned")
            continue

        df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])

        data = evaluate_signal(df, side="BUY")

        if data is None:
            print("[NO SIGNAL]")
            continue

        if data["accuracy"] < 60:
            print(f"[SKIP] Low Accuracy {data['accuracy']}%")
            continue

        send_signal(sym, mode_name, "BUY", data)

        time.sleep(1)

    print(f"=== END {mode_name.upper()} MODE ===\n")


def main():
    print("\n🚀 SIGNAL ENGINE STARTED\n")

    while True:
        for mode_name, delay in MODES.items():
            run_mode(mode_name)
            print(f"⏳ Sleeping {delay}s...\n")
            time.sleep(delay)


if __name__ == "__main__":
    main()