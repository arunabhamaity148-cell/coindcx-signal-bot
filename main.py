import time
from helpers import load_symbols, get_ohlcv_sample


def main():
    print("INFO: Bot starting (minimal version)…")

    symbols = load_symbols()
    print("Loaded coins:", symbols)

    # sample fetch
    for s in symbols:
        print("\nChecking:", s)
        candle = get_ohlcv_sample(s)

        if candle is None:
            print("ERROR: No data for", s)
        else:
            print("Last Candle:", candle)

        time.sleep(1)

    print("INFO: Bot loop finished.")


if __name__ == "__main__":
    main()