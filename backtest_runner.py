# backtest_runner.py
import os
import asyncio
import pandas as pd

from helpers import backtest_symbol

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
MODE   = os.getenv("MODE", "quick")           # quick / mid / trend
CSV_PATH = os.getenv("CSV_PATH", "data_1m.csv")

async def main():
    print(f"Running backtest: symbol={SYMBOL}, mode={MODE}, csv={CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} file পাওয়া যায়নি. "
            "আগে 1m OHLCV data CSV হিসাবে repo-তে upload করতে হবে।"
        )

    results = await backtest_symbol(SYMBOL, MODE, CSV_PATH)

    if not results:
        print("No trades generated in backtest.")
        return

    df = pd.DataFrame(results)

    total   = len(df)
    tp_hits = (df["hit"] == "TP").sum()
    sl_hits = (df["hit"] == "SL").sum()
    avg_pnl = df["pnl"].mean()

    print("===== BACKTEST SUMMARY =====")
    print(f"Symbol: {SYMBOL}")
    print(f"Mode:   {MODE}")
    print(f"Total trades: {total}")
    print(f"TP hit: {tp_hits} ({tp_hits/total*100:.1f}%)")
    print(f"SL hit: {sl_hits} ({sl_hits/total*100:.1f}%)")
    print(f"Avg PnL: {avg_pnl:.2f}%")

    out_csv = f"backtest_{SYMBOL}_{MODE}.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    asyncio.run(main())