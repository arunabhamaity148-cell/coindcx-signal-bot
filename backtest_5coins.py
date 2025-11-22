# backtest_5coins.py
import asyncio, pandas as pd, os
from helpers import backtest_symbol, send_telegram

COIN_LIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
MODES = ["quick", "mid", "trend"]

async def run():
    summary = []
    for mode in MODES:
        for coin in COIN_LIST:
            path = f"data/{coin.lower()}_15m_30d.csv"
            if not os.path.exists(path):
                continue
            res = await backtest_symbol(coin, mode, path)
            if not res:
                continue
            df = pd.DataFrame(res)
            total = len(df)
            win = (df["hit"] == "TP").sum()
            win_rate = win / total * 100 if total else 0
            avg_pnl = df["pnl"].mean()
            summary.append({
                "coin": coin,
                "mode": mode,
                "total": total,
                "win": win,
                "win_rate": round(win_rate, 2),
                "avg_pnl": round(avg_pnl, 3)
            })

    df_sum = pd.DataFrame(summary)
    for mode in MODES:
        part = df_sum[df_sum["mode"] == mode] if not df_sum.empty else pd.DataFrame()
        if part.empty:
            continue
        msg = f"📊 <b>Backtest ({mode.upper()})</b>\n" \
              f"🎯 Signals: {part['total'].sum()}\n" \
              f"✅ Win: {part['win_rate'].mean():.1f}%\n" \
              f"📈 Avg PnL: {part['avg_pnl'].mean():.2f}%"
        await send_telegram(msg)
        top = part.nlargest(3, "win_rate")
        for _, r in top.iterrows():
            await send_telegram(f"🏆 {r['coin']} – {r['win_rate']}% Win, {r['avg_pnl']}% Avg")

if __name__ == "__main__":
    asyncio.run(run())
