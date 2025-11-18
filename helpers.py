import ccxt
import pandas as pd
import time
import datetime as dt


def fetch_ohlcv(exchange, symbol, timeframe="1m", limit=200):
    for _ in range(5):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            return df
        except Exception:
            time.sleep(1)
    return None


def ema(df, period):
    return df["close"].ewm(span=period, adjust=False).mean()


def check_signal(df):
    df["ema20"] = ema(df, 20)
    df["ema50"] = ema(df, 50)
    df["ema200"] = ema(df, 200)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if last["ema20"] > last["ema50"] > last["ema200"] and prev["ema20"] <= prev["ema50"]:
        return "BUY"

    if last["ema20"] < last["ema50"] < last["ema200"] and prev["ema20"] >= prev["ema50"]:
        return "SELL"

    return None