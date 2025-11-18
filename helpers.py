import ccxt
import csv
import time
import datetime as dt


def load_symbols():
    """Load coin symbols from CSV (header auto-skip)."""
    symbols = []
    with open("coins.csv") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header row

        for row in reader:
            if not row:
                continue
            sym = row[0].strip()
            if sym.lower() == "symbol":
                continue
            symbols.append(sym)

    return symbols


def get_exchange():
    """Load Binance exchange for OHLCV."""
    ex = ccxt.binance({
        "enableRateLimit": True,
    })
    return ex


def fetch_ohlcv_safe(ex, symbol, timeframe="1m", limit=50, retries=3):
    """Fetch OHLCV with retry + error handling."""
    for i in range(retries):
        try:
            pair = f"{symbol}/USDT"
            return ex.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        except Exception as e:
            print("Retry", i+1, "for", symbol, "error:", e)
            time.sleep(2)
    return None


def get_ohlcv_sample(symbol):
    """Return last OHLCV candle for the symbol."""
    ex = get_exchange()
    ohlcv = fetch_ohlcv_safe(ex, symbol)

    if ohlcv is None:
        print("ERROR: No OHLCV for", symbol)
        return None

    return ohlcv[-1]  # last candle


def format_time(ts):
    """Convert timestamp to readable IST."""
    return dt.datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M:%S")