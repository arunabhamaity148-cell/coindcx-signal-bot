"""
download_data.py — Download historical klines from Binance
"""
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

def download_klines(symbol, timeframe="1m", days=30):
    """Download historical data"""
    print(f"📥 Downloading {symbol} {timeframe} ({days} days)...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    all_candles = []
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            
            print(f"  {len(all_candles)} candles...", end="\r")
            
            if len(candles) < 1000:
                break
            
            time.sleep(exchange.rateLimit / 1000)
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break
    
    df = pd.DataFrame(all_candles, columns=["t", "o", "h", "l", "c", "v"])
    filename = f"data/{symbol.replace('/', '')}_{timeframe}.csv"
    df.to_csv(filename, index=False)
    print(f"\n✅ Saved {len(df)} candles to {filename}")

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
    
    for sym in symbols:
        download_klines(sym, "1m", days=30)
        time.sleep(2)
    
    print("\n✅ Done! Run: python train_ml.py")