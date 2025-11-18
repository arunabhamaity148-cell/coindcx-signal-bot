# main.py
import time
import logging
import ccxt
from helpers import load_coins, get_ohlcv_sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting coindcx-signal-bot (minimal)")

    # load coins from csv (helpers handles path)
    coins = load_coins('coins.csv')
    logger.info(f"Loaded {len(coins)} coins, sample: {coins[:5]}")

    # simple example: fetch 1 recent candle for first coin on gate or binance
    if coins:
        symbol = coins[0]
        try:
            candle = get_ohlcv_sample(symbol)
            logger.info(f"Sample OHLCV for {symbol}: {candle}")
        except Exception as e:
            logger.exception("Error fetching sample OHLCV")

    # keep process alive (if you want a long-running worker)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down")

if __name__ == '__main__':
    main()