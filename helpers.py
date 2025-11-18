# helpers.py
import csv
import ccxt
import time

DEFAULT_EXCHANGE = 'binance'

def load_coins(path='coins.csv'):
    coins = []
    try:
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                symbol = row[0].strip()
                if symbol:
                    coins.append(symbol)
    except FileNotFoundError:
        # file may be missing in quick test
        return []
    return coins


def get_exchange(name=DEFAULT_EXCHANGE):
    # create a public ccxt exchange instance
    exchange_class = getattr(ccxt, name)
    ex = exchange_class({'enableRateLimit': True})
    return ex


def get_ohlcv_sample(symbol, timeframe='1m', limit=1, exchange_name=DEFAULT_EXCHANGE):
    ex = get_exchange(exchange_name)
    # ensure symbol uses exchange format (e.g., BTC/USDT)
    if '/' not in symbol and symbol.endswith('USDT'):
        pair = symbol[:-4] + '/USDT'
    elif '/' not in symbol:
        pair = symbol + '/USDT'
    else:
        pair = symbol

    # fetch ohlcv (public endpoint)
    ohlcv = ex.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
    if not ohlcv:
        raise ValueError('No OHLCV returned')
    return ohlcv[0]