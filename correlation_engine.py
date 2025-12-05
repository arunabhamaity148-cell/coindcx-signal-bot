import numpy as np

def calc_return_series(prices):
    return np.diff(prices) / prices[:-1]


def correlation_with_btc(sym_close, btc_close):
    if len(sym_close) < 20 or len(btc_close) < 20:
        return 0

    r1 = calc_return_series(sym_close)
    r2 = calc_return_series(btc_close)

    corr = np.corrcoef(r1[-50:], r2[-50:])[0][1]
    return float(corr)