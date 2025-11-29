# guards.py
# News, funding, spread, market-awake filters

import datetime, requests, pytz

# ---------- NEWS GUARD (high-impact USD/EUR/GBP) ----------
def news_guard() -> bool:
    """Return True if high-impact news within ±30 min UTC"""
    now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    start, end = now - datetime.timedelta(minutes=30), now + datetime.timedelta(minutes=30)
    try:
        cal = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=5).json()
        for ev in cal:
            if ev.get("impact") == "High" and ev.get("currency") in ["USD","EUR","GBP"]:
                event_time = datetime.datetime.fromisoformat(ev["date"].replace("Z","")).replace(tzinfo=pytz.utc)
                if start <= event_time <= end:
                    return True
    except: pass
    return False

# ---------- FUNDING GUARD ----------
def funding_filter(symbol: str) -> bool:
    """Skip if funding > 0.01 % (expensive long)"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        r = requests.get(url, timeout=3).json()
        return float(r["lastFundingRate"]) > 0.0001
    except: return False

# ---------- SPREAD GUARD ----------
def spread_guard(symbol: str) -> bool:
    """Skip if bid-ask spread > 0.05 %"""
    try:
        url = "https://api.binance.com/api/v3/ticker/bookTicker"
        r = requests.get(url, params={"symbol": symbol}, timeout=3).json()
        spread = (float(r["askPrice"]) - float(r["bidPrice"])) / float(r["bidPrice"])
        return spread > 0.0005
    except: return False

# ---------- MARKET AWAKE (BTC 1 h ATR) ----------
def market_awake() -> bool:
    """Return False if BTC 1 h ATR < 0.4 % (sleeping)"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 24}
        k = requests.get(url, params=params, timeout=5).json()
        atr = max(float(x[2]) for x in k) - min(float(x[3]) for x in k)
        mid = (float(k[0][4]) + float(k[-1][4])) / 2
        return (atr / mid) >= 0.004
    except: return True   # fail-safe = allow
