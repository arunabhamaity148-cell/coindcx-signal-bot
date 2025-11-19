import os, time, logging, requests, math
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# external
import ccxt

load_dotenv()

LOG = logging.getLogger("helpers")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel("INFO")

# -----------------------
# CONFIG
# -----------------------
EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "binance")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")

MIN_SIGNAL_SCORE = 85
THRESH_QUICK = 92
THRESH_MID   = 85
THRESH_TREND = 72

MIN_VOL = 150000
MAX_SPREAD = 0.004

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1","true")

EMA_FAST = 20
EMA_SLOW = 50
EMA_LONG = 200

# ------------------------------------------
# EXCHANGE
# ------------------------------------------
def get_exchange():
    ex_name = EXCHANGE_NAME.lower()
    if ex_name == "binance":
        ex = ccxt.binance({'enableRateLimit': True})
    else:
        ex = getattr(ccxt, ex_name, ccxt.binance)({'enableRateLimit': True})

    try:
        ex.load_markets()
    except:
        pass
    return ex

def normalize_symbol(s):
    s = s.upper().strip()
    if "/" not in s:
        s = f"{s}/{QUOTE_ASSET}"
    return s

# ------------------------------------------
# FETCH OHLCV
# ------------------------------------------
def df_from_ohlcv(data):
    if not data:
        return None
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

def fetch_ohlcv(ex, symbol, tf="1m", lim=200):
    sym = normalize_symbol(symbol)
    try:
        data = ex.fetch_ohlcv(sym, timeframe=tf, limit=lim)
        time.sleep(0.4)
        return df_from_ohlcv(data)
    except:
        return None

# ------------------------------------------
# INDICATORS
# ------------------------------------------
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/length, adjust=False).mean()
    rs = up/(down+1e-12)
    return 100 - (100/(1+rs))

def atr(df, length=14):
    h = df["high"]; l = df["low"]; c = df["close"]
    prev = c.shift(1)
    tr = pd.concat([
        (h-l),
        (h-prev).abs(),
        (l-prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean().bfill()

# ------------------------------------------
# ORDERBOOK
# ------------------------------------------
def fetch_orderbook(ex, symbol):
    try:
        ob = ex.fetch_order_book(normalize_symbol(symbol), limit=20)
        time.sleep(0.4)
        return ob
    except:
        return {}

def spread_from_ob(ob):
    try:
        bid = ob["bids"][0][0]
        ask = ob["asks"][0][0]
        return abs(ask - bid)/ask
    except:
        return 999

# ------------------------------------------
# BTC STABILITY
# ------------------------------------------
def btc_stable(ex):
    df = fetch_ohlcv(ex, "BTC/USDT", "1m", 30)
    if df is None: return False
    vol = df["close"].pct_change().abs().tail(8).mean()
    return vol < 0.0018

# ------------------------------------------
# SCORE ENGINE
# ------------------------------------------
def compute_score(df, ob_spread):
    reasons = []
    close = df["close"]

    score = 40
    e20 = ema(close, EMA_FAST).iloc[-1]
    e50 = ema(close, EMA_SLOW).iloc[-1]
    e200 = ema(close, EMA_LONG).iloc[-1]

    if e20 > e50:
        score += 12; reasons.append("EMA20>50")
    else:
        reasons.append("EMA20<=50")

    if e20 > e200:
        score += 8; reasons.append("EMA20>200")

    macd_val = (ema(close,12) - ema(close,26)).iloc[-1]
    if macd_val > 0:
        score += 10; reasons.append("MACD_pos")

    r = rsi(close).iloc[-1]
    if 40 < r < 70:
        score += 8; reasons.append("RSI_ok")

    vol = df["volume"]
    avg = vol.rolling(20).mean().iloc[-1]
    if vol.iloc[-1] > avg*1.5:
        score += 10; reasons.append("Vol_spike")

    if ob_spread <= MAX_SPREAD:
        score += 6; reasons.append("Spread_ok")

    return min(score,100), reasons

# ------------------------------------------
# BUILD SIGNAL
# ------------------------------------------
def build_signal(symbol, df, score, reasons):
    close = df["close"].iloc[-1]
    atr_val = atr(df).iloc[-1]

    if score >= THRESH_QUICK: mode = "QUICK"
    elif score >= THRESH_MID: mode = "MID"
    else: mode = "TREND"

    if mode == "QUICK":
        tp = close + atr_val*1.4
        sl = close - atr_val*0.9
    elif mode == "MID":
        tp = close + atr_val*2.0
        sl = close - atr_val*1.2
    else:
        tp = close + atr_val*2.8
        sl = close - atr_val*1.6

    # BUY/SELL detection
    direction = "BUY"
    e20 = ema(df["close"],20).iloc[-1]
    e50 = ema(df["close"],50).iloc[-1]
    if e20 < e50:
        direction = "SELL"
        tp = close - (tp - close)
        sl = close + (close - sl)

    danger_low  = round(close - atr_val*1.1,8)
    danger_high = round(close + atr_val*1.1,8)

    return {
        "pair": symbol,
        "mode": mode,
        "score": score,
        "entry": round(close,8),
        "tp": round(tp,8),
        "sl": round(sl,8),
        "atr": round(atr_val,8),
        "direction": direction,
        "danger": (danger_low, danger_high),
        "reasons": reasons
    }

# ------------------------------------------
# ANALYZE SYMBOL
# ------------------------------------------
def analyze_symbol(ex, symbol):
    sym = normalize_symbol(symbol)

    # Market check
    try:
        if sym not in ex.markets:
            return None
    except:
        pass

    df = fetch_ohlcv(ex, symbol, "1m", 200)
    if df is None or df.empty:
        return None

    # volume last candle
    if df["volume"].iloc[-1] < MIN_VOL:
        return None

    # orderbook spread
    ob = fetch_orderbook(ex, symbol)
    spread = spread_from_ob(ob)
    if spread > MAX_SPREAD:
        return None

    score, reasons = compute_score(df, spread)
    if score < MIN_SIGNAL_SCORE:
        return None

    return build_signal(sym, df, score, reasons)

# ------------------------------------------
# TELEGRAM
# ------------------------------------------
def send_telegram(msg):
    if NOTIFY_ONLY:
        LOG.info("Telegram Preview:\n%s", msg)
        return True

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        LOG.warning("Telegram missing keys")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except:
        return False

# ------------------------------------------
# FORMAT MSG
# ------------------------------------------
def format_msg(sig):
    em = "⚡" if sig["mode"]=="QUICK" else ("🔥" if sig["mode"]=="MID" else "🚀")
    d1, d2 = sig["danger"]

    return (
        f"{em} <b>{sig['direction']} SIGNAL — {sig['mode']}</b>\n"
        f"Pair: <b>{sig['pair']}</b>  Score: <b>{sig['score']}</b>\n"
        f"Entry: <code>{sig['entry']}</code>\n"
        f"TP: <code>{sig['tp']}</code>   SL: <code>{sig['sl']}</code>\n"
        f"Qty: (manual)  Size: (manual)  Lev: (manual)\n"
        f"⚠️ Danger Zone: <code>{d1}</code> — <code>{d2}</code> (ATR={sig['atr']})\n"
        f"Reason: {', '.join(sig['reasons'])}"
    )