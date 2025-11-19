# ============================
# helpers.py — FINAL VERSION
# Stable, Redis-free, Lightweight
# ============================

import os, time, math, logging
import pandas as pd
import numpy as np
import ccxt
import requests
from datetime import datetime

LOG = logging.getLogger("helpers")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel("INFO")

# -------- ENV --------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID","")
MIN_SCORE = 85
TH_Q = 92
TH_M = 85
TH_T = 72
COOLDOWN = 30 * 60     # 30 min
_last_send = {}        # cooldown map


# ------- Exchange -------
def get_exchange():
    ex = ccxt.binance({'enableRateLimit': True})
    try:
        ex.load_markets()
    except:
        pass
    return ex


# ------- Indicators -------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff()
    u = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    v = -d.clip(upper=0).ewm(alpha=1/n, adjust=False).mean()
    rs = u / (v + 1e-9)
    return 100 - 100/(1+rs)

def atr(df, n=14):
    h,l,c = df['high'], df['low'], df['close']
    prev = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev).abs(),
        (l - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean().bfill()


# ------- Telegram -------
def send_telegram_message(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        LOG.info("Telegram message preview (not sent)\n" + msg)
        return True
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }, timeout=10)
        return True
    except Exception as e:
        LOG.warning("Telegram send fail: %s", e)
        return False


# ------- BTC Calm -------
def btc_calm(ex):
    try:
        df = ex.fetch_ohlcv("BTC/USDT", "1m", limit=20)
        df = pd.DataFrame(df, columns=["ts","open","high","low","close","vol"])
        df['close'] = pd.to_numeric(df['close'])
        vol = df['close'].pct_change().abs().tail(8).mean()
        return vol < 0.002          # stable
    except:
        return True


# ------- Score -------
def compute_score(df):
    close = df['close']
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1]
    score = 40
    reasons = []

    if e20 > e50: score += 12; reasons.append("EMA20>50")
    if e20 > e200: score += 8; reasons.append("EMA20>200")

    macd = (ema(close,12) - ema(close,26)).iloc[-1]
    if macd > 0: score += 12; reasons.append("MACD_pos")

    r = rsi(close).iloc[-1]
    if 40 < r < 70: score += 8; reasons.append("RSI_ok")

    vol = df['volume']
    if vol.iloc[-1] > vol.rolling(20).mean().iloc[-1] * 1.5:
        score += 10; reasons.append("Vol_spike")

    return min(score,100), reasons


# ------- TP/SL by MODE -------
def calc_tp_sl(entry, atr_v, mode):
    if mode == "QUICK":   # super fast
        tp = entry + atr_v * 1.2
        sl = entry - atr_v * 0.8
    elif mode == "TREND":
        tp = entry + atr_v * 3.2
        sl = entry - atr_v * 1.8
    else: # MID
        tp = entry + atr_v * 2.1
        sl = entry - atr_v * 1.0
    return round(tp,8), round(sl,8)


# ------- Cooldown -------
def cooled(sym):
    now = time.time()
    if sym not in _last_send: return True
    return (now - _last_send[sym]) > COOLDOWN


# ------- Analyze -------
def analyze_coin(ex, symbol):
    if not cooled(symbol):
        return None

    try:
        df = ex.fetch_ohlcv(symbol, "1m", limit=200)
    except:
        return None

    df = pd.DataFrame(df, columns=["ts","open","high","low","close","volume"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c])

    if df.empty: return None

    score, reasons = compute_score(df)
    if score < MIN_SCORE:
        return None

    # mode select
    if score >= TH_Q: mode = "QUICK"
    elif score >= TH_M: mode = "MID"
    else: mode = "TREND"

    entry = df['close'].iloc[-1]
    atr_v = float(atr(df).iloc[-1])
    tp, sl = calc_tp_sl(entry, atr_v, mode)

    direction = "BUY"
    if ema(df['close'],20).iloc[-1] < ema(df['close'],50).iloc[-1]:
        direction = "SELL"
        tp = entry - (tp-entry)
        sl = entry + (entry-sl)

    _last_send[symbol] = time.time()

    return {
        "symbol": symbol,
        "mode": mode,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "score": score,
        "atr": atr_v,
        "direction": direction,
        "reasons": reasons,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }


# ------- Format Telegram -------
def format_signal_message(sig):
    em = "⚡" if sig["mode"]=="QUICK" else ("🔥" if sig["mode"]=="MID" else "🚀")
    return (
        f"{em} <b>{sig['direction']} SIGNAL — {sig['mode']}</b>\n"
        f"Pair: <b>{sig['symbol']}</b>\n"
        f"Entry: <code>{sig['entry']}</code>\n"
        f"TP: <code>{sig['tp']}</code>   SL: <code>{sig['sl']}</code>\n"
        f"Leverage: <b>{auto_leverage(sig['mode'])}x</b>\n"
        f"Score: <b>{sig['score']}</b>\n"
        f"ATR: {sig['atr']}\n"
        f"Reason: {', '.join(sig['reasons'])}\n"
        f"Time: {sig['time']} UTC"
    )


# ------- Auto Leverage -------
def auto_leverage(mode):
    if mode=="QUICK": return 35
    if mode=="MID":   return 25
    return 10