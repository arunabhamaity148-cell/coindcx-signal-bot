# helpers.py (Part 1 of 5)
# -------------------------------------------------------------
# ASCII-SAFE VERSION (NO UNICODE SYMBOLS)
# BASIC IMPORTS, LOGGING, SESSION, OHLCV, TICKER
# -------------------------------------------------------------

import asyncio
import aiohttp
import logging
import time
import math
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# BINANCE API URL
# -------------------------------------------------------------
BINANCE_BASE = "https://api.binance.com"

# -------------------------------------------------------------
# ASYNC HTTP SESSION
# -------------------------------------------------------------
class HTTPSession:
    _session = None

    @classmethod
    async def get(cls, url):
        if cls._session is None:
            cls._session = aiohttp.ClientSession()
        async with cls._session.get(url) as resp:
            return await resp.json()

    @classmethod
    async def close(cls):
        if cls._session:
            await cls._session.close()
            cls._session = None

# -------------------------------------------------------------
# FETCH OHLCV
# -------------------------------------------------------------
async def fetch_ohlcv(symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        data = await HTTPSession.get(url)
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","trades",
            "taker_base","taker_quote","ignore"
        ])
        df = df[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        logger.error(f"OHLCV fetch failed for {symbol}: {e}")
        return pd.DataFrame()

# -------------------------------------------------------------
# FETCH TICKER + SPREAD
# -------------------------------------------------------------
async def fetch_ticker(symbol: str) -> Dict[str, Any]:
    url = f"{BINANCE_BASE}/api/v3/ticker/bookTicker?symbol={symbol}"
    try:
        return await HTTPSession.get(url)
    except:
        return {}

async def fetch_spread(symbol: str) -> float:
    t = await fetch_ticker(symbol)
    try:
        bid = float(t['bidPrice'])
        ask = float(t['askPrice'])
        return ((ask - bid) / bid) * 100
    except:
        return 999.0

# -------------------------------------------------------------
# FETCH FUNDING RATE
# -------------------------------------------------------------
async def fetch_funding(symbol: str) -> float:
    url = f"{BINANCE_BASE}/fapi/v1/premiumIndex?symbol={symbol}"
    try:
        data = await HTTPSession.get(url)
        return float(data.get("lastFundingRate", 0)) * 100
    except:
        return 0

# -------------------------------------------------------------
# SERVER TIME
# -------------------------------------------------------------
async def fetch_server_time() -> int:
    url = f"{BINANCE_BASE}/api/v3/time"
    try:
        data = await HTTPSession.get(url)
        return data.get("serverTime", 0)
    except:
        return 0

# -------------------------------------------------------------
# INDICATORS: EMA, RSI, MACD, ATR, VWAP
# -------------------------------------------------------------
def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    return macd_line, signal, macd_line - signal

def atr(df: pd.DataFrame, length: int = 14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def vwap(df: pd.DataFrame):
    typical = (df['high'] + df['low'] + df['close']) / 3
    return (typical * df['volume']).cumsum() / df['volume'].cumsum()
# helpers.py (Part 2 of 5)
# -------------------------------------------------------------
# ADVANCED FILTERS: BTC STABILITY + HTF PATTERNS + PRICE ACTION
# -------------------------------------------------------------

import numpy as np
import pandas as pd
from typing import Dict, Any

# -------------------------------------------------------------
# BTC STABILITY CHECK
# -------------------------------------------------------------
async def btc_stable() -> Dict[str, Any]:
    df = await fetch_ohlcv("BTCUSDT", "1m", 50)
    if df.empty:
        return {"ok": False, "reason": "BTC data error"}

    close = df['close']
    high = df['high']
    low = df['low']

    # volatility
    vol = (high.tail(10).max() - low.tail(10).min()) / close.iloc[-1] * 100

    # wick check
    last_high = high.iloc[-1]
    last_low = low.iloc[-1]
    last_close = close.iloc[-1]
    last_open = df['open'].iloc[-1]
    upper_wick = last_high - max(last_close, last_open)
    lower_wick = min(last_close, last_open) - last_low
    wick_pct = (upper_wick + lower_wick) / last_close * 100

    stable = vol < 1 and wick_pct < 0.5

    return {
        "ok": stable,
        "vol": vol,
        "wick": wick_pct
    }

# -------------------------------------------------------------
# HTF CANDLE PATTERN DETECTION (15m + 1h)
# -------------------------------------------------------------
def detect_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"

    o = df['open'].iloc[-1]
    c = df['close'].iloc[-1]
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]

    # basic patterns
    if c > o and (c - o) > (h - l) * 0.6:
        return "bull_engulf"
    if o > c and (o - c) > (h - l) * 0.6:
        return "bear_engulf"
    if (h - max(o, c)) > (c - o) * 2:
        return "shooting_star"
    if (min(o, c) - l) > (c - o) * 2:
        return "hammer"

    return "none"

async def htf_signal(symbol: str) -> Dict[str, str]:
    df15 = await fetch_ohlcv(symbol, "15m", 50)
    df1h = await fetch_ohlcv(symbol, "1h", 50)

    if df15.empty or df1h.empty:
        return {"15m": "none", "1h": "none"}

    p15 = detect_pattern(df15)
    p1h = detect_pattern(df1h)

    return {"15m": p15, "1h": p1h}

# -------------------------------------------------------------
# PRICE ACTION FILTERS
# -------------------------------------------------------------
def liquidity_sweep(df: pd.DataFrame) -> bool:
    if len(df) < 5:
        return False
    w = (df['high'].iloc[-1] - df['low'].iloc[-1])
    body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    return w > body * 3

def ema21_pullback(df: pd.DataFrame) -> bool:
    e21 = ema(df['close'], 21)
    if len(df) < 2:
        return False
    return abs(df['close'].iloc[-1] - e21.iloc[-1]) / df['close'].iloc[-1] < 0.002

def range_break_retest(df: pd.DataFrame) -> bool:
    if len(df) < 20:
        return False
    high_line = df['high'].tail(20).max()
    broke = df['close'].iloc[-1] > high_line
    retest = df['low'].iloc[-1] <= high_line
    return broke and retest

# -------------------------------------------------------------
# ORDER BLOCK + FVG (FAIR VALUE GAP)
# -------------------------------------------------------------
def detect_order_block(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    if df['close'].iloc[-2] < df['open'].iloc[-2] and df['close'].iloc[-1] > df['open'].iloc[-1]:
        return "bull_OB"
    if df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] < df['open'].iloc[-1]:
        return "bear_OB"
    return "none"

def detect_fvg(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    h1 = df['high'].iloc[-3]
    l1 = df['low'].iloc[-3]
    h2 = df['high'].iloc[-1]
    l2 = df['low'].iloc[-1]

    if l1 > h2:
        return "bull_FVG"
    if h1 < l2:
        return "bear_FVG"
    return "none"

# -------------------------------------------------------------
# SPREAD + FUNDING + SESSION
# -------------------------------------------------------------
async def spread_ok(symbol: str) -> bool:
    sp = await fetch_spread(symbol)
    return sp < 0.06

async def funding_ok(symbol: str) -> bool:
    f = await fetch_funding(symbol)
    return abs(f) < 0.02

def session_now(ts: int) -> str:
    h = time.gmtime(ts/1000).tm_hour
    if 1 <= h < 8:
        return "asia"
    if 8 <= h < 16:
        return "europe"
    return "us"
# helpers.py (Part 3 of 5)
# -------------------------------------------------------------
# VOLUME FILTERS • SCORE SYSTEM • MODES (QUICK/MID/TREND)
# COOLDOWN SYSTEM • COIN LIST (50 COINS)
# -------------------------------------------------------------

import time
from typing import Dict, Any

# -------------------------------------------------------------
# 1) VOLUME SPIKE FILTER
# -------------------------------------------------------------
def volume_spike(df) -> bool:
    if len(df) < 30:
        return False
    recent = df['volume'].iloc[-1]
    avg = df['volume'].tail(20).mean()
    return recent > avg * 2   # 200% spike

# -------------------------------------------------------------
# 2) SCORE SYSTEM (90+ REQUIRED FOR BALANCED MODE)
# -------------------------------------------------------------
def calc_score(symbol: str, df, htf: Dict[str, str], pa: Dict[str, bool], spread_ok: bool, funding_ok: bool) -> int:
    score = 0

    # EMA trend
    close = df['close']
    if len(close) > 50:
        e20 = ema(close, 20).iloc[-1]
        e50 = ema(close, 50).iloc[-1]
        if e20 > e50:
            score += 15

    # volume spike
    if volume_spike(df):
        score += 15

    # HTF match
    if htf['15m'] in ["bull_engulf", "hammer"]:
        score += 10
    if htf['1h'] in ["bull_engulf", "hammer"]:
        score += 10

    # PA (price action)
    if pa.get('sweep'):
        score += 10
    if pa.get('pullback'):
        score += 10
    if pa.get('range_retest'):
        score += 10
    if pa.get('ob') != "none":
        score += 10
    if pa.get('fvg') != "none":
        score += 10

    # Spread + funding
    if spread_ok:
        score += 5
    if funding_ok:
        score += 5

    return min(score, 100)

# -------------------------------------------------------------
# 3) MODE LOGIC (QUICK / MID / TREND)
# -------------------------------------------------------------
def mode_requirements(df, mode: str) -> bool:
    close = df['close']
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]

    # Quick mode
    if mode == "quick":
        return e20 > e50

    # Mid mode
    if mode == "mid":
        return e20 > e50 and volume_spike(df)

    # Trend mode
    if mode == "trend":
        return e20 > e50 and close.iloc[-1] > e20

    return False

# -------------------------------------------------------------
# 4) COOLDOWN SYSTEM (30 MIN PER COIN)
# -------------------------------------------------------------
COOLDOWN: Dict[str, int] = {}

def cooldown_ok(symbol: str) -> bool:
    now = int(time.time())
    if symbol not in COOLDOWN:
        return True
    return (now - COOLDOWN[symbol]) > 1800   # 30 min

def set_cooldown(symbol: str):
    COOLDOWN[symbol] = int(time.time())

# -------------------------------------------------------------
# 5) COIN LIST (50 COINS)
# -------------------------------------------------------------
COIN_LIST = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","MATICUSDT",
    "DOTUSDT","LTCUSDT","BCHUSDT","AVAXUSDT","UNIUSDT","LINKUSDT","ATOMUSDT","ETCUSDT",
    "FILUSDT","ICPUSDT","NEARUSDT","APTUSDT","SANDUSDT","AXSUSDT","THETAUSDT","FTMUSDT",
    "RUNEUSDT","ALGOUSDT","EGLDUSDT","IMXUSDT","INJUSDT","OPUSDT","ARBUSDT","SUIUSDT",
    "TIAUSDT","PEPEUSDT","TRBUSDT","SEIUSDT","JTOUSDT","PYTHUSDT","RAYUSDT","GMTUSDT",
    "MINAUSDT","WLDUSDT","ZKUSDT","STRKUSDT","DYDXUSDT","VETUSDT","GALAUSDT","KAVAUSDT",
    "FLOWUSDT","SKLUSDT"
]
# helpers.py (Part 4 of 5)
# -------------------------------------------------------------
# SIGNAL BUILDER • TP/SL ENGINE • PREMIUM TELEGRAM FORMATTER
# RED-DANGER EXIT ALERT BUILDER
# -------------------------------------------------------------

from typing import Dict, Any
import time

# -------------------------------------------------------------
# 1) TP/SL AUTO CALCULATOR (MODE-WISE)
# -------------------------------------------------------------
def calc_tp_sl(entry: float, mode: str) -> Dict[str, float]:
    if mode == "quick":
        tp = entry * 1.004    # 0.4%
        sl = entry * 0.996    # -0.4%
    elif mode == "mid":
        tp = entry * 1.020    # 2%
        sl = entry * 0.990    # -1%
    elif mode == "trend":
        tp = entry * 1.030    # 3%
        sl = entry * 0.985    # -1.5%
    else:
        tp = entry * 1.010
        sl = entry * 0.990

    return {"tp": round(tp, 8), "sl": round(sl, 8)}

# -------------------------------------------------------------
# 2) SIGNAL BUILDER (FINAL DECISION + FORMATTED OUTPUT)
# -------------------------------------------------------------
async def build_signal(symbol: str, mode: str) -> Dict[str, Any]:
    if not cooldown_ok(symbol):
        return {"ok": False, "reason": "cooldown"}

    df = await fetch_ohlcv(symbol, "1m", 200)
    if df.empty:
        return {"ok": False, "reason": "data_error"}

    # HTF
    htf = await htf_signal(symbol)

    # PA (Price Action)
    pa = {
        "sweep": liquidity_sweep(df),
        "pullback": ema21_pullback(df),
        "range_retest": range_break_retest(df),
        "ob": detect_order_block(df),
        "fvg": detect_fvg(df)
    }

    # Spread + Funding
    sp_ok = await spread_ok(symbol)
    f_ok = await funding_ok(symbol)

    # Score
    score = calc_score(symbol, df, htf, pa, sp_ok, f_ok)
    if score < 90:
        return {"ok": False, "reason": "low_score", "score": score}

    # Mode Requirement
    if not mode_requirements(df, mode):
        return {"ok": False, "reason": "mode_not_fit"}

    # Entry
    entry = float(df['close'].iloc[-1])
    tpsl = calc_tp_sl(entry, mode)

    # Final Output
    set_cooldown(symbol)

    return {
        "ok": True,
        "symbol": symbol,
        "mode": mode,
        "entry": entry,
        "tp": tpsl['tp'],
        "sl": tpsl['sl'],
        "score": score,
        "htf": htf,
        "pa": pa,
        "time": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    }

# -------------------------------------------------------------
# 3) PREMIUM TELEGRAM SIGNAL FORMATTER
# -------------------------------------------------------------
def format_signal(sig: Dict[str, Any]) -> str:

    return f"""
🔥 BUY SIGNAL — {sig['mode'].upper()} MODE
━━━━━━━━━━━━━━━━
Pair: {sig['symbol']}
━━━━━━━━━━━━━━━━
🎯 Entry: {sig['entry']}
🏆 TP: {sig['tp']}
🛑 SL: {sig['sl']}
📈 Score: {sig['score']}
━━━━━━━━━━━━━━━━
🔍 HTF:
15m → {sig['htf']['15m']}
1h → {sig['htf']['1h']}
━━━━━━━━━━━━━━━━
📊 Price Action:
Sweep: {sig['pa']['sweep']}
Pullback: {sig['pa']['pullback']}
Range Retest: {sig['pa']['range_retest']}
OB: {sig['pa']['ob']}
FVG: {sig['pa']['fvg']}
━━━━━━━━━━━━━━━━
⏱ Time: {sig['time']}
"""

# -------------------------------------------------------------
# 4) PREMIUM RED-DANGER EXIT WARNING FORMATTER
# -------------------------------------------------------------
def format_exit_alert(symbol: str, reason_main: str, extra: str = "") -> str:
    return f"""
🛑 EMERGENCY EXIT — {symbol}
━━━━━━━━━━━━━━━━━━━━
⚠️ DANGER DETECTED
Reason: {reason_main}
{extra}
━━━━━━━━━━━━━━━━━━━━
🚨 ACTION: EXIT NOW — PROTECT CAPITAL
━━━━━━━━━━━━━━━━━━━━
⏱ Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
"""
# helpers.py (Part 5 of 5)
# -------------------------------------------------------------
# MAIN SCAN LOOP • MULTI-MODE RUNNER • OVERRIDE DANGER WATCHER
# -------------------------------------------------------------

import asyncio
from typing import Dict, Any, List

# -------------------------------------------------------------
# 1) SINGLE COIN SCAN
# -------------------------------------------------------------
async def scan_coin(symbol: str, mode: str) -> Dict[str, Any]:
    try:
        sig = await build_signal(symbol, mode)
        return sig
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -------------------------------------------------------------
# 2) RUNNER: CHECK ALL 50 COINS FOR A GIVEN MODE
# -------------------------------------------------------------
async def run_mode(mode: str) -> List[Dict[str, Any]]:
    tasks = [scan_coin(sym, mode) for sym in COIN_LIST]
    results = await asyncio.gather(*tasks)

    # only valid signals
    valid = [r for r in results if r.get("ok")]
    return valid

# -------------------------------------------------------------
# 3) MASTER RUNNER: QUICK + MID + TREND
# -------------------------------------------------------------
async def run_all_modes() -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    out["quick"] = await run_mode("quick")
    out["mid"] = await run_mode("mid")
    out["trend"] = await run_mode("trend")
    return out

# -------------------------------------------------------------
# 4) OVERRIDE GUARDIAN — LIVE DANGER WATCH
# -------------------------------------------------------------
async def danger_watch(symbol: str, entry: float, sl: float) -> str:
    """
    Track coin after entry.
    Trigger danger alert BEFORE liquidation or SL hit.
    """
    for _ in range(60):  # watch 60 cycles (~60 seconds)
        df = await fetch_ohlcv(symbol, "1m", 5)
        if df.empty:
            await asyncio.sleep(1)
            continue

        price = df['close'].iloc[-1]

        # 1) SL danger
        if price <= sl * 1.002:  # within 0.2% of SL
            return format_exit_alert(symbol, "Price approaching SL / liquidation zone")

        # 2) sudden dump
        c = df['close']; h = df['high']; l = df['low']
        wick = (h.iloc[-1] - l.iloc[-1]) / c.iloc[-1] * 100
        if wick > 1.5:
            return format_exit_alert(symbol, "Heavy wick detected - whale movement")

        # 3) funding flip
        f = await fetch_funding(symbol)
        if abs(f) > 0.03:
            return format_exit_alert(symbol, "Funding rate spike - trend danger")

        # 4) BTC check
        btc = await btc_stable()
        if not btc["ok"]:
            return format_exit_alert(symbol, "BTC unstable - macro danger")

        await asyncio.sleep(1)

    return ""  # no danger

# -------------------------------------------------------------
# 5) MULTI-COIN OVERRIDE WATCHER
# -------------------------------------------------------------
async def multi_override_watch(active_signals: List[Dict[str, Any]]) -> List[str]:
    alerts = []
    tasks = []

    for sig in active_signals:
        symbol = sig['symbol']
        entry = sig['entry']
        sl = sig['sl']
        tasks.append(danger_watch(symbol, entry, sl))

    results = await asyncio.gather(*tasks)

    for r in results:
        if r:
            alerts.append(r)

    return alerts

# -------------------------------------------------------------
# 6) EXPORT FINAL
# -------------------------------------------------------------
__all__ = [
    "scan_coin", "run_mode", "run_all_modes",
    "danger_watch", "multi_override_watch",
    "format_signal", "format_exit_alert"
]