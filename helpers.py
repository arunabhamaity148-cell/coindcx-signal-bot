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
# -------------------------
# UPGRADE PACK (helpers.py)
# ASCII SAFE
# -------------------------

import math
import asyncio
import time
from typing import Dict, Any, List

# safe-get helpers (use existing implementations if present)
def _safe_get(name: str, default=None):
    return globals().get(name, default)

logger = _safe_get("logger", None)
if logger is None:
    import logging as _logging
    logger = _logging.getLogger("helpers-upgrade")
    logger.setLevel(_logging.INFO)

# Attempt to use existing fetch_ohlcv / fetch_orderbook / fetch_funding if available
fetch_ohlcv = _safe_get("fetch_ohlcv")
fetch_orderbook = _safe_get("fetch_orderbook")
fetch_funding = _safe_get("fetch_funding")
format_signal = _safe_get("format_signal")  # optional formatter for telegram

# -------------------------
# 1) HTF ALIGNMENT CHECK
# returns True if 15m and 1h EMAs agree for direction
# expecting ohlcv frames (list of dicts or pandas DataFrame) — be defensive
# -------------------------
def _to_pd(df_like):
    try:
        import pandas as pd
        if isinstance(df_like, pd.DataFrame):
            return df_like
        # if list of dicts -> convert
        return pd.DataFrame(df_like)
    except Exception as e:
        logger.debug("pandas missing or conversion failed: %s", str(e))
        return None

def htf_alignment(ohlcv_15m, ohlcv_1h, ema_short=20, ema_long=50) -> bool:
    """
    Return True if direction aligned (both short>long or both short<long)
    Defensive: returns True if cannot compute (prefer allowing signals rather than blocking).
    """
    try:
        df15 = _to_pd(ohlcv_15m)
        df60 = _to_pd(ohlcv_1h)
        if df15 is None or df60 is None or len(df15) < ema_long or len(df60) < ema_long:
            logger.debug("htf_alignment: insufficient data, allowing by default")
            return True

        def ema(series, span):
            return series.ewm(span=span, adjust=False).mean()

        s15 = ema(df15["close"], ema_short).iloc[-1]
        l15 = ema(df15["close"], ema_long).iloc[-1]
        s60 = ema(df60["close"], ema_short).iloc[-1]
        l60 = ema(df60["close"], ema_long).iloc[-1]

        dir15 = 1 if s15 > l15 else -1
        dir60 = 1 if s60 > l60 else -1
        aligned = dir15 == dir60
        logger.debug("htf_alignment: dir15=%s dir60=%s aligned=%s", dir15, dir60, aligned)
        return aligned
    except Exception as e:
        logger.warning("htf_alignment failed: %s", str(e))
        return True

# -------------------------
# 2) LIQUIDITY SWEEP DETECTOR
# looks for large wick beyond body and volume context
# returns dict with boolean + reason
# -------------------------
def detect_liquidity_sweep(ohlcv, wick_ratio_threshold=2.5, body_threshold=0.002) -> Dict[str, Any]:
    """
    ohlcv: pandas DataFrame or list of dicts with keys: open, high, low, close, volume
    returns { 'sweep': bool, 'reason': str }
    """
    try:
        df = _to_pd(ohlcv)
        if df is None or len(df) < 3:
            return {"sweep": False, "reason": "insufficient_data"}

        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last["close"] - last["open"])
        upper_wick = last["high"] - max(last["open"], last["close"])
        lower_wick = min(last["open"], last["close"]) - last["low"]
        avg_vol = df["volume"].rolling(10, min_periods=1).mean().iloc[-2]
        vol = last["volume"]

        # big wick relative to body and previous volatility
        if body <= 0:
            return {"sweep": False, "reason": "zero_body"}

        wick_ratio = max(upper_wick, lower_wick) / (body + 1e-12)
        vol_spike = vol > max(1.5 * avg_vol, 1e-6)

        if wick_ratio >= wick_ratio_threshold and vol_spike:
            direction = "upper" if upper_wick > lower_wick else "lower"
            reason = f"wick_ratio={wick_ratio:.2f}, vol_spike={vol_spike}, dir={direction}"
            logger.info("liquidity_sweep detected: %s", reason)
            return {"sweep": True, "reason": reason, "direction": direction}

        return {"sweep": False, "reason": "none"}
    except Exception as e:
        logger.warning("detect_liquidity_sweep failed: %s", str(e))
        return {"sweep": False, "reason": "error"}

# -------------------------
# 3) SUDDEN SPIKE DETECTOR (volume + price)
# -------------------------
def detect_sudden_spike(ohlcv, vol_multiplier=3.0, price_move_pct=0.01) -> Dict[str, Any]:
    """
    returns { 'spike': bool, 'reason': str }
    """
    try:
        df = _to_pd(ohlcv)
        if df is None or len(df) < 10:
            return {"spike": False, "reason": "insufficient_data"}

        last = df.iloc[-1]
        prev_avg_vol = df["volume"].rolling(10, min_periods=5).mean().iloc[-2]
        vol_spike = last["volume"] > max(prev_avg_vol * vol_multiplier, 1e-6)
        price_move = abs(last["close"] - df["close"].iloc[-2]) / (df["close"].iloc[-2] + 1e-12)

        if vol_spike and price_move >= price_move_pct:
            reason = f"vol_mult={(last['volume']/max(prev_avg_vol,1e-12)):.2f}, price_move={price_move:.3f}"
            logger.info("sudden_spike detected: %s", reason)
            return {"spike": True, "reason": reason, "price_move": price_move, "vol_mult": last["volume"]/max(prev_avg_vol,1e-12)}
        return {"spike": False, "reason": "none"}
    except Exception as e:
        logger.warning("detect_sudden_spike failed: %s", str(e))
        return {"spike": False, "reason": "error"}

# -------------------------
# 4) SL-BEFORE-TOUCH CHECK
# This checks active signals and orderbook distance to liquidation/SL
# returns list of warning messages (strings)
# -------------------------
async def sl_before_touch_check(active_signals: List[Dict[str, Any]]) -> List[str]:
    """
    active_signals: list of dicts expected to include keys:
       symbol, entry, sl, leverage (optional), position_size (optional)
    This function will fetch orderbook (if available) and compute distance.
    """
    warnings = []
    if not active_signals:
        return warnings

    # try to use fetch_orderbook if available
    for sig in active_signals:
        try:
            symbol = sig.get("symbol") or sig.get("pair") or sig.get("market")
            if not symbol:
                continue
            sl = float(sig.get("sl", 0))
            entry = float(sig.get("entry", 0))
            if sl == 0 or entry == 0:
                continue

            # compute percent distance from entry to SL
            dist_pct = abs(entry - sl) / (entry + 1e-12)
            # safe thresholds (if SL very close)
            if dist_pct <= 0.006:  # 0.6% distance
                # optionally check orderbook top levels to see liquidity
                ob_ok = True
                if fetch_orderbook:
                    try:
                        ob = await fetch_orderbook(symbol, depth=5)
                        # expected ob format: {"bids":[[price,size],...],"asks":[...]}
                        top_bid = float(ob.get("bids", [[0,0]])[0][0]) if ob.get("bids") else 0
                        top_ask = float(ob.get("asks", [[0,0]])[0][0]) if ob.get("asks") else 0
                        # if book spread huge or top levels small, mark warning
                        top_liq = (ob.get("bids",[])[0][1] if ob.get("bids") else 0) + (ob.get("asks",[])[0][1] if ob.get("asks") else 0)
                        if top_liq < 0.5:  # small size on top
                            ob_ok = False
                    except Exception as oe:
                        logger.debug("fetch_orderbook error: %s", str(oe))
                        ob_ok = True
                if not ob_ok:
                    warnings.append(f"⚠️ SL CLOSE & LOW BOOK LIQUIDITY: {symbol} entry={entry} sl={sl} dist={dist_pct:.4f}")
                else:
                    warnings.append(f"⚠️ SL CLOSE: {symbol} entry={entry} sl={sl} dist={dist_pct:.4f}")
        except Exception as e:
            logger.debug("sl_before_touch_check item fail: %s", str(e))
            continue
    return warnings

# -------------------------
# 5) MULTI OVERRIDE WATCH (MAIN)
# Accepts active_signals (list of dicts) and returns list of alert strings
# -------------------------
async def multi_override_watch(active_signals: List[Dict[str, Any]]) -> List[str]:
    """
    Main override watcher that protects capital.
    Steps:
      - run SL-before-touch checks
      - for each signal check HTF alignment (if possible)
      - detect liquidity sweep & sudden spike on symbol
      - build alert messages (strings) and return list
    """
    alerts: List[str] = []
    try:
        if not isinstance(active_signals, list):
            logger.warning("multi_override_watch: expected list, got %s", type(active_signals))
            return alerts

        # 1) SL-before-touch global check
        sl_warns = await sl_before_touch_check(active_signals)
        alerts.extend(sl_warns)

        # 2) per-signal checks
        for sig in active_signals:
            try:
                symbol = sig.get("symbol") or sig.get("pair") or sig.get("market")
                if not symbol:
                    continue

                # fetch ohlcv for 1m, 15m, 1h — use fetch_ohlcv if available
                ohlcv_1m = None
                ohlcv_15m = None
                ohlcv_1h = None
                if fetch_ohlcv:
                    try:
                        ohlcv_1m = await fetch_ohlcv(symbol, "1m", 80)
                        ohlcv_15m = await fetch_ohlcv(symbol, "15m", 80)
                        ohlcv_1h = await fetch_ohlcv(symbol, "1h", 80)
                    except Exception as e:
                        logger.debug("fetch_ohlcv error for %s: %s", symbol, str(e))

                # HTF alignment
                htf_ok = True
                if ohlcv_15m is not None and ohlcv_1h is not None:
                    htf_ok = htf_alignment(ohlcv_15m, ohlcv_1h)
                    if not htf_ok:
                        alerts.append(f"⚠️ HTF MISALIGN: {symbol} — skip or review (15m vs 1h conflict)")

                # liquidity sweep check on 1m or 15m
                sweep = {"sweep": False}
                if ohlcv_1m is not None:
                    sweep = detect_liquidity_sweep(ohlcv_1m)
                if sweep.get("sweep"):
                    alerts.append(f"⚠️ LIQ SWEEP: {symbol} — {sweep.get('reason')}")

                # sudden spike check on 1m
                spike = {"spike": False}
                if ohlcv_1m is not None:
                    spike = detect_sudden_spike(ohlcv_1m)
                if spike.get("spike"):
                    alerts.append(f"⚠️ SUDDEN SPIKE: {symbol} — {spike.get('reason')}")

                # combine into formatted override alert if any critical found
                critical = []
                if not htf_ok:
                    critical.append("HTF_MISALIGN")
                if sweep.get("sweep"):
                    critical.append("LIQ_SWEEP")
                if spike.get("spike"):
                    critical.append("SUDDEN_SPIKE")

                if critical:
                    # build a premium red alert message string
                    parts = [
                        "🚨 ACTION: EXIT NOW — PROTECT CAPITAL",
                        f"Pair: {symbol}",
                        f"Issues: {', '.join(critical)}",
                        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
                    ]
                    # include signal details if available
                    if isinstance(sig, dict):
                        entry = sig.get("entry")
                        tp = sig.get("tp") or sig.get("tps")
                        sl = sig.get("sl")
                        if entry:
                            parts.append(f"Entry: {entry}")
                        if tp:
                            parts.append(f"TP: {tp}")
                        if sl:
                            parts.append(f"SL: {sl}")
                    alert_text = "\n".join(parts)
                    # format via format_signal if provided
                    if callable(format_signal):
                        try:
                            alert_text = format_signal(sig, override_reason=", ".join(critical))
                        except Exception:
                            pass
                    alerts.append(alert_text)

            except Exception as e:
                logger.debug("multi_override per-sig fail: %s", str(e))
                continue

        # return unique alerts (dedup)
        unique = []
        seen = set()
        for a in alerts:
            if a not in seen:
                unique.append(a)
                seen.add(a)
        return unique

    except Exception as e:
        logger.error("multi_override_watch top-level error: %s", str(e))
        return alerts

# -------------------------
# END UPGRADE PACK
# -------------------------