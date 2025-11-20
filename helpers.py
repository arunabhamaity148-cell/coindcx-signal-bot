# helpers.py (Part 1 of 4)
# -------------------------------------------------------------
# BASIC IMPORTS, LOGGING, SESSION, OHLCV, TICKER, INDICATORS
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
    except Exception:
        return {}

async def fetch_spread(symbol: str) -> float:
    t = await fetch_ticker(symbol)
    try:
        bid = float(t['bidPrice'])
        ask = float(t['askPrice'])
        return ((ask - bid) / bid) * 100
    except Exception:
        return 999.0

# -------------------------------------------------------------
# FETCH FUNDING RATE
# -------------------------------------------------------------
async def fetch_funding(symbol: str) -> float:
    url = f"{BINANCE_BASE}/fapi/v1/premiumIndex?symbol={symbol}"
    try:
        data = await HTTPSession.get(url)
        return float(data.get("lastFundingRate", 0)) * 100
    except Exception:
        return 0.0

# -------------------------------------------------------------
# SERVER TIME
# -------------------------------------------------------------
async def fetch_server_time() -> int:
    url = f"{BINANCE_BASE}/api/v3/time"
    try:
        data = await HTTPSession.get(url)
        return data.get("serverTime", 0)
    except Exception:
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
    rs = avg_gain / (avg_loss + 1e-12)
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
# helpers.py (Part 2 of 4)
# -------------------------------------------------------------
# ADVANCED FILTERS: BTC STABILITY + HTF PATTERNS + PRICE ACTION
# -------------------------------------------------------------

from typing import Dict, Any

# -------------------------------------------------------------
# BTC STABILITY CHECK
# -------------------------------------------------------------
async def btc_stable() -> Dict[str, Any]:
    df = await fetch_ohlcv("BTCUSDT", "1m", 50)
    if df.empty:
        return {"ok": False, "reason": "BTC data error"}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # volatility (last 10 candle range vs last close)
    vol = (high.tail(10).max() - low.tail(10).min()) / close.iloc[-1] * 100

    # wick check (last candle)
    last_high = high.iloc[-1]
    last_low = low.iloc[-1]
    last_close = close.iloc[-1]
    last_open = df["open"].iloc[-1]

    upper_wick = last_high - max(last_close, last_open)
    lower_wick = min(last_close, last_open) - last_low
    wick_pct = (upper_wick + lower_wick) / last_close * 100

    stable = vol < 1 and wick_pct < 0.5

    return {
        "ok": stable,
        "vol": vol,
        "wick": wick_pct,
    }

# -------------------------------------------------------------
# HTF CANDLE PATTERN DETECTION (15m + 1h)
# -------------------------------------------------------------
def detect_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"

    o = df["open"].iloc[-1]
    c = df["close"].iloc[-1]
    h = df["high"].iloc[-1]
    l = df["low"].iloc[-1]

    body = abs(c - o)
    range_ = h - l

    # basic engulf / hammer / shooting star
    if c > o and body > range_ * 0.6:
        return "bull_engulf"
    if o > c and body > range_ * 0.6:
        return "bear_engulf"
    if (h - max(o, c)) > body * 2:
        return "shooting_star"
    if (min(o, c) - l) > body * 2:
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
    w = df["high"].iloc[-1] - df["low"].iloc[-1]
    body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    return w > body * 3

def ema21_pullback(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    e21 = ema(df["close"], 21)
    return abs(df["close"].iloc[-1] - e21.iloc[-1]) / df["close"].iloc[-1] < 0.002

def range_break_retest(df: pd.DataFrame) -> bool:
    if len(df) < 20:
        return False
    high_line = df["high"].tail(20).max()
    broke = df["close"].iloc[-1] > high_line
    retest = df["low"].iloc[-1] <= high_line
    return broke and retest

# -------------------------------------------------------------
# ORDER BLOCK + FVG (FAIR VALUE GAP)
# -------------------------------------------------------------
def detect_order_block(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    # bearish candle then strong bull candle
    if df["close"].iloc[-2] < df["open"].iloc[-2] and df["close"].iloc[-1] > df["open"].iloc[-1]:
        return "bull_OB"
    # bullish candle then strong bear candle
    if df["close"].iloc[-2] > df["open"].iloc[-2] and df["close"].iloc[-1] < df["open"].iloc[-1]:
        return "bear_OB"
    return "none"

def detect_fvg(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    h1 = df["high"].iloc[-3]
    l1 = df["low"].iloc[-3]
    h2 = df["high"].iloc[-1]
    l2 = df["low"].iloc[-1]

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
    return sp < 0.06   # 0.06% max spread

async def funding_ok(symbol: str) -> bool:
    f = await fetch_funding(symbol)
    return abs(f) < 0.02  # ±0.02% funding

def session_now(ts: int) -> str:
    """
    ts: server time in milliseconds (Binance serverTime)
    """
    h = time.gmtime(ts / 1000).tm_hour
    if 1 <= h < 8:
        return "asia"
    if 8 <= h < 16:
        return "europe"
    return "us"
# helpers.py (Part 3 of 4)
# -------------------------------------------------------------
# VOLUME FILTERS • SCORE SYSTEM • MODES (QUICK/MID/TREND)
# COOLDOWN SYSTEM • COIN LIST • TP/SL • SIGNAL BUILDER
# TELEGRAM FORMATTERS
# -------------------------------------------------------------

import time
from typing import Dict, Any

# -------------------------------------------------------------
# 1) VOLUME SPIKE FILTER
# -------------------------------------------------------------
def volume_spike(df: pd.DataFrame) -> bool:
    if len(df) < 30:
        return False
    recent = df["volume"].iloc[-1]
    avg = df["volume"].tail(20).mean()
    return recent > avg * 2   # 200% spike

# -------------------------------------------------------------
# 2) SCORE SYSTEM (90+ REQUIRED FOR BALANCED MODE)
# -------------------------------------------------------------
def calc_score(
    symbol: str,
    df: pd.DataFrame,
    htf: Dict[str, str],
    pa: Dict[str, Any],
    spread_ok: bool,
    funding_ok_flag: bool,
) -> int:
    score = 0

    close = df["close"]
    if len(close) > 50:
        e20 = ema(close, 20).iloc[-1]
        e50 = ema(close, 50).iloc[-1]
        if e20 > e50:
            score += 15

    # volume spike
    if volume_spike(df):
        score += 15

    # HTF match
    if htf.get("15m") in ["bull_engulf", "hammer"]:
        score += 10
    if htf.get("1h") in ["bull_engulf", "hammer"]:
        score += 10

    # Price action
    if pa.get("sweep"):
        score += 10
    if pa.get("pullback"):
        score += 10
    if pa.get("range_retest"):
        score += 10
    if pa.get("ob") != "none":
        score += 10
    if pa.get("fvg") != "none":
        score += 10

    # Spread + funding
    if spread_ok:
        score += 5
    if funding_ok_flag:
        score += 5

    return min(score, 100)

# -------------------------------------------------------------
# 3) MODE LOGIC (QUICK / MID / TREND)
# -------------------------------------------------------------
def mode_requirements(df: pd.DataFrame, mode: str) -> bool:
    close = df["close"]
    if len(close) < 50:
        return False

    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]

    # Quick mode: light trend
    if mode == "quick":
        return e20 > e50

    # Mid mode: trend + volume spike
    if mode == "mid":
        return e20 > e50 and volume_spike(df)

    # Trend mode: strong trend, price above EMA20
    if mode == "trend":
        return e20 > e50 and close.iloc[-1] > e20

    return False

# -------------------------------------------------------------
# 4) COOLDOWN SYSTEM (30 MIN PER COIN)
# -------------------------------------------------------------
COOLDOWN: Dict[str, int] = {}

def cooldown_ok(symbol: str) -> bool:
    now = int(time.time())
    last = COOLDOWN.get(symbol)
    if last is None:
        return True
    return (now - last) > 1800  # 30 minutes

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
    "FLOWUSDT","SKLUSDT",
]
# -------------------------------------------------------------
# SCANNER HELPERS • SINGLE/MODE/ALL MODES
# (add this after COIN_LIST)
# -------------------------------------------------------------
import asyncio
from typing import Dict, Any, List

async def scan_coin(symbol: str, mode: str) -> Dict[str, Any]:
    """
    Single coin scan – build_signal use kore.
    """
    try:
        sig = await build_signal(symbol, mode)
        return sig
    except Exception as e:
        return {"ok": False, "error": str(e)}

async def run_mode(mode: str) -> List[Dict[str, Any]]:
    """
    Ekta mode (quick/mid/trend) er jonno 50 ta coin scan.
    """
    tasks = [scan_coin(sym, mode) for sym in COIN_LIST]
    results = await asyncio.gather(*tasks)
    # sudhu ok==True signal gula rakhbo
    return [r for r in results if isinstance(r, dict) and r.get("ok")]

async def run_all_modes() -> Dict[str, List[Dict[str, Any]]]:
    """
    Main runner: quick + mid + trend
    """
    return {
        "quick": await run_mode("quick"),
        "mid":   await run_mode("mid"),
        "trend": await run_mode("trend"),
    }

# -------------------------------------------------------------
# 6) TP/SL AUTO CALCULATOR (MODE-WISE)
# -------------------------------------------------------------
def calc_tp_sl(entry: float, mode: str) -> Dict[str, float]:
    if mode == "quick":
        tp = entry * 1.004   # +0.4%
        sl = entry * 0.996   # -0.4%
    elif mode == "mid":
        tp = entry * 1.020   # +2%
        sl = entry * 0.990   # -1%
    elif mode == "trend":
        tp = entry * 1.030   # +3%
        sl = entry * 0.985   # -1.5%
    else:
        tp = entry * 1.010
        sl = entry * 0.990

    return {"tp": round(tp, 8), "sl": round(sl, 8)}

# -------------------------------------------------------------
# 7) SIGNAL BUILDER (FINAL DECISION + STRUCTURED OUTPUT)
# -------------------------------------------------------------
async def build_signal(symbol: str, mode: str) -> Dict[str, Any]:
    # cooldown check
    if not cooldown_ok(symbol):
        return {"ok": False, "reason": "cooldown"}

    # main data
    df = await fetch_ohlcv(symbol, "1m", 200)
    if df.empty:
        return {"ok": False, "reason": "data_error"}

    # HTF
    htf = await htf_signal(symbol)

    # Price action set
    pa = {
        "sweep": liquidity_sweep(df),
        "pullback": ema21_pullback(df),
        "range_retest": range_break_retest(df),
        "ob": detect_order_block(df),
        "fvg": detect_fvg(df),
    }

    # Spread + funding
    sp_ok = await spread_ok(symbol)
    f_ok = await funding_ok(symbol)

    # Score
    score = calc_score(symbol, df, htf, pa, sp_ok, f_ok)
    if score < 90:
        return {"ok": False, "reason": "low_score", "score": score}

    # Mode requirement
    if not mode_requirements(df, mode):
        return {"ok": False, "reason": "mode_not_fit"}

    # Final entry + TP/SL
    entry = float(df["close"].iloc[-1])
    tpsl = calc_tp_sl(entry, mode)

    set_cooldown(symbol)

    return {
        "ok": True,
        "symbol": symbol,
        "mode": mode,
        "entry": entry,
        "tp": tpsl["tp"],
        "sl": tpsl["sl"],
        "score": score,
        "htf": htf,
        "pa": pa,
        "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

# -------------------------------------------------------------
# 8) TELEGRAM SIGNAL FORMATTER (BUY)
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
""".strip()

# -------------------------------------------------------------
# 9) PREMIUM RED-DANGER EXIT WARNING FORMATTER
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
""".strip()
# helpers.py (Part 4 of 4)
# -------------------------------------------------------------
# -------------------------------------------------------------
# OVERRIDE / PROTECT LAYER V2
# - Active trades watch
# - 3-level alerts (YELLOW / ORANGE / RED)
# - BTC stability check
# - Per-mode SL sensitivity (quick / mid / trend)
# - De-duplicate alerts
# -------------------------------------------------------------

import time
from typing import Dict, Any, List

# ধরেছি helpers.py তে আগেই আছে:
#   fetch_ohlcv(symbol, interval, limit)
#   fetch_funding(symbol)
#   btc_stable()
#   ema()
# না থাকলে আগে এগুলো রাখবে।

def _to_pd(df_like):
    """Safely convert to pandas DataFrame."""
    try:
        import pandas as pd
        if isinstance(df_like, pd.DataFrame):
            return df_like
        return pd.DataFrame(df_like)
    except Exception:
        return None

def htf_alignment(ohlcv_15m, ohlcv_1h, ema_short: int = 20, ema_long: int = 50) -> bool:
    """
    15m আর 1h EMA trend একদিকে কিনা চেক করে।
    mismatch হলে False, নাহলে True।
    """
    try:
        import pandas as pd
        df15 = _to_pd(ohlcv_15m)
        df60 = _to_pd(ohlcv_1h)
        if df15 is None or df60 is None:
            return True
        if len(df15) < ema_long or len(df60) < ema_long:
            return True  # data কম হলে block না করে allow

        def _ema(series: "pd.Series", span: int):
            return series.ewm(span=span, adjust=False).mean()

        s15 = _ema(df15["close"], ema_short).iloc[-1]
        l15 = _ema(df15["close"], ema_long).iloc[-1]
        s60 = _ema(df60["close"], ema_short).iloc[-1]
        l60 = _ema(df60["close"], ema_long).iloc[-1]

        dir15 = 1 if s15 > l15 else -1
        dir60 = 1 if s60 > l60 else -1
        return dir15 == dir60
    except Exception:
        return True

def detect_liquidity_sweep(ohlcv, wick_ratio_threshold: float = 2.5) -> Dict[str, Any]:
    """
    Simple liquidity sweep detector:
    - body ছোট, wick বড় + volume বেশি => sweep True
    """
    try:
        df = _to_pd(ohlcv)
        if df is None or len(df) < 3:
            return {"sweep": False}

        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        upper_wick = last["high"] - max(last["open"], last["close"])
        lower_wick = min(last["open"], last["close"]) - last["low"]

        if body <= 0:
            return {"sweep": False}

        wick_ratio = max(upper_wick, lower_wick) / (body + 1e-12)

        avg_vol = df["volume"].rolling(10, min_periods=3).mean().iloc[-2]
        vol = last["volume"]
        vol_spike = vol > max(1.5 * avg_vol, 1e-6)

        if wick_ratio >= wick_ratio_threshold and vol_spike:
            direction = "upper" if upper_wick > lower_wick else "lower"
            return {
                "sweep": True,
                "direction": direction,
                "wick_ratio": float(wick_ratio),
                "vol_spike": bool(vol_spike),
            }

        return {"sweep": False}
    except Exception:
        return {"sweep": False}

def detect_sudden_spike(ohlcv, vol_multiplier: float = 3.0, price_move_pct: float = 0.01) -> Dict[str, Any]:
    """
    Volume + price sudden spike detector.
    """
    try:
        df = _to_pd(ohlcv)
        if df is None or len(df) < 10:
            return {"spike": False}

        last = df.iloc[-1]
        prev_close = df["close"].iloc[-2]
        prev_avg_vol = df["volume"].rolling(10, min_periods=5).mean().iloc[-2]

        price_move = abs(last["close"] - prev_close) / (prev_close + 1e-12)
        vol_spike = last["volume"] > max(prev_avg_vol * vol_multiplier, 1e-6)

        if vol_spike and price_move >= price_move_pct:
            return {
                "spike": True,
                "price_move": float(price_move),
                "vol_mult": float(last["volume"] / max(prev_avg_vol, 1e-12)),
            }

        return {"spike": False}
    except Exception:
        return {"spike": False}

# -------------------------------------------------------------
# ACTIVE TRADES STATE
# -------------------------------------------------------------

ACTIVE_TRADES: List[Dict[str, Any]] = []
LAST_ISSUES: Dict[str, Dict[str, Any]] = {}  # symbol -> {"level": int, "issues": set}
MAX_TRADE_AGE = 60 * 60  # 60 মিনিট পর পুরোনো trade drop

def _trade_key(sig: Dict[str, Any]) -> str:
    """symbol+mode+entry দিয়ে simple key বানাই (duplicate avoid)."""
    symbol = sig.get("symbol") or sig.get("pair") or sig.get("market") or ""
    mode = sig.get("mode", "mid")
    entry = sig.get("entry")
    try:
        entry = float(entry) if entry is not None else 0.0
    except Exception:
        entry = 0.0
    return f"{symbol}:{mode}:{entry:.6f}"

def _update_active_trades(new_signals: List[Dict[str, Any]]):
    """run_all_modes থেকে আসা signal দিয়ে ACTIVE_TRADES আপডেট।"""
    now = time.time()
    existing_keys = {t["key"] for t in ACTIVE_TRADES}
    for sig in new_signals:
        if not isinstance(sig, dict):
            continue
        if not sig.get("ok", True):
            continue

        key = _trade_key(sig)
        if key in existing_keys:
            continue

        symbol = sig.get("symbol") or sig.get("pair") or sig.get("market")
        if not symbol:
            continue

        try:
            entry = float(sig.get("entry"))
        except Exception:
            continue

        tp = sig.get("tp") or sig.get("tps") or 0
        sl = sig.get("sl") or 0
        try:
            tp = float(tp)
        except Exception:
            tp = 0.0
        try:
            sl = float(sl)
        except Exception:
            sl = 0.0

        ACTIVE_TRADES.append(
            {
                "key": key,
                "symbol": symbol,
                "mode": sig.get("mode", "mid"),
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "start_ts": now,
            }
        )

    # পুরোনো trade পরিষ্কার
    fresh: List[Dict[str, Any]] = []
    for t in ACTIVE_TRADES:
        if now - t.get("start_ts", now) <= MAX_TRADE_AGE:
            fresh.append(t)
    ACTIVE_TRADES.clear()
    ACTIVE_TRADES.extend(fresh)

# -------------------------------------------------------------
# THRESHOLD HELPERS (mode + age অনুযায়ী SL distance)
# -------------------------------------------------------------

def _sl_thresholds(mode: str, age_sec: float):
    """
    mode ও trade এর বয়স দেখে দুইটা threshold ফেরত দেয়:
    close_th  = Level-1 (YELLOW) এর জন্য approx distance
    very_th   = Level-3 এর জন্য খুব কাছের SL
    """
    mode = (mode or "mid").lower()
    if mode == "quick":
        base_close = 0.008   # 0.8%
        base_very = 0.004    # 0.4%
    elif mode == "trend":
        base_close = 0.015   # 1.5%
        base_very = 0.010    # 1.0%
    else:  # mid
        base_close = 0.010   # 1.0%
        base_very = 0.006    # 0.6%

    # প্রথম ৫ মিনিটে extra কড়া guard
    if age_sec < 300:
        factor = 0.8
    elif age_sec < 900:
        factor = 1.0
    else:
        factor = 1.2

    close_th = base_close * factor
    very_th = base_very * factor
    return close_th, very_th

# -------------------------------------------------------------
# OVERRIDE MESSAGE BUILDER
# -------------------------------------------------------------

def _build_override_message(
    symbol: str,
    mode: str,
    level: int,
    issues: List[str],
    price: float,
    dist_to_sl: float,
    trade: Dict[str, Any],
    btc_ok: bool,
) -> str:
    mode = (mode or "mid").upper()
    issues_txt = ", ".join(issues)
    base = [
        f"Pair: {symbol}",
        f"Mode: {mode}",
        f"Issues: {issues_txt}",
        f"Price: {price}",
        f"Entry: {trade.get('entry')}",
        f"TP: {trade.get('tp')}",
        f"SL: {trade.get('sl')}",
        f"SL distance: {dist_to_sl*100:.2f}%",
        f"BTC_ok: {btc_ok}",
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
    ]
    detail = "\n".join(base)

    if level == 1:
        title = "🟡 CAUTION — WATCH CLOSELY"
        advice = "Note: Risk increased. SL tighten / partial exit ভাবতে পারো।"
    elif level == 2:
        title = "🟠 HIGH RISK — REVIEW POSITION"
        advice = "Note: Strong warning. নতুন entry নিও না, চলমান trade দ্রুত review করো।"
    else:  # 3
        title = "🔴 ACTION: EXIT NOW — PROTECT CAPITAL"
        advice = "Note: Aggressive scalper rule — সাথে সাথে exit নিলে বড় ক্ষতি থেকে বাঁচা যাবে।"

    return f"{title}\n{detail}\n{advice}"

# -------------------------------------------------------------
# MAIN OVERRIDE WATCHER
# -------------------------------------------------------------

async def multi_override_watch(active_signals: List[Dict[str, Any]]) -> List[str]:
    """
    Main override watcher:
    - নতুন signal থেকে ACTIVE_TRADES আপডেট করে
    - প্রতি active trade এর জন্য market double-check
    - condition গুলো থেকে ৩-লেভেলের alert return করে (string list)
    """
    alerts: List[str] = []

    # 1) active trade আপডেট
    if isinstance(active_signals, list):
        _update_active_trades(active_signals)

    if not ACTIVE_TRADES:
        return alerts

    # 2) একবারই BTC stable check
    try:
        btc_info = await btc_stable()
        btc_ok = bool(btc_info.get("ok", True))
    except Exception:
        btc_ok = True

    now = time.time()

    # 3) প্রতি trade এর জন্য check
    for trade in list(ACTIVE_TRADES):
        try:
            symbol = trade["symbol"]
            mode = trade.get("mode", "mid")
            age_sec = now - trade.get("start_ts", now)
            entry = float(trade.get("entry", 0.0))
            sl = float(trade.get("sl", 0.0))

            # fresh price data
            ohlcv_1m = await fetch_ohlcv(symbol, "1m", 80)
            if ohlcv_1m is None or ohlcv_1m.empty:
                continue

            df1 = _to_pd(ohlcv_1m)
            if df1 is None or len(df1) < 5:
                continue

            current_price = float(df1["close"].iloc[-1])

            if sl > 0:
                dist_to_sl = abs(current_price - sl) / (current_price + 1e-12)
            else:
                dist_to_sl = 1.0  # দূরে ধরে নিলাম

            close_th, very_th = _sl_thresholds(mode, age_sec)
            sl_close = dist_to_sl <= close_th
            sl_very_close = dist_to_sl <= very_th

            # HTF data
            ohlcv_15m = await fetch_ohlcv(symbol, "15m", 80)
            ohlcv_1h = await fetch_ohlcv(symbol, "1h", 80)
            htf_ok = True
            if (
                ohlcv_15m is not None
                and not ohlcv_15m.empty
                and ohlcv_1h is not None
                and not ohlcv_1h.empty
            ):
                htf_ok = htf_alignment(ohlcv_15m, ohlcv_1h)

            # sweep + spike
            sweep_info = detect_liquidity_sweep(ohlcv_1m)
            spike_info = detect_sudden_spike(ohlcv_1m)

            sweep = bool(sweep_info.get("sweep"))
            big_spike = bool(spike_info.get("spike")) and float(
                spike_info.get("price_move", 0.0)
            ) >= 0.02  # >= 2% move
            small_spike = bool(spike_info.get("spike")) and not big_spike

            # funding spike
            funding_bad = False
            try:
                f = await fetch_funding(symbol)
                funding_bad = abs(f) > 0.03
            except Exception:
                funding_bad = False

            # --- issue list বানাও ---
            issues: List[str] = []
            level = 0

            if not htf_ok:
                issues.append("HTF_MISALIGN")
            if sweep:
                issues.append("LIQ_SWEEP")
            if big_spike:
                issues.append("BIG_SPIKE")
            elif small_spike:
                issues.append("SMALL_SPIKE")
            if sl_very_close:
                issues.append("SL_VERY_CLOSE")
            elif sl_close:
                issues.append("SL_CLOSE")
            if not btc_ok:
                issues.append("BTC_UNSTABLE")
            if funding_bad:
                issues.append("FUNDING_SPIKE")

            if not issues:
                LAST_ISSUES.pop(symbol, None)
                continue

            # ---- level decide ----
            if (
                sl_very_close
                and (sweep or big_spike or not btc_ok or not htf_ok)
            ) or (
                (not htf_ok and (sweep or big_spike))
            ) or (
                not btc_ok and (big_spike or sl_close)
            ):
                level = 3
            elif (
                sl_close
                or sweep
                or big_spike
                or (not htf_ok)
                or (not btc_ok)
                or funding_bad
            ):
                level = 2
            else:
                level = 1

            if level == 0:
                continue

            issue_set = set(issues)
            prev = LAST_ISSUES.get(symbol)

            # ডুপ্লিকেট কমানোর জন্য:
            if prev:
                prev_level = prev.get("level", 0)
                prev_issues = prev.get("issues", set())
                if prev_level >= level and issue_set.issubset(prev_issues):
                    continue

            LAST_ISSUES[symbol] = {"level": level, "issues": issue_set}

            msg = _build_override_message(
                symbol=symbol,
                mode=mode,
                level=level,
                issues=issues,
                price=current_price,
                dist_to_sl=dist_to_sl,
                trade=trade,
                btc_ok=btc_ok,
            )
            alerts.append(msg)

        except Exception:
            continue

    return alerts

# -------------------------------------------------------------
# EXPORT SYMBOLS
# -------------------------------------------------------------
try:
    __all__  # type: ignore
except NameError:
    __all__ = []

for name in ["scan_coin", "run_mode", "run_all_modes", "multi_override_watch", "format_signal"]:
    if name not in __all__:
        __all__.append(name)