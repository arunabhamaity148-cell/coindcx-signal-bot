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
# UPGRADE PACK • CLEAN MULTI_OVERRIDE_WATCH (SINGLE EXIT MSG)
# -------------------------------------------------------------

import math as _math
from typing import Dict, Any, List

def _safe_get(name: str, default=None):
    return globals().get(name, default)

# existing helpers jodi thake, segulo reuse
fetch_ohlcv = _safe_get("fetch_ohlcv")
fetch_orderbook = _safe_get("fetch_orderbook")   # optional
fetch_funding = _safe_get("fetch_funding")       # optional

# small helper: pandas frame e convert
def _to_pd(df_like):
    try:
        if isinstance(df_like, pd.DataFrame):
            return df_like
        return pd.DataFrame(df_like)
    except Exception:
        return None

# 1) HTF ALIGNMENT (15m vs 1h EMA)
def htf_alignment(ohlcv_15m, ohlcv_1h, ema_short=20, ema_long=50) -> bool:
    try:
        df15 = _to_pd(ohlcv_15m)
        df60 = _to_pd(ohlcv_1h)
        if df15 is None or df60 is None or len(df15) < ema_long or len(df60) < ema_long:
            return True

        def _ema(series, span):
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

# 2) LIQUIDITY SWEEP DETECTOR
def detect_liquidity_sweep(ohlcv, wick_ratio_threshold=2.5, vol_mult_threshold=1.5) -> Dict[str, Any]:
    try:
        df = _to_pd(ohlcv)
        if df is None or len(df) < 3:
            return {"sweep": False, "reason": "insufficient_data"}

        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        if body <= 0:
            return {"sweep": False, "reason": "zero_body"}

        upper_wick = last["high"] - max(last["open"], last["close"])
        lower_wick = min(last["open"], last["close"]) - last["low"]
        wick_ratio = max(upper_wick, lower_wick) / (body + 1e-12)

        avg_vol = df["volume"].rolling(10, min_periods=1).mean().iloc[-2]
        vol = last["volume"]
        vol_mult = vol / max(avg_vol, 1e-12)

        if wick_ratio >= wick_ratio_threshold and vol_mult >= vol_mult_threshold:
            direction = "upper" if upper_wick > lower_wick else "lower"
            reason = f"wick_ratio={wick_ratio:.2f}, vol_mult={vol_mult:.2f}, dir={direction}"
            return {"sweep": True, "reason": reason, "direction": direction}
        return {"sweep": False, "reason": "none"}
    except Exception:
        return {"sweep": False, "reason": "error"}

# 3) SUDDEN SPIKE DETECTOR
def detect_sudden_spike(ohlcv, vol_multiplier=3.0, price_move_pct=0.01) -> Dict[str, Any]:
    try:
        df = _to_pd(ohlcv)
        if df is None or len(df) < 10:
            return {"spike": False, "reason": "insufficient_data"}

        last = df.iloc[-1]
        prev_close = df["close"].iloc[-2]
        prev_avg_vol = df["volume"].rolling(10, min_periods=5).mean().iloc[-2]

        vol_spike = last["volume"] > max(prev_avg_vol * vol_multiplier, 1e-6)
        price_move = abs(last["close"] - prev_close) / (prev_close + 1e-12)

        if vol_spike and price_move >= price_move_pct:
            reason = f"vol_mult={(last['volume']/max(prev_avg_vol,1e-12)):.2f}, price_move={price_move:.3f}"
            return {"spike": True, "reason": reason}
        return {"spike": False, "reason": "none"}
    except Exception:
        return {"spike": False, "reason": "error"}

# 4) SL-BEFORE-TOUCH CHECK (only data, no message)
async def sl_before_touch_check(active_signals: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    if not active_signals:
        return info

    for sig in active_signals:
        try:
            symbol = sig.get("symbol") or sig.get("pair") or sig.get("market")
            if not symbol:
                continue
            sl = float(sig.get("sl", 0.0))
            entry = float(sig.get("entry", 0.0))
            if sl == 0.0 or entry == 0.0:
                continue

            dist_pct = abs(entry - sl) / (entry + 1e-12)
            low_liq = False

            if fetch_orderbook and dist_pct <= 0.006:
                try:
                    ob = await fetch_orderbook(symbol, depth=5)
                    bids = ob.get("bids") or []
                    asks = ob.get("asks") or []
                    top_liq = 0.0
                    if bids:
                        top_liq += float(bids[0][1])
                    if asks:
                        top_liq += float(asks[0][1])
                    if top_liq < 0.5:
                        low_liq = True
                except Exception:
                    low_liq = False

            info[symbol] = {"dist_pct": dist_pct, "low_liq": low_liq}
        except Exception:
            continue

    return info

# 5) MAIN MULTI_OVERRIDE_WATCH — SINGLE EXIT MESSAGE PER COIN
async def multi_override_watch(active_signals: List[Dict[str, Any]]) -> List[str]:
    alerts: List[str] = []
    if not active_signals:
        return alerts

    try:
        sl_info_map = await sl_before_touch_check(active_signals)

        for sig in active_signals:
            try:
                symbol = sig.get("symbol") or sig.get("pair") or sig.get("market")
                if not symbol:
                    continue

                issues: List[str] = []
                detail_lines: List[str] = []

                # SL distance
                sl_info = sl_info_map.get(symbol)
                if sl_info and sl_info["dist_pct"] <= 0.006:
                    issues.append("SL_CLOSE")
                    detail_lines.append(f"SL distance: {sl_info['dist_pct']:.4f}")
                    if sl_info["low_liq"]:
                        issues.append("LOW_BOOK_LIQUIDITY")
                        detail_lines.append("Orderbook top liquidity low")

                # OHLCV load
                ohlcv_1m = ohlcv_15m = ohlcv_1h = None
                if fetch_ohlcv:
                    try:
                        ohlcv_1m = await fetch_ohlcv(symbol, "1m", 80)
                        ohlcv_15m = await fetch_ohlcv(symbol, "15m", 80)
                        ohlcv_1h = await fetch_ohlcv(symbol, "1h", 80)
                    except Exception:
                        pass

                # HTF misalign
                if ohlcv_15m is not None and ohlcv_1h is not None:
                    if not htf_alignment(ohlcv_15m, ohlcv_1h):
                        issues.append("HTF_MISALIGN")
                        detail_lines.append("15m vs 1h EMA conflict")

                # Liquidity sweep
                if ohlcv_1m is not None:
                    sweep = detect_liquidity_sweep(ohlcv_1m)
                    if sweep.get("sweep"):
                        issues.append("LIQ_SWEEP")
                        detail_lines.append(f"Sweep: {sweep.get('reason')}")

                # Sudden spike
                if ohlcv_1m is not None:
                    spike = detect_sudden_spike(ohlcv_1m)
                    if spike.get("spike"):
                        issues.append("SUDDEN_SPIKE")
                        detail_lines.append(f"Spike: {spike.get('reason')}")

                if not issues:
                    continue

                parts = [
                    "🚨 ACTION: EXIT NOW — PROTECT CAPITAL",
                    f"Pair: {symbol}",
                    f"Issues: {', '.join(issues)}",
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
                ]

                entry = sig.get("entry")
                tp = sig.get("tp") or sig.get("tps")
                sl = sig.get("sl")
                if entry is not None:
                    parts.append(f"Entry: {entry}")
                if tp is not None:
                    parts.append(f"TP: {tp}")
                if sl is not None:
                    parts.append(f"SL: {sl}")

                if detail_lines:
                    parts.append("Details:")
                    parts.extend(detail_lines)

                alert_text = "\n".join(parts)

                if alert_text not in alerts:
                    alerts.append(alert_text)

            except Exception:
                continue

        return alerts
    except Exception:
        return alerts

# -------------------------------------------------------------
# EXPORTS
# -------------------------------------------------------------
__all__ = [
    "scan_coin",
    "run_mode",
    "run_all_modes",
    "multi_override_watch",
    "format_signal",
    "format_exit_alert",
]