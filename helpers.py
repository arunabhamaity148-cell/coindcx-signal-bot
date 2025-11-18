# helpers.py  (PART 1)
# Part 1: imports, config, env, exchange, fetching OHLCV, indicators

import os
import time
import csv
import json
import math
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

import ccxt
import requests
import pandas as pd
import numpy as np

# Try to load dotenv if available (optional)
try:
    from dotenv import load_dotenv as _ld
    _ld()
except Exception:
    # dotenv optional; env vars can be set by Railway runtime
    pass

# ----------------- Logging -----------------
logger = logging.getLogger("helpers")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

# ----------------- Config / env -----------------
MIN_SIGNAL_SCORE = int(os.getenv("MIN_SIGNAL_SCORE", 60))
MODE_THRESHOLDS = {
    "quick": int(os.getenv("THRESH_QUICK", 75)),
    "mid": int(os.getenv("THRESH_MID", 60)),
    "trend": int(os.getenv("THRESH_TREND", 50)),
}
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1", "true", "yes")
COOLDOWN_MINUTES = {
    "quick": int(os.getenv("COOLDOWN_QUICK_MIN", 30)),
    "mid": int(os.getenv("COOLDOWN_MID_MIN", 45)),
    "trend": int(os.getenv("COOLDOWN_TREND_MIN", 120)),
}
MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", 100000))  # in quote currency (USDT)
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", 0.5))  # percent

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1m")

# weights for scoring (simple tunable)
WEIGHTS = {
    "ema": 25,
    "macd": 20,
    "rsi": 15,
    "atr_vol": 15,
    "adx_trend": 15,
    "volume": 10,
}

# In-memory cooldown map: { (symbol, mode) : unix_ts_until_available }
_COOLDOWN_MAP: Dict[Tuple[str, str], float] = {}

# ----------------- Utility functions -----------------
def now_ts() -> float:
    return time.time()

def to_unix_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

# ----------------- Load coins from CSV -----------------
def load_coins(path: str = "coins.csv") -> List[str]:
    """
    Reads a simple CSV where first column header is 'symbol' and following rows contain coin symbols
    e.g.
    symbol
    BTC
    ETH
    ...
    Returns list like ['BTC','ETH', ...]
    """
    coins = []
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                val = row[0].strip()
                if not val:
                    continue
                # normalize: uppercase, no slash
                coins.append(val.upper())
    except FileNotFoundError:
        logger.warning(f"coins.csv not found at {path}. Returning empty list.")
    except Exception as e:
        logger.exception("Error reading coins.csv: %s", e)
    return coins

# ----------------- Exchange factory -----------------
def get_exchange(name: str = "binance", api_key: Optional[str] = None, secret: Optional[str] = None):
    """
    Returns a ccxt exchange instance. If api_key/secret provided, sets them.
    """
    ex_cls = getattr(ccxt, name)
    kwargs = {"enableRateLimit": True}
    ex = ex_cls(kwargs)
    if api_key:
        ex.apiKey = api_key
    if secret:
        ex.secret = secret
    # set timeouts / options
    try:
        ex.options["adjustForTimeDifference"] = True
    except Exception:
        pass
    return ex

# ----------------- Fetch OHLCV robust -----------------
def fetch_ohlcv(exchange, symbol: str, timeframe: str = DEFAULT_TIMEFRAME, limit: int = 200) -> Optional[pd.DataFrame]:
    """
    Returns pandas DataFrame with columns: timestamp, open, high, low, close, volume
    If error returns None
    """
    pair = f"{symbol}/USDT" if "/" not in symbol else symbol
    try:
        raw = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except ccxt.BadSymbol:
        logger.error("BadSymbol: exchange does not have market %s", pair)
        return None
    except Exception as e:
        logger.exception("Error fetching OHLCV for %s: %s", pair, e)
        return None

# ----------------- Indicator calculations -----------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    hist = ema_fast - ema_slow
    signal_line = hist.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": hist, "signal": signal_line, "hist": hist - signal_line})

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def _dx_plus_minus(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    # used for ADX: compute +DM, -DM, TR
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_)
    minus_di = 100 * (minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx}).fillna(0)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns: ema_fast, ema_slow, rsi, macd, macd_signal, macd_hist, atr, adx, plus_di, minus_di
    """
    out = df.copy()
    close = out["close"]
    out["ema_fast"] = ema(close, 9)
    out["ema_slow"] = ema(close, 21)
    out["rsi"] = rsi(close, 14)
    mac = macd(close)
    out["macd"] = mac["macd"]
    out["macd_signal"] = mac["signal"]
    out["macd_hist"] = mac["hist"]
    out["atr"] = atr(out, 14)
    adx_df = _dx_plus_minus(out, 14)
    out = pd.concat([out, adx_df], axis=1)
    return out
# helpers.py  (PART 2)
# Part 2: scoring, filters, cooldown, sl/tp, telegram, helpers and main-use utilities

from math import isfinite

# ----------------- Scoring -----------------
def compute_score(symbol: str, df: pd.DataFrame) -> Dict:
    """
    Compute score (0-100) for the latest candle.
    Returns dict containing score and breakdown and useful values used in filters.
    """
    result = {
        "symbol": symbol,
        "score": 0,
        "breakdown": {},
        "entry": None,
        "atr": None,
        "volume_24h": None,
        "spread_pct": None,
        "trend_higher_tf": False,
    }

    if df is None or len(df) < 30:
        return result

    last = df.iloc[-1]
    # values
    ema_up = last["ema_fast"] > last["ema_slow"]
    macd_pos = last["macd"] > last["macd_signal"]
    rsi_val = float(last["rsi"])
    atr_val = float(last["atr"])
    adx_val = float(last.get("adx", 0.0))
    plus_di = float(last.get("plus_di", 0.0))
    minus_di = float(last.get("minus_di", 0.0))
    volume_recent = df["volume"].tail(14).mean()

    # simple spread estimate: (ask-bid)/mid *100 ; we don't have ask/bid in OHLCV -> approximate using high/low
    # caution: this is only approximate; better via orderbook
    spread_pct = ((last["high"] - last["low"]) / last["close"]) * 100

    # trend higher TF: naive check using last few ema slope
    ema_slope = (df["ema_slow"].iloc[-1] - df["ema_slow"].iloc[-5]) if len(df) > 6 else 0.0
    trend_higher_tf = ema_slope > 0

    # compute partial scores
    s_ema = WEIGHTS["ema"] if ema_up else 0
    s_macd = WEIGHTS["macd"] if macd_pos else 0
    s_rsi = 0
    if rsi_val < 30:
        s_rsi = int(WEIGHTS["rsi"] * 0.2)  # oversold small
    elif 30 <= rsi_val <= 60:
        s_rsi = int(WEIGHTS["rsi"] * 0.9)
    elif rsi_val > 60:
        s_rsi = int(WEIGHTS["rsi"] * 0.6)

    # ATR/vol: if ATR relative small vs price -> good for scalping; we reward moderate ATR.
    atr_ratio = atr_val / max(1.0, last["close"])
    s_atr = 0
    if 0 < atr_ratio < 0.01:
        s_atr = WEIGHTS["atr_vol"]
    elif 0.01 <= atr_ratio < 0.03:
        s_atr = int(WEIGHTS["atr_vol"] * 0.7)
    else:
        s_atr = int(WEIGHTS["atr_vol"] * 0.3)

    # ADX trend strength
    s_adx = 0
    if adx_val >= 25:
        s_adx = WEIGHTS["adx_trend"]
    elif 15 <= adx_val < 25:
        s_adx = int(WEIGHTS["adx_trend"] * 0.6)
    else:
        s_adx = int(WEIGHTS["adx_trend"] * 0.2)

    # volume score
    s_vol = 0
    if volume_recent >= MIN_24H_VOLUME / 24:  # per-bar average rough check
        s_vol = WEIGHTS["volume"]

    total = s_ema + s_macd + s_rsi + s_atr + s_adx + s_vol
    # normalize to 0-100 (weights sum can be >100 depending config)
    max_weight_sum = sum(WEIGHTS.values())
    score = int((total / max_weight_sum) * 100)
    score = max(0, min(100, score))

    # entry candidate: use last close as entry by default
    entry = float(last["close"])

    result.update({
        "score": score,
        "breakdown": {
            "ema": s_ema, "macd": s_macd, "rsi": s_rsi,
            "atr": s_atr, "adx": s_adx, "volume": s_vol
        },
        "entry": entry,
        "atr": atr_val,
        "volume_24h": volume_recent * 1440 if volume_recent and not math.isnan(volume_recent) else 0.0,
        "spread_pct": spread_pct,
        "trend_higher_tf": trend_higher_tf,
        "rsi": rsi_val,
        "adx": adx_val,
    })
    return result

# ----------------- Cooldown management -----------------
def is_in_cooldown(symbol: str, mode: str) -> bool:
    key = (symbol.upper(), mode)
    ts = _COOLDOWN_MAP.get(key)
    if ts and ts > now_ts():
        return True
    return False

def set_cooldown(symbol: str, mode: str):
    key = (symbol.upper(), mode)
    minutes = COOLDOWN_MINUTES.get(mode, 30)
    _COOLDOWN_MAP[key] = now_ts() + minutes * 60
    logger.info("Cooldown set %s %s until %s", symbol, mode, datetime.fromtimestamp(_COOLDOWN_MAP[key]).isoformat())

# ----------------- Filters -----------------
def passes_basic_filters(signal: Dict) -> Tuple[bool, str]:
    """
    Returns (True/False, reason)
    Expects keys: score, mode, volume_24h, spread_pct, trend_higher_tf, symbol
    """
    score = signal.get("score", 0)
    mode = signal.get("mode", "mid")
    vol = signal.get("volume_24h", 0.0)
    spread = signal.get("spread_pct", 999.0)
    trend_ok = signal.get("trend_higher_tf", False)
    symbol = signal.get("symbol", "")

    # threshold by mode
    threshold = MODE_THRESHOLDS.get(mode, MIN_SIGNAL_SCORE)
    if score < threshold:
        return False, f"score {score} < threshold {threshold}"

    if vol < MIN_24H_VOLUME:
        return False, f"low_volume {vol:.1f}"

    if spread * 100 > MAX_SPREAD_PCT:  # note spread is fraction we constructed earlier as percent; be safe
        # spread came as fraction*100 earlier; if small misunderstandings, still safe
        return False, f"spread_too_high {spread:.4f}"

    if not trend_ok and mode == "trend":
        return False, "higher_tf_trend_mismatch"

    # cooldown
    if is_in_cooldown(symbol, mode):
        return False, "in_cooldown"

    return True, "ok"

# ----------------- SL/TP calculation -----------------
def compute_sl_tp(entry: float, atr_val: float, mode: str = "mid") -> Tuple[float, float]:
    """
    Returns (sl, tp)
    SL is below entry for BUY. Use ATR multipliers by mode.
    """
    if not isfinite(entry) or not isfinite(atr_val):
        raise ValueError("Invalid entry/atr values")

    if mode == "quick":
        k_sl = 1.5; rr = 1.6
    elif mode == "mid":
        k_sl = 2.0; rr = 2.2
    else:
        k_sl = 2.5; rr = 2.8

    sl = max(0.00000001, entry - k_sl * atr_val)
    tp = entry + (entry - sl) * rr
    return sl, tp

# ----------------- Telegram formatting and send -----------------
def format_signal_message(payload: Dict) -> str:
    """
    Compose telegram message (emoji style).
    Expect payload: symbol, mode, score, entry, tp, sl, qty, size_quote, lev, danger_low, danger_high, reason
    """
    sym = payload.get("symbol")
    mode = payload.get("mode", "MID").upper()
    score = payload.get("score", 0)
    entry = payload.get("entry")
    tp = payload.get("tp")
    sl = payload.get("sl")
    qty = payload.get("qty", 0)
    sizeq = payload.get("size_quote", 0)
    lev = payload.get("lev", 1)
    danger_low = payload.get("danger_low")
    danger_high = payload.get("danger_high")
    atr = payload.get("atr")
    reason = payload.get("reason", "")
    emoji_mode = "🔥" if mode == "QUICK" else "⚡" if mode == "MID" else "💎"
    lines = []
    lines.append(f"{emoji_mode} BUY SIGNAL — {mode}")
    lines.append(f"Pair: {sym}  Score: {score}")
    lines.append(f"Entry: {entry:,}")
    lines.append(f"TP: {tp:,}   SL: {sl:,}")
    lines.append(f"Qty: {qty}   Size(quote): {sizeq}  Lev: {lev}x")
    if danger_low is not None and danger_high is not None:
        lines.append(f"⚠️ Danger: {danger_low:,} — {danger_high:,} (ATR={atr:.6f})")
    if reason:
        lines.append(f"Reason: {reason}")
    lines.append(f"Mode: {mode} | Confidence: {score}%")
    return "\n".join(lines)

def send_telegram_message(text: str) -> bool:
    """
    Sends message to TELEGRAM_CHAT_ID using TELEGRAM_TOKEN. Returns True on 200.
    If TELEGRAM_TOKEN not set or NOTIFY_ONLY True, just logs and returns True (simulate).
    """
    if NOTIFY_ONLY or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("NOTIFY_ONLY or missing token/chat_id — would send telegram:\n%s", text)
        return True
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("Telegram sent")
            return True
        else:
            logger.error("Telegram failed status=%s body=%s", resp.status_code, resp.text)
            return False
    except Exception as e:
        logger.exception("Telegram send exception: %s", e)
        return False

# ----------------- Convenience: build full signal and emit -----------------
def emit_signal(exchange, symbol: str, mode: str, indicator_info: Dict, size_quote: float = 10.0, lev: int = 1):
    """
    indicator_info is result returned from compute_score: includes entry, atr, score, etc.
    Emits telegram if passes filters and sets cooldown.
    """
    info = indicator_info.copy()
    info["mode"] = mode
    info["symbol"] = symbol
    # compute sl/tp
    entry = info.get("entry")
    atr_val = info.get("atr", 0.0) or 0.0
    sl, tp = compute_sl_tp(entry, atr_val, mode)
    info["sl"] = sl
    info["tp"] = tp
    info["size_quote"] = size_quote
    info["qty"] = 0  # leave to main/executor to compute based on account and risk
    info["lev"] = lev
    info["danger_low"] = max(0, entry - atr_val)
    info["danger_high"] = entry + atr_val
    info["reason"] = ",".join([k for k, v in info.get("breakdown", {}).items() if v > 0])

    # Pass filters
    ok, reason = passes_basic_filters(info)
    if not ok:
        logger.info("Signal %s %s blocked by filter: %s", symbol, mode, reason)
        return False

    # Format message and send
    msg = format_signal_message(info)
    sent = send_telegram_message(msg)
    if sent:
        set_cooldown(symbol, mode)
        logger.info("SIGNAL SENT for %s/%s", symbol, mode)
        return True
    else:
        logger.error("Signal NOT sent for %s/%s", symbol, mode)
        return False

# ----------------- Small helper for main loop: safe scan single symbol -----------------
def scan_symbol(exchange, symbol: str, mode: str = "mid", timeframe: str = DEFAULT_TIMEFRAME) -> Dict:
    """
    Fetch OHLCV, compute indicators, compute score dict
    returns dict with keys: symbol, score, entry, atr, trend_higher_tf, volume_24h, spread_pct, df
    """
    df = fetch_ohlcv(exchange, symbol, timeframe=timeframe, limit=200)
    if df is None:
        return {"symbol": symbol, "score": 0}

    df = compute_indicators(df)
    result = compute_score(symbol, df)
    # attach df for debugging if needed (not huge ideally)
    result["df_last"] = df.iloc[-5:].to_dict(orient="list")
    return result

# ----------------- Optionally persist cooldown (stub) -----------------
def persist_cooldown_map(path: str = "cooldown.json"):
    """
    Save _COOLDOWN_MAP to disk (useful for restarts). Optional.
    """
    try:
        simple = {f"{k[0]}::{k[1]}": v for k, v in _COOLDOWN_MAP.items()}
        with open(path, "w") as f:
            json.dump(simple, f)
    except Exception:
        logger.exception("Failed to persist cooldown map")

def load_cooldown_map(path: str = "cooldown.json"):
    try:
        if not os.path.exists(path):
            return
        with open(path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            sym, mode = k.split("::")
            _COOLDOWN_MAP[(sym, mode)] = float(v)
    except Exception:
        logger.exception("Failed to load cooldown map")

# ----------------- End of helpers.py -----------------