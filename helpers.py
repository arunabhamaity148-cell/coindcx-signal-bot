# helpers.py — PRO (100/100) (paste whole file)
from __future__ import annotations
import os, time, json, math, threading, logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# try ccxt import
try:
    import ccxt
except Exception:
    ccxt = None

import requests

# dotenv optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------- logging -----------------
logger = logging.getLogger("helpers")
logger.setLevel(os.getenv("HELPERS_LOG_LEVEL", "INFO"))
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

# ----------------- Config (env-friendly) -----------------
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1","true","yes")
AUTO_EXECUTE = os.getenv("AUTO_EXECUTE", "False").lower() in ("1","true","yes")  # we will keep False by default
SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
OHLCV_MAX_RETRIES = int(os.getenv("OHLCV_MAX_RETRIES", "3"))
OHLCV_RETRY_DELAY = float(os.getenv("OHLCV_RETRY_DELAY", "0.8"))
RATE_LIMIT_PAUSE = float(os.getenv("RATE_LIMIT_PAUSE", "0.6"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "3"))

MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "70"))
MODE_THRESHOLDS = {
    "quick": float(os.getenv("THRESH_QUICK", os.getenv("MIN_SIGNAL_SCORE", "80"))),
    "mid": float(os.getenv("THRESH_MID", "68")),
    "trend": float(os.getenv("THRESH_TREND", "55"))
}

MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", "150000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.5"))  # percent
SAME_COIN_BLOCK = int(os.getenv("SAME_COIN_BLOCK", "1800"))  # seconds

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

COOLDOWN_PERSIST_PATH = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")
BACKTEST_DB = os.getenv("BACKTEST_DB", "backtest_history.json")
TRADE_LOG = os.getenv("TRADE_LOG", "trade_log.json")

# scoring weights (tunable)
WEIGHTS = {
    "ema": float(os.getenv("W_EMA", "25")),
    "macd": float(os.getenv("W_MACD", "20")),
    "rsi": float(os.getenv("W_RSI", "15")),
    "atr_vol": float(os.getenv("W_ATR_VOL", "15")),
    "adx_trend": float(os.getenv("W_ADX", "15")),
    "volume": float(os.getenv("W_VOLUME", "10")),
}

# internal state
_COOLDOWN_LOCK = threading.Lock()
_COOLDOWN_MAP: Dict[Tuple[str,str], float] = {}
_BACKTEST_LOCK = threading.Lock()

# ----------------- utils -----------------
def now_ts() -> float:
    return time.time()

def safe_write_atomic(path: str, data: Any):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        logger.exception("safe_write_atomic failed for %s", path)

def safe_load_json(path: str):
    try:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
    except Exception:
        logger.exception("safe_load_json failed for %s", path)
        return None

# ----------------- exchange helper -----------------
def create_exchange_instance(name: str = "binance", api_key: Optional[str] = None, api_secret: Optional[str] = None):
    if ccxt is None:
        raise RuntimeError("ccxt not installed")
    ex_cls = getattr(ccxt, name, None)
    if ex_cls is None:
        ex_cls = getattr(ccxt, name.lower(), None)
    if ex_cls is None:
        raise RuntimeError(f"Exchange '{name}' not found in ccxt")
    opts = {"enableRateLimit": True}
    ex = ex_cls(opts)
    if api_key:
        ex.apiKey = api_key
    if api_secret:
        ex.secret = api_secret
    try:
        ex.options = ex.options or {}
        ex.options["adjustForTimeDifference"] = True
    except Exception:
        pass
    # best-effort load markets
    try:
        ex.load_markets()
    except Exception:
        logger.debug("load_markets skipped/failed")
    logger.info("Exchange created: %s api_set=%s", name, bool(api_key and api_secret))
    return ex

# ----------------- OHLCV with retry and rate-limit pause -----------------
def fetch_ohlcv_with_retry(exchange, pair: str, timeframe: str = "1m", limit: int = 200):
    pair = pair if "/" in pair else f"{pair}/USDT"
    last_exc = None
    for attempt in range(1, OHLCV_MAX_RETRIES + 1):
        try:
            data = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            # pause small to avoid rate limit
            time.sleep(RATE_LIMIT_PAUSE)
            return data
        except Exception as e:
            last_exc = e
            logger.debug("fetch_ohlcv attempt %d failed for %s: %s", attempt, pair, e)
            time.sleep(OHLCV_RETRY_DELAY * attempt)
    logger.error("fetch_ohlcv failed for %s after %d attempts: %s", pair, OHLCV_MAX_RETRIES, last_exc)
    return None

# ----------------- df builder -----------------
def ohlcv_to_df(ohlcv):
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    return df

# ----------------- indicators -----------------
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df: pd.DataFrame):
    prev = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev).abs()
    tr3 = (df['low'] - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14):
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _dx_plus_minus(df: pd.DataFrame, n: int = 14):
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/n, adjust=False).mean() / (atr_ + 1e-12))
    minus_di = 100 * (minus_dm.ewm(alpha=1/n, adjust=False).mean() / (atr_ + 1e-12))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx_series = dx.ewm(alpha=1/n, adjust=False).mean()
    return plus_di.fillna(0), minus_di.fillna(0), adx_series.fillna(0)

def vwap(df: pd.DataFrame):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pv = (tp * df['volume']).cumsum()
    denom = df['volume'].cumsum().replace(0, np.nan)
    return (pv / denom).ffill().fillna(df['close'])

# ----------------- add indicators -----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['close']
    df['ema9'] = ema(close, 9)
    df['ema20'] = ema(close, 20)
    df['ema50'] = ema(close, 50)
    df['ema200'] = ema(close, 200)
    df['rsi14'] = rsi(close, 14)
    macd_line, macd_signal, hist = macd(close)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist'] = hist
    df['atr14'] = atr(df, 14)
    plus, minus, adx_series = _dx_plus_minus(df, 14)
    df['plus_di'] = plus
    df['minus_di'] = minus
    df['adx14'] = adx_series
    df['vwap'] = vwap(df)
    df['vol_ma20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['vol_rel'] = df['volume'] / df['vol_ma20'].replace(0,1)
    df['candle_body'] = (df['close'] - df['open']).abs()
    return df

# ----------------- scoring engine -----------------
def compute_signal_score(df: pd.DataFrame) -> Dict[str,Any]:
    if df is None or len(df) < 40:
        return {"score": 0, "reasons": []}
    latest = df.iloc[-1]
    score = 0.0
    reasons = []
    # ema
    if latest['ema20'] > latest['ema50']:
        score += WEIGHTS['ema']; reasons.append("EMA_up")
    else:
        score -= WEIGHTS['ema']/2; reasons.append("EMA_down")
    # macd
    if latest['macd_hist'] > 0:
        score += WEIGHTS['macd']; reasons.append("MACD_pos")
    else:
        score -= WEIGHTS['macd']/2
    # rsi
    r = float(latest['rsi14'])
    if 30 <= r <= 60:
        score += WEIGHTS['rsi']
    elif r < 30:
        score += WEIGHTS['rsi']*0.6; reasons.append("RSI_oversold")
    else:
        score -= WEIGHTS['rsi']*0.6; reasons.append("RSI_overbought")
    # atr/vol
    atrv = float(latest['atr14'])
    atr_ratio = atrv / max(1.0, float(latest['close']))
    if atr_ratio < 0.02:
        score += WEIGHTS['atr_vol']*1.0
    elif atr_ratio < 0.04:
        score += WEIGHTS['atr_vol']*0.6
    else:
        score -= WEIGHTS['atr_vol']*0.6; reasons.append("High_ATR")
    # adx
    if latest['adx14'] > 25:
        score += WEIGHTS['adx_trend']; reasons.append("ADX_strong")
    # volume
    vol_rel = float(latest.get('vol_rel',0))
    if vol_rel > 1.5:
        score += WEIGHTS['volume']; reasons.append("Vol_spike")
    # normalize
    max_possible = sum(WEIGHTS.values()) + 10
    score = max(0.0, min(100.0, (score / max_possible) * 100.0))
    return {"score": round(score,1), "reasons": reasons, "entry": float(latest['close']), "atr": atrv, "vol_rel": vol_rel, "adx": float(latest['adx14'])}

# ----------------- adaptive tp/sl -----------------
def adaptive_tp_sl(entry: float, atr_val: float, mode: str, score: float) -> Tuple[float,float]:
    if atr_val <= 0:
        atr_val = max(1e-8, entry*0.001)
    if mode.lower()=="trend":
        tp_mul, sl_mul = 1.6, 1.0
    elif mode.lower()=="mid":
        tp_mul, sl_mul = 1.1, 1.8
    else:
        tp_mul, sl_mul = 0.9, 1.6
    score_factor = max(0.8, min(1.4, 1.0 + (score - 60)/300.0))
    tp = entry + tp_mul * score_factor * atr_val
    sl = entry - sl_mul / score_factor * atr_val
    return round(tp,8), round(sl,8)

# ----------------- cooldown persist -----------------
def load_cooldown_map(path: str = COOLDOWN_PERSIST_PATH):
    global _COOLDOWN_MAP
    data = safe_load_json(path) or {}
    out = {}
    for k,v in data.items():
        try:
            sym,mode = k.split("::")
            out[(sym,mode)] = float(v)
        except Exception:
            pass
    with _COOLDOWN_LOCK:
        _COOLDOWN_MAP = out
    logger.info("Cooldown map loaded (%d entries)", len(_COOLDOWN_MAP))

def persist_cooldown_map(path: str = COOLDOWN_PERSIST_PATH):
    with _COOLDOWN_LOCK:
        simple = {f"{k[0]}::{k[1]}": v for k,v in _COOLDOWN_MAP.items()}
    safe_write_atomic(path, simple)
    logger.debug("Cooldown map persisted")

def is_in_cooldown(symbol: str, mode: str) -> bool:
    key = (symbol.upper(), mode.lower())
    with _COOLDOWN_LOCK:
        ts = _COOLDOWN_MAP.get(key, 0)
    return ts > now_ts()

def set_cooldown(symbol: str, mode: str, seconds: int = SAME_COIN_BLOCK):
    key = (symbol.upper(), mode.lower())
    with _COOLDOWN_LOCK:
        _COOLDOWN_MAP[key] = now_ts() + seconds
    logger.info("Cooldown set %s %s until %s", symbol, mode, datetime.fromtimestamp(_COOLDOWN_MAP[key]).isoformat())

# ----------------- backtest record -----------------
def record_backtest(symbol: str, payload: Dict[str,Any], result: Dict[str,Any]):
    with _BACKTEST_LOCK:
        data = safe_load_json(BACKTEST_DB) or []
        data.append({"ts": now_ts(), "symbol": symbol, "payload": payload, "result": result})
        if len(data) > 10000: data = data[-10000:]
        safe_write_atomic(BACKTEST_DB, data)

# ----------------- message formatting (emoji + HTML) -----------------
def format_signal_message(symbol: str, mode: str, score: float, entry: float, tp: float, sl: float, atr_val: float, reasons: List[str], vol_est: float):
    mode_u = mode.upper()
    emoji = "🔥" if mode_u=="QUICK" else "⚡" if mode_u=="MID" else "💎"
    danger_low = entry - atr_val
    danger_high = entry + atr_val
    lines = []
    lines.append(f"{emoji} <b>{'BUY' if score>=0 else 'SIGNAL'} — {mode_u}</b>")
    lines.append(f"💱 <b>Pair:</b> {symbol}/USDT   📊 <b>Score:</b> {score}%")
    lines.append(f"💰 <b>Entry:</b> <code>{entry:.8f}</code>")
    lines.append(f"🎯 <b>TP:</b> <code>{tp:.8f}</code>   🛑 <b>SL:</b> <code>{sl:.8f}</code>")
    lines.append(f"📦 <b>Qty:</b> (manual)   💵 <b>Size:</b> (manual)   ⚡ <b>Lev:</b> (manual)")
    lines.append(f"⚠️ <b>Danger Zone:</b> <code>{danger_low:.8f}</code> — <code>{danger_high:.8f}</code>  (ATR={atr_val:.6f})")
    lines.append(f"📌 <b>Reason:</b> {' · '.join(reasons) if reasons else '-'}")
    lines.append(f"⏱ <b>Est time to TP:</b> (est)  |  🎛 <b>Mode:</b> {mode_u}")
    return "\n".join(lines)

def send_telegram(text: str):
    if NOTIFY_ONLY or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("NOTIFY_ONLY or telegram not configured — preview:\n%s", text)
        return True
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            logger.info("Telegram sent")
            return True
        else:
            logger.error("Telegram failed %s %s", r.status_code, r.text)
            return False
    except Exception:
        logger.exception("Telegram send exception")
        return False

# ----------------- scan + decide + emit -----------------
def scan_and_maybe_emit(exchange, symbol: str, mode: str = "mid", timeframe: str = "1m"):
    pair = symbol if "/" in symbol else f"{symbol}/USDT"
    ohlcv = fetch_ohlcv_with_retry(exchange, pair, timeframe=timeframe, limit=240)
    if not ohlcv:
        return False
    df = ohlcv_to_df(ohlcv)
    if df is None or len(df) < 40:
        return False
    df = add_indicators(df)
    score_info = compute_signal_score(df)
    score = score_info.get("score", 0.0)
    # mode specific threshold
    thr = MODE_THRESHOLDS.get(mode.lower(), MIN_SIGNAL_SCORE)
    if score < thr:
        logger.debug("%s %s score %.1f < thr %.1f", symbol, mode, score, thr)
        return False
    # volume 24h approx
    vol_avg = df['volume'].tail(1440).mean() if len(df) > 1440 else df['volume'].tail(100).mean() * 24
    spread_pct = ((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]) * 100
    # filter
    if vol_avg < MIN_24H_VOLUME:
        logger.debug("%s blocked low volume %.1f", symbol, vol_avg); return False
    if spread_pct > MAX_SPREAD_PCT:
        logger.debug("%s blocked spread %.3f", symbol, spread_pct); return False
    # cooldown
    if is_in_cooldown(symbol, mode):
        logger.debug("%s in cooldown %s", symbol, mode); return False
    # compute tp/sl
    entry = score_info['entry']; atrv = score_info['atr']
    tp, sl = adaptive_tp_sl(entry, atrv, mode, score)
    msg = format_signal_message(symbol, mode, score, entry, tp, sl, atrv, score_info.get('reasons',[]), vol_avg)
    ok = send_telegram(msg)
    if ok:
        # record log & backtest stub
        record_backtest(symbol, {"mode":mode,"score":score,"entry":entry,"tp":tp,"sl":sl}, {"tp_hit":None})
        set_cooldown(symbol, mode, seconds=SAME_COIN_BLOCK)
    return ok

# ----------------- helper: safe symbol resolver (try many variants) -----------------
def resolve_symbol_if_needed(exchange, symbol: str) -> Optional[str]:
    if "/" in symbol:
        return symbol
    try:
        markets = getattr(exchange, "markets", None)
        if not markets:
            try:
                exchange.load_markets()
                markets = exchange.markets
            except Exception:
                markets = None
        # prefer SYMBOL/USDT
        cand = symbol + "/USDT"
        if markets and cand in markets:
            return cand
        # search markets by base symbol
        if markets:
            for m in markets:
                if m.startswith(symbol + "/"):
                    return m
        # fallback return SYMBOL/USDT
        return symbol + "/USDT"
    except Exception:
        return symbol + "/USDT"

# ----------------- end of helpers.py -----------------