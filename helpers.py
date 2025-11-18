# helpers.py — PART 1 (CORE)
from __future__ import annotations
import os
import time
import math
import json
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ccxt
import requests

# dotenv optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -----------------------------
# Logging & config
# -----------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("helpers")

# Constants & paths
COOLDOWN_FILE = os.environ.get("COOLDOWN_FILE", "cooldowns.json")
BACKTEST_DB = os.environ.get("BACKTEST_DB", "backtest_history.json")
TRADE_LOG = os.environ.get("TRADE_LOG", "trade_log.json")
COINS_CSV = os.environ.get("COINS_CSV", "coins.csv")

# Defaults
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "20"))
OHLCV_MAX_RETRIES = int(os.environ.get("OHLCV_MAX_RETRIES", "3"))
OHLCV_RETRY_DELAY = float(os.environ.get("OHLCV_RETRY_DELAY", "0.8"))

# Scoring weights (tweakable via env)
WEIGHTS = {
    "ema": float(os.environ.get("W_EMA", 1.0)),
    "rsi": float(os.environ.get("W_RSI", 1.0)),
    "macd": float(os.environ.get("W_MACD", 1.0)),
    "adx": float(os.environ.get("W_ADX", 1.0)),
    "volume": float(os.environ.get("W_VOLUME", 1.0)),
    "supertrend": float(os.environ.get("W_SUPERTREND", 1.0)),
    "vwap": float(os.environ.get("W_VWAP", 1.0)),
    "volatility": float(os.environ.get("W_VOL", 1.0)),
}

# Cooldowns (seconds)
COOLDOWNS = {
    "QUICK": int(os.environ.get("CD_QUICK", 60 * 10)),   # 10 min
    "MID":   int(os.environ.get("CD_MID", 60 * 30)),     # 30 min
    "TREND": int(os.environ.get("CD_TREND", 60 * 60)),   # 60 min
}
SAME_COIN_BLOCK = int(os.environ.get("SAME_COIN_BLOCK", 30 * 60))  # 30 min

# Telegram env
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# thread lock
_file_lock = threading.Lock()

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class ExchangeConfig:
    id: str = os.environ.get("EXCHANGE_ID", "binance")
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    enable_rate_limit: bool = True
    options: Dict[str, Any] = None

# -----------------------------
# Utility helpers
# -----------------------------
def now_ts() -> float:
    return time.time()

def safe_ffill(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.ffill()

def safe_bfill(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.bfill()

def round_tick(price: float) -> float:
    if price < 1:
        return 0.0001
    if price < 100:
        return 0.01
    if price < 1000:
        return 0.1
    return 1.0

# -----------------------------
# Load coins
# -----------------------------
def load_coins(path: str = COINS_CSV) -> List[str]:
    if not os.path.exists(path):
        log.warning("coins.csv not found: %s", path)
        return []
    try:
        df = pd.read_csv(path, header=0)
        if "symbol" in df.columns:
            syms = df["symbol"].dropna().astype(str).str.strip().str.upper().tolist()
        else:
            syms = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
        return [s for s in syms if s]
    except Exception:
        log.exception("load_coins error")
        return []

# -----------------------------
# Exchange factory (ccxt)
# -----------------------------
def create_exchange(cfg: Optional[ExchangeConfig] = None) -> ccxt.Exchange:
    cfg = cfg or ExchangeConfig()
    exchange_id = cfg.id
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        exchange_cls = getattr(ccxt, exchange_id.lower(), None)
    if exchange_cls is None:
        raise RuntimeError(f"Exchange {exchange_id} not available in ccxt")
    opts = {"enableRateLimit": cfg.enable_rate_limit}
    if cfg.options:
        opts.update(cfg.options)
    ex = exchange_cls(opts)
    key = cfg.api_key or os.environ.get("EXCHANGE_API_KEY")
    secret = cfg.api_secret or os.environ.get("EXCHANGE_API_SECRET")
    if key and secret:
        ex.apiKey = key
        ex.secret = secret
    try:
        # set session timeout if possible
        if hasattr(ex, "timeout"):
            ex.timeout = HTTP_TIMEOUT * 1000
    except Exception:
        pass
    try:
        # try to load markets (best-effort)
        ex.load_markets()
    except Exception:
        pass
    log.info("Exchange created: %s (api set: %s)", exchange_id, bool(key and secret))
    return ex

# -----------------------------
# Fetch OHLCV robust
# -----------------------------
def fetch_ohlcv_sample(ex: ccxt.Exchange, symbol: str, timeframe: str = "1m", limit: int = 300, retries: int = OHLCV_MAX_RETRIES) -> Optional[pd.DataFrame]:
    # normalize pair
    pair = symbol if "/" in symbol else f"{symbol}/USDT"
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            raw = ex.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            if not raw:
                log.warning("No ohlcv for %s", pair)
                return None
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # numeric coercion
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].apply(pd.to_numeric, errors='coerce')
            df = safe_ffill(df)
            df = safe_bfill(df)
            return df
        except ccxt.BadSymbol:
            log.error("BadSymbol: %s", pair)
            return None
        except Exception as e:
            last_exc = e
            log.debug("OHLCV fetch attempt %d for %s failed: %s", attempt, pair, e)
            time.sleep(OHLCV_RETRY_DELAY * attempt)
    log.error("Failed to fetch OHLCV for %s after %d attempts. last_err=%s", pair, retries, last_exc)
    return None

# -----------------------------
# Basic indicators (used by Part2)
# -----------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev).abs()
    tr3 = (df['low'] - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(window=period, min_periods=1).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = true_range(df)
    atr_series = tr.rolling(period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).sum() / (atr_series + 1e-12))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).sum() / (atr_series + 1e-12))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)) * 100
    adx_series = dx.rolling(period, min_periods=1).mean()
    return adx_series.fillna(0)

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pv = (tp * df['volume']).cumsum()
    denom = df['volume'].cumsum().replace(0, np.nan)
    return (pv / denom).ffill().fillna(df['close'])

def volatility_std(series: pd.Series, period: int = 20) -> pd.Series:
    returns = series.pct_change().fillna(0)
    return returns.rolling(window=period, min_periods=1).std() * np.sqrt(period)

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    hl2 = (df['high'] + df['low']) / 2
    atr_s = atr(df, period)
    upperband = hl2 + (multiplier * atr_s)
    lowerband = hl2 - (multiplier * atr_s)
    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    st = pd.Series(index=df.index, dtype='bool')
    for i in range(len(df)):
        if i == 0:
            final_upper.iat[i] = upperband.iat[i]
            final_lower.iat[i] = lowerband.iat[i]
            st.iat[i] = True
            continue
        if (upperband.iat[i] < final_upper.iat[i-1]) or (df['close'].iat[i-1] > final_upper.iat[i-1]):
            final_upper.iat[i] = upperband.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i-1]
        if (lowerband.iat[i] > final_lower.iat[i-1]) or (df['close'].iat[i-1] < final_lower.iat[i-1]):
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i-1]
        if st.iat[i-1] and (df['close'].iat[i] <= final_upper.iat[i]):
            st.iat[i] = True
        elif (not st.iat[i-1]) and (df['close'].iat[i] >= final_lower.iat[i]):
            st.iat[i] = False
        else:
            st.iat[i] = st.iat[i-1]
    return st

# End of Part 1
# helpers.py — PART 2 (ADVANCED: scoring, adaptive TP/SL, risk, cooldown, backtest, emit)

# ---- persistence helpers ----
_LOCK = threading.Lock()

def _load_json_safe(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _save_json_safe(path: str, data: Any):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

# ---- cooldown storage ----
def _load_cooldown_storage(path: str = COOLDOWN_FILE) -> Dict[str, Any]:
    data = _load_json_safe(path) or {}
    data.setdefault("last_signal", {})
    data.setdefault("blocks", {})
    data.setdefault("history", [])
    return data

def _save_cooldown_storage(data: Dict[str, Any], path: str = COOLDOWN_FILE):
    _save_json_safe(path, data)

def is_in_cooldown(symbol: str, mode: str) -> bool:
    data = _load_cooldown_storage()
    now = time.time()
    last = float(data.get("last_signal", {}).get(symbol, 0))
    cd = COOLDOWNS.get(mode, COOLDOWNS.get("QUICK", 600))
    if last and now - last < cd:
        return True
    block_until = float(data.get("blocks", {}).get(symbol, 0))
    if block_until and now < block_until:
        return True
    return False

def set_cooldown(symbol: str, mode: str):
    data = _load_cooldown_storage()
    now = time.time()
    data.setdefault("last_signal", {})[symbol] = now
    data.setdefault("blocks", {})[symbol] = now + float(SAME_COIN_BLOCK)
    hist = data.setdefault("history", [])
    hist.append({"symbol": symbol, "mode": mode, "ts": now})
    if len(hist) > 2000:
        data["history"] = hist[-2000:]
    _save_cooldown_storage(data)

# ---- scoring engine ----
def compute_signal_score(df: pd.DataFrame) -> Dict[str, Any]:
    close = df['close']
    vol = df['volume']
    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    ema200 = ema(close, 200).iloc[-1]
    ema_trend = (ema20 - ema50) / (abs(ema50) + 1e-12)
    rsi_val = float(rsi(close, 14).iloc[-1])
    macd_line, signal_line, hist = macd(close)
    macd_h = float(hist.iloc[-1])
    adx_val = float(adx(df, 14).iloc[-1])
    atr_val = float(atr(df, 14).iloc[-1] or 0.0)
    vwap_val = float(vwap(df).iloc[-1])
    vol_std = float(volatility_std(close, 20).iloc[-1] or 0.0)
    st = supertrend(df, period=10, multiplier=3)
    st_bull = bool(st.iloc[-1])

    score = 0.0
    reasons = []

    # EMA
    if ema_trend > 0:
        score += min(ema_trend * 100 * WEIGHTS.get('ema', 1.0), 30)
        reasons.append("EMA_up")
    else:
        score += max(ema_trend * 100 * WEIGHTS.get('ema', 1.0), -30)
        reasons.append("EMA_down")

    # RSI
    if rsi_val < 30:
        score += 15 * WEIGHTS.get('rsi', 1.0)
        reasons.append("RSI_oversold")
    elif rsi_val < 45:
        score += 6 * WEIGHTS.get('rsi', 1.0)
    elif rsi_val > 70:
        score -= 12 * WEIGHTS.get('rsi', 1.0)
        reasons.append("RSI_overbought")

    # MACD
    score += (10 if macd_h > 0 else -8) * WEIGHTS.get('macd', 1.0)
    if macd_h > 0:
        reasons.append("MACD_pos")

    # ADX
    if adx_val > 25:
        score += 8 * WEIGHTS.get('adx', 1.0)
        reasons.append("ADX_strong")

    # volume spike
    try:
        median_vol = vol.tail(100).median()
        vol_spike = vol.iloc[-1] > (median_vol * 1.8 if median_vol > 0 else 0)
    except Exception:
        vol_spike = False
    if vol_spike:
        score += 10 * WEIGHTS.get('volume', 1.0)
        reasons.append("Vol_spike")

    # supertrend
    score += (18 if st_bull else -15) * WEIGHTS.get('supertrend', 1.0)
    if st_bull:
        reasons.append("Supertrend_bull")

    # VWAP
    if close.iloc[-1] > vwap_val:
        score += 3 * WEIGHTS.get('vwap', 1.0)
    else:
        score -= 2 * WEIGHTS.get('vwap', 1.0)

    # volatility penalty
    if vol_std > 0.06:
        score -= 12 * WEIGHTS.get('volatility', 1.0)
        reasons.append("High_vol")

    score = max(min(score, 100.0), -100.0)
    return {
        "ema20": float(ema20), "ema50": float(ema50), "ema200": float(ema200),
        "ema_trend": float(ema_trend), "rsi": rsi_val, "macd_hist": macd_h,
        "adx": adx_val, "atr": atr_val, "vwap": vwap_val,
        "volatility": vol_std, "supertrend": st_bull,
        "vol_spike": vol_spike, "score": float(score), "reasons": reasons
    }

# ---- signal conditions (quick/mid/trend) ----
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['rsi14'] = rsi(df['close'], 14)
    macd_line, macd_signal, hist = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist'] = hist
    df['adx14'] = adx(df, 14)
    df['vol_ma20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['vol_rel'] = df['volume'] / df['vol_ma20'].replace(0, 1)
    df['candle_body'] = (df['close'] - df['open']).abs()
    df['candle_body_rel'] = df['candle_body'] / df['close']
    return df

def detect_regime(df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    if latest['ema20'] > latest['ema50'] > latest['ema200']:
        return "bull"
    if latest['ema20'] < latest['ema50'] < latest['ema200']:
        return "bear"
    return "sideways"

def quick_condition(df: pd.DataFrame) -> Optional[str]:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    ema_cross_buy = (prev['ema20'] <= prev['ema50']) and (latest['ema20'] > latest['ema50'])
    ema_cross_sell = (prev['ema20'] >= prev['ema50']) and (latest['ema20'] < latest['ema50'])
    vol_spike = latest['vol_rel'] > 2.0
    macd_momentum = latest['macd_hist'] > 0
    if ema_cross_buy and vol_spike and macd_momentum and latest['rsi14'] < 80:
        return "BUY"
    if ema_cross_sell and vol_spike and not macd_momentum and latest['rsi14'] > 20:
        return "SELL"
    return None

def mid_condition(df: pd.DataFrame) -> Optional[str]:
    latest = df.iloc[-1]
    strong_trend = latest['adx14'] > 20
    bull = latest['ema50'] > latest['ema200']
    bear = latest['ema50'] < latest['ema200']
    if bull and strong_trend and latest['macd'] > latest['macd_signal']:
        return "BUY"
    if bear and strong_trend and latest['macd'] < latest['macd_signal']:
        return "SELL"
    return None

def trend_condition(df: pd.DataFrame) -> Optional[str]:
    latest = df.iloc[-1]
    if latest['ema20'] > latest['ema50'] > latest['ema200'] and latest['adx14'] > 25 and latest['macd_hist'] > 0:
        return "BUY"
    if latest['ema20'] < latest['ema50'] < latest['ema200'] and latest['adx14'] > 25 and latest['macd_hist'] < 0:
        return "SELL"
    return None

# ---- adaptive TP/SL & leverage & position sizing ----
def adaptive_tp_sl(entry: float, atr_val: float, score: float, mode: str) -> Tuple[float, float]:
    if atr_val is None or atr_val <= 0:
        # fallback percents
        if mode == "TREND":
            return round(entry * 1.01, 8), round(entry * 0.995, 8)
        if mode == "MID":
            return round(entry * 1.006, 8), round(entry * 0.997, 8)
        return round(entry * 1.003, 8), round(entry * 0.998, 8)
    # base multipliers
    if mode == "TREND":
        tp_mul, sl_mul = 1.4, 1.0
    elif mode == "MID":
        tp_mul, sl_mul = 1.1, 1.2
    else:
        tp_mul, sl_mul = 0.9, 1.4
    score_factor = max(0.8, min(1.4, 1.0 + (score - 60) / 300.0))
    tp_move = max(1e-8, tp_mul * score_factor * atr_val)
    sl_move = max(1e-8, sl_mul / score_factor * atr_val)
    tp = entry + tp_move
    sl = entry - sl_move
    return round(tp, 8), round(sl, 8)

def decide_mode_and_leverage(score: float) -> Tuple[str, int]:
    if score >= 85:
        return "TREND", 50
    if score >= 70:
        return "MID", 25
    if score >= 45:
        return "QUICK", 10
    return "NO_TRADE", 1

def position_size(balance: float, risk_pct: float, entry: float, sl: float, leverage: int = 1) -> float:
    risk_money = balance * (risk_pct / 100.0)
    per_unit_risk = abs(entry - sl) * leverage
    if per_unit_risk <= 1e-12:
        return 0.0
    size = risk_money / per_unit_risk
    return float(max(0.0, size))

# ---- backtest / feedback helpers ----
def _load_backtest(path: str = BACKTEST_DB) -> List[Dict[str, Any]]:
    return _load_json_safe(path) or []

def _save_backtest(data: List[Dict[str, Any]], path: str = BACKTEST_DB):
    _save_json_safe(path, data)

def record_signal_result(symbol: str, payload: Dict[str, Any], result: Dict[str, Any]):
    data = _load_backtest() or []
    data.append({"ts": time.time(), "symbol": symbol, "payload": payload, "result": result})
    if len(data) > 10000:
        data = data[-10000:]
    _save_backtest(data)

def feedback_adjusted_score(raw_score: float, symbol: str) -> float:
    data = _load_backtest()
    if not data:
        return raw_score
    recent = [d for d in data if d.get("symbol") == symbol]
    recent = recent[-200:]
    if not recent:
        return raw_score
    tp_hits = sum(1 for r in recent if r.get("result", {}).get("tp_hit"))
    rate = tp_hits / (len(recent) + 1e-9)
    adjust = (rate - 0.5) * 10 * 0.05
    return float(max(0.0, min(100.0, raw_score * (1 + adjust))))

# ---- evaluate all modes & choose best ----
def evaluate_all_modes(df: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
    dfx = add_indicators(df)
    base = compute_signal_score(dfx)
    raw_score = float(base.get("score", 0.0))
    candidates = []
    for mode in ("QUICK", "MID", "TREND"):
        if mode == "QUICK":
            mod = raw_score * 0.98
        elif mode == "MID":
            mod = raw_score * 1.00
        else:
            mod = raw_score * 1.03
        adjusted = feedback_adjusted_score(mod, symbol)
        entry = float(dfx['close'].iloc[-1])
        atr_val = float(atr(dfx, 14).iloc[-1] or 0.0)
        tp, sl = adaptive_tp_sl(entry, atr_val, adjusted, mode)
        if mode == "QUICK":
            direction = quick_condition(dfx)
        elif mode == "MID":
            direction = mid_condition(dfx)
        else:
            direction = trend_condition(dfx)
        if direction:
            candidates.append({
                "mode": mode, "direction": direction, "score": adjusted,
                "entry": entry, "tp": tp, "sl": sl, "atr": atr_val
            })
    if not candidates:
        return None
    best = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
    return best

# ---- danger zone ----
def compute_danger_zone(df: pd.DataFrame, mult: float = 1.5) -> str:
    last = float(df['close'].iloc[-1])
    atr_val = float(atr(df, 14).iloc[-1] or 0.0)
    low = last - mult * atr_val
    high = last + mult * atr_val
    return f"{low:.8f} — {high:.8f} (ATR={atr_val:.8f})"

# ---- telegram ----
def send_telegram_message(text: str) -> bool:
    tkn = TELEGRAM_TOKEN
    cid = TELEGRAM_CHAT_ID
    if not tkn or not cid:
        log.debug("telegram not configured")
        return False
    url = f"https://api.telegram.org/bot{tkn}/sendMessage"
    payload = {"chat_id": cid, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        log.exception("telegram send failed")
        return False

# ---- final analyze & emit pipeline ----
def analyze_and_emit(exchange: ccxt.Exchange, symbol: str,
                     timeframe: str = "1m", limit: int = 300,
                     pair_suffix: str = "/USDT",
                     balance_for_risk: float = 100.0,
                     risk_pct: float = 1.0,
                     send_telegram_flag: bool = True) -> Optional[Dict[str, Any]]:
    pair = symbol if "/" in symbol else f"{symbol}{pair_suffix}"
    df = fetch_ohlcv_sample(exchange, pair, timeframe=timeframe, limit=limit)
    if df is None or df.empty:
        return None
    best = evaluate_all_modes(df, symbol)
    if not best:
        return None
    if is_in_cooldown(symbol, best['mode']):
        log.info("cooldown blocks %s for %s", symbol, best['mode'])
        return None
    regime = detect_regime(add_indicators(df))
    # decide leverage
    _, lev = decide_mode_and_leverage(best['score'])
    # position sizing
    entry = float(best['entry']); tp = float(best['tp']); sl = float(best['sl'])
    size_quote = position_size(balance_for_risk, risk_pct, entry, sl, lev)
    qty = size_quote / entry if entry > 0 else 0.0
    danger = compute_danger_zone(df)
    # message
    msg = (
        f"🔥 <b>{best['direction']} SIGNAL — {best['mode']}</b>\n"
        f"<b>Pair:</b> {pair}  <b>Score:</b> {best['score']:.1f}\n"
        f"<b>Entry:</b> <code>{entry:.8f}</code>\n"
        f"<b>TP:</b> <code>{tp:.8f}</code>  <b>SL:</b> <code>{sl:.8f}</code>\n"
        f"<b>Qty:</b> {qty:.6f}  <b>Size(quote):</b> {size_quote:.2f}  <b>Lev:</b> {lev}x\n"
        f"<b>Danger:</b> {danger}\n"
        f"<b>Reason:</b> {', '.join(compute_signal_score(df).get('reasons', []))}\n"
    )
    sent = False
    if send_telegram_flag:
        sent = send_telegram_message(msg)
    # log trade
    entry_log = {"ts": time.time(), "symbol": symbol, "pair": pair, "payload": best, "sent": sent}
    try:
        data = _load_json_safe(TRADE_LOG) or []
        data.append(entry_log)
        if len(data) > 10000:
            data = data[-10000:]
        _save_json_safe(TRADE_LOG, data)
    except Exception:
        log.exception("trade log save failed")
    # set cooldown if message sent
    if sent:
        set_cooldown(symbol, best['mode'])
    else:
        # soft block
        data = _load_cooldown_storage()
        now = time.time()
        data.setdefault("last_signal", {})[symbol] = now
        data.setdefault("blocks", {})[symbol] = now + max(60, int(COOLDOWNS.get(best['mode'], 300) / 3))
        _save_cooldown_storage(data)
    return {"symbol": symbol, "pair": pair, "mode": best['mode'], "direction": best['direction'], "entry": entry, "tp": tp, "sl": sl, "qty": qty, "size_quote": size_quote, "lev": lev, "telegram_sent": sent}

# ---- backtest helper (local) ----
def backtest_on_df(df: pd.DataFrame, symbol: str, lookahead_bars: int = 20) -> Dict[str, Any]:
    df = add_indicators(df)
    samples = []
    for i in range(80, len(df) - lookahead_bars - 1):
        slice_df = df.iloc[:i+1].reset_index(drop=True)
        cand = evaluate_all_modes(slice_df, symbol)
        if not cand:
            continue
        entry = cand['entry']; tp = cand['tp']; sl = cand['sl']
        future = df.iloc[i+1:i+1+lookahead_bars]
        tp_hit = any(future['high'] >= tp)
        sl_hit = any(future['low'] <= sl)
        samples.append({"ts": slice_df.index[-1], "entry": entry, "tp": tp, "sl": sl, "tp_hit": bool(tp_hit), "sl_hit": bool(sl_hit), "mode": cand['mode'], "score": cand['score']})
    total = len(samples)
    tp_hits = sum(1 for s in samples if s['tp_hit'])
    sl_hits = sum(1 for s in samples if s['sl_hit'])
    return {"total": total, "tp_hits": tp_hits, "sl_hits": sl_hits, "tp_rate_pct": (tp_hits/total*100) if total else 0, "samples": samples[:500]}

# End of Part 2