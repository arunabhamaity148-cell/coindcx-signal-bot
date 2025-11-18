# helpers.py  -- PART 1 (CORE / REQUIRED)
# This is Part 1. After you paste this and confirm "Done",
# I'll give Part 2 which depends on these functions/constants.

import os
import time
import math
import logging
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import ccxt
from dotenv import load_dotenv

# load env
load_dotenv()

# ---------------------------
# Basic config (customize)
# ---------------------------
# File where cooldowns are persisted
COOLDOWN_FILE = os.getenv("COOLDOWN_FILE", "cooldowns.json")

# Telegram defaults (set in .env)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Cooldowns (seconds) for modes
COOLDOWNS = {
    "QUICK": int(os.getenv("CD_QUICK", 300)),   # 5 min default
    "MID": int(os.getenv("CD_MID", 1800)),      # 30 min
    "TREND": int(os.getenv("CD_TREND", 3600)),  # 60 min
}
# block same coin for N seconds after signal (extra safety)
SAME_COIN_BLOCK = int(os.getenv("SAME_COIN_BLOCK", 1800))  # 30 min

# Weights for scoring (Part2 uses)
WEIGHTS = {
    "ema": float(os.getenv("W_EMA", 1.0)),
    "rsi": float(os.getenv("W_RSI", 1.0)),
    "macd": float(os.getenv("W_MACD", 1.0)),
    "adx": float(os.getenv("W_ADX", 1.0)),
    "volume": float(os.getenv("W_VOLUME", 1.0)),
    "supertrend": float(os.getenv("W_SUPERTREND", 1.0)),
    "vwap": float(os.getenv("W_VWAP", 1.0)),
    "volatility": float(os.getenv("W_VOL", 1.0)),
}

# Default exchange id (binance by default)
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")

# Coins source file
COINS_CSV = os.getenv("COINS_CSV", "coins.csv")

# Timezone/timestamp helper
def now_ts() -> float:
    return time.time()

# ---------------------------
# Logger setup
# ---------------------------
def setup_logger(name: str = "bot", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

log = setup_logger("coindcx-bot")

# ---------------------------
# Exchange helper
# ---------------------------
def get_exchange(api_key: Optional[str] = None, secret: Optional[str] = None, exchange_id: str = EXCHANGE_ID, sandbox: bool=False) -> ccxt.Exchange:
    """
    Returns a configured ccxt exchange instance (public only if keys not provided).
    """
    exchange_class = getattr(ccxt, exchange_id)
    params = {}
    if sandbox:
        # some exchanges have sandbox URLs (not generic)
        params["enableRateLimit"] = True
    try:
        ex = exchange_class({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
            "timeout": 20000,
        })
        # set api keys if present in env or passed
        key = api_key or os.getenv("EXCHANGE_API_KEY")
        sec = secret or os.getenv("EXCHANGE_API_SECRET")
        if key and sec:
            ex.apiKey = key
            ex.secret = sec
        return ex
    except Exception as e:
        log.exception(f"Failed to init exchange {exchange_id}: {e}")
        raise

# ---------------------------
# Coins loader
# ---------------------------
def load_coins(path: str = COINS_CSV) -> List[str]:
    """
    Simple csv loader: expects first column header 'symbol' and each row symbol like BTC or ETH.
    Returns list of symbols (without /USDT suffix).
    """
    if not os.path.exists(path):
        log.warning(f"coins.csv not found at {path}, returning default list")
        return ["BTC","ETH","XRP","SOL","BNB","ADA"]
    try:
        df = pd.read_csv(path, header=0)
        if "symbol" in df.columns:
            coins = df["symbol"].dropna().astype(str).str.strip().tolist()
            # filter empties and uppercase
            return [c.upper() for c in coins if str(c).strip()]
        else:
            # fallback: first column
            coins = df.iloc[:,0].dropna().astype(str).str.strip().tolist()
            return [c.upper() for c in coins if str(c).strip()]
    except Exception as e:
        log.exception("Failed to load coins.csv")
        return ["BTC","ETH","XRP","SOL","BNB","ADA"]

# ---------------------------
# OHLCV fetcher (returns pandas DataFrame)
# ---------------------------
def get_ohlcv_sample(exchange: ccxt.Exchange, symbol: str, timeframe: str="1m", limit: int=300, pair_suffix: str="/USDT") -> pd.DataFrame:
    """
    Fetch OHLCV for symbol from exchange and return DataFrame with columns:
    ['timestamp','open','high','low','close','volume'] indexed by timestamp (ms).
    symbol param expected like 'BTC' -> will use 'BTC/USDT' by default.
    """
    pair = f"{symbol}{pair_suffix}"
    try:
        # unify symbol name for ccxt if needed
        if hasattr(exchange, "market") and hasattr(exchange, "load_markets"):
            try:
                # ensure markets loaded
                exchange.load_markets()
            except Exception:
                pass

        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        # ensure numeric
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].apply(pd.to_numeric, errors='coerce')
        return df
    except ccxt.BaseError as e:
        log.error(f"ccxt error fetching {pair}: {e}")
        raise
    except Exception as e:
        log.exception(f"Error fetching ohlcv {pair}: {e}")
        raise

# ---------------------------
# Indicators (vectorized) - simple and robust
# ---------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series: pd.Series, fast: int=12, slow: int=26, signal: int=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period).sum() / (atr_series + 1e-12))
    minus_di = 100 * (minus_dm.rolling(period).sum() / (atr_series + 1e-12))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)) * 100
    adx = dx.rolling(window=period, min_periods=1).mean()
    return adx.fillna(0)

def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["high"] + df["low"] + df["close"]) / 3 * df["volume"]
    cumsum_pv = pv.cumsum()
    cumsum_v = df["volume"].cumsum().replace(0, np.nan)
    return (cumsum_pv / cumsum_v).fillna(method="ffill").fillna(df["close"])

def supertrend(df: pd.DataFrame, period: int=10, multiplier: float=3.0) -> pd.Series:
    """
    Return boolean series: True -> bullish, False -> bearish
    Implementation: ATR-based SuperTrend (simplified).
    """
    hl2 = (df["high"] + df["low"]) / 2
    atr_s = atr(df, period)
    upperband = hl2 + (multiplier * atr_s)
    lowerband = hl2 - (multiplier * atr_s)

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    st = pd.Series(index=df.index, dtype="bool")

    for i in range(len(df)):
        if i == 0:
            final_upper.iat[i] = upperband.iat[i]
            final_lower.iat[i] = lowerband.iat[i]
            st.iat[i] = True
            continue
        # final upper
        if (upperband.iat[i] < final_upper.iat[i-1]) or (df["close"].iat[i-1] > final_upper.iat[i-1]):
            final_upper.iat[i] = upperband.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i-1]
        # final lower
        if (lowerband.iat[i] > final_lower.iat[i-1]) or (df["close"].iat[i-1] < final_lower.iat[i-1]):
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i-1]
        # state
        if st.iat[i-1] and (df["close"].iat[i] <= final_upper.iat[i]):
            st.iat[i] = True
        elif (not st.iat[i-1]) and (df["close"].iat[i] >= final_lower.iat[i]):
            st.iat[i] = False
        else:
            st.iat[i] = st.iat[i-1]
    return st

def volatility_std(series: pd.Series, window: int=20) -> pd.Series:
    returns = series.pct_change().fillna(0)
    return returns.rolling(window=window, min_periods=1).std()

# ---------------------------
# Small utils
# ---------------------------
def round_tick(price: float) -> float:
    """Return tick rounding size for a price"""
    if price < 1:
        return 0.0001
    if price < 100:
        return 0.01
    if price < 1000:
        return 0.1
    return 1.0

def safe_div(a: float, b: float, default: float=0.0) -> float:
    try:
        return a / b if b else default
    except Exception:
        return default

# ---------------------------
# End of Part 1
# ---------------------------
if __name__ == "__main__":
    # quick sanity self-test (without keys)
    log.info("helpers PART1 self-test start")
    ex = get_exchange()  # public
    coins = load_coins()
    log.info(f"Loaded coins: {coins[:10]}")
    # fetch a single sample (non blocking)
    try:
        df = get_ohlcv_sample(ex, coins[0], timeframe="1m", limit=100)
        log.info(f"OHLCV sample rows: {len(df)}")
        # compute simple indicators
        log.info(f"EMA20 last: {ema(df['close'],20).iloc[-1]:.6f}")
    except Exception as e:
        log.warning("Could not fetch market sample in self-test (public mode).")
# -------------------- helpers.py — PART 2 (FINAL) --------------------
# Append this to the end of Part1 (which must define indicators, exchange, config, log)

import time
import json
import threading
import requests
from typing import List, Dict, Any, Optional, Tuple

_lock = threading.Lock()

# -------------------------
# Persistence helpers for cooldown/history
# -------------------------
def _load_json_safe(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json_safe(path: str, data: Dict[str, Any]):
    try:
        tmp = path + ".tmp"
        with _lock:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            # atomic replace
            try:
                os.replace(tmp, path)
            except Exception:
                # fallback
                with open(path, "w") as f2:
                    json.dump(data, f2, indent=2, default=str)
    except Exception as e:
        log.exception("Failed to save json %s: %s", path, e)


# -------------------------
# Cooldown storage (persistent)
# -------------------------
def _load_cooldown_storage(path: str = COOLDOWN_FILE) -> Dict[str, Any]:
    data = _load_json_safe(path)
    if not data:
        return {"last_signal": {}, "blocks": {}, "history": []}
    # ensure keys
    data.setdefault("last_signal", {})
    data.setdefault("blocks", {})
    data.setdefault("history", [])
    return data

def _save_cooldown_storage(data: Dict[str, Any], path: str = COOLDOWN_FILE):
    _save_json_safe(path, data)

def is_in_cooldown(symbol: str, mode: str) -> bool:
    """Return True if symbol is blocked by mode cooldown or same-coin block."""
    data = _load_cooldown_storage()
    now = time.time()
    last_signals = data.get("last_signal", {})
    last_ts = float(last_signals.get(symbol, 0))
    cd = COOLDOWNS.get(mode, COOLDOWNS.get("QUICK", 300))
    if last_ts and (now - last_ts) < cd:
        return True
    blocks = data.get("blocks", {})
    block_until = float(blocks.get(symbol, 0))
    if block_until and now < block_until:
        return True
    return False

def set_cooldown(symbol: str, mode: str):
    data = _load_cooldown_storage()
    now = time.time()
    data.setdefault("last_signal", {})[symbol] = now
    # set same coin block
    data.setdefault("blocks", {})[symbol] = now + float(SAME_COIN_BLOCK)
    # append history (short)
    hist = data.setdefault("history", [])
    hist.append({"symbol": symbol, "mode": mode, "ts": now})
    # keep recent 200 entries
    if len(hist) > 200:
        data["history"] = hist[-200:]
    _save_cooldown_storage(data)


# -------------------------
# Scoring (wrapper around indicators) — returns dict with values + score + reasons
# -------------------------
def compute_signal_score(df: 'pd.DataFrame') -> Dict[str, Any]:
    """
    Build composite score using EMAs, RSI, MACD hist, ADX, VWAP, SuperTrend, Volume spike, Volatility.
    Returns dict with fields: score(float), reasons(list), and indicator values for debug.
    """
    try:
        close = df['close']
        vol = df['volume']

        # basic indicators
        ema20 = float(ema(close, 20).iloc[-1])
        ema50 = float(ema(close, 50).iloc[-1])
        ema200 = float(ema(close, 200).iloc[-1])
        ema_trend = (ema20 - ema50) / (abs(ema50) + 1e-12)

        rsi_val = float(rsi(close, 14).iloc[-1])

        macd_line, macd_signal, macd_hist = macd(close)
        macd_h = float(macd_hist.iloc[-1])

        adx_val = float(adx(df, 14).iloc[-1])

        atr_val = float(atr(df, 14).iloc[-1])

        vwap_val = float(vwap(df).iloc[-1])

        vol_std = float(volatility_std(close, 20).iloc[-1])

        st = supertrend(df, period=10, multiplier=3)
        st_bull = bool(st.iloc[-1])

        # compute score using WEIGHTS (from Part1)
        score = 0.0
        reasons: List[str] = []

        # EMA
        if ema_trend > 0:
            score += min(ema_trend * 100 * WEIGHTS.get("ema", 1.0), 30)
            reasons.append("EMA_up")
        else:
            score += max(ema_trend * 100 * WEIGHTS.get("ema", 1.0), -30)
            reasons.append("EMA_down")

        # RSI
        if rsi_val < 30:
            score += 15 * WEIGHTS.get("rsi", 1.0)
            reasons.append("RSI_oversold")
        elif rsi_val < 45:
            score += 6 * WEIGHTS.get("rsi", 1.0)
        elif rsi_val > 70:
            score -= 12 * WEIGHTS.get("rsi", 1.0)
            reasons.append("RSI_overbought")

        # MACD
        score += (10 if macd_h > 0 else -8) * WEIGHTS.get("macd", 1.0)
        if macd_h > 0:
            reasons.append("MACD_pos")

        # ADX
        if adx_val > 25:
            score += 8 * WEIGHTS.get("adx", 1.0)
            reasons.append("ADX_strong")

        # Volume spike
        try:
            median_vol = vol.tail(100).median()
            vol_spike = bool(vol.iloc[-1] > (median_vol * 1.8 if median_vol > 0 else 0))
        except Exception:
            vol_spike = False
        if vol_spike:
            score += 10 * WEIGHTS.get("volume", 1.0)
            reasons.append("Volume_spike")

        # SuperTrend
        score += (18 if st_bull else -15) * WEIGHTS.get("supertrend", 1.0)
        if st_bull:
            reasons.append("Supertrend_bull")

        # VWAP alignment
        if close.iloc[-1] > vwap_val:
            score += 3 * WEIGHTS.get("vwap", 1.0)
        else:
            score -= 2 * WEIGHTS.get("vwap", 1.0)

        # Volatility penalty
        if vol_std > 0.06:
            score -= 12 * WEIGHTS.get("volatility", 1.0)
            reasons.append("High_vol")

        # clamp and normalize small negative to 0 floor for easier thresholds
        score = max(min(score, 100.0), -100.0)

        return {
            "ema20": ema20, "ema50": ema50, "ema200": ema200,
            "ema_trend": ema_trend, "rsi": rsi_val, "macd_hist": macd_h,
            "adx": adx_val, "atr": atr_val, "vwap": vwap_val,
            "volatility": vol_std, "supertrend": st_bull,
            "vol_spike": vol_spike,
            "score": float(score),
            "reasons": reasons
        }
    except Exception:
        log.exception("compute_signal_score error")
        raise


# -------------------------
# Mode & leverage decision
# -------------------------
def decide_mode_and_leverage(score: float) -> Tuple[str, int]:
    """
    Return (mode, leverage)
    """
    if score >= 85:
        return "TREND", 50
    if score >= 70:
        return "MID", 25
    if score >= 45:
        return "QUICK", 10
    return "NO_TRADE", 1


# -------------------------
# TP / SL calculation
# -------------------------
def compute_tp_sl(df: 'pd.DataFrame', mode: str, score: float) -> Dict[str, Any]:
    last = float(df['close'].iloc[-1])
    atr_val = float(atr(df, 14).iloc[-1] or 0.0)
    vol = float(volatility_std(df['close'], 20).iloc[-1] or 0.0)

    if mode == "TREND":
        tp_pct = 0.008 + (vol * 0.6) + (score / 1000.0)
        sl_pct = 0.004 + (atr_val / (last + 1e-10))
    elif mode == "MID":
        tp_pct = 0.005 + (vol * 0.45) + (score / 1500.0)
        sl_pct = 0.003 + (atr_val / (last + 1e-10))
    elif mode == "QUICK":
        tp_pct = 0.0025 + (vol * 0.3) + (score / 3000.0)
        sl_pct = 0.0012 + (atr_val / (last + 1e-10))
    else:
        return {"entry": last, "tp": None, "sl": None, "leverage": 1, "mode": mode}

    tp = last * (1 + tp_pct)
    sl = last * (1 - sl_pct)

    tick = round_tick(last)
    tp = round(tp / tick) * tick
    sl = round(sl / tick) * tick

    _, leverage = decide_mode_and_leverage(score)
    return {"entry": last, "tp": float(tp), "sl": float(sl), "leverage": leverage, "mode": mode}


# -------------------------
# Danger zone helper (ATR-based)
# -------------------------
def compute_danger_zone(df: 'pd.DataFrame', mult: float = 1.5) -> Optional[str]:
    try:
        last = float(df['close'].iloc[-1])
        atr_val = float(atr(df, 14).iloc[-1] or 0.0)
        low = last - mult * atr_val
        high = last + mult * atr_val
        return f"{round(low, 8)} — {round(high, 8)} (ATR={atr_val:.8f})"
    except Exception:
        return None


# -------------------------
# Telegram: format & send
# -------------------------
def format_telegram_message(payload: Dict[str, Any]) -> str:
    symbol = payload.get("symbol")
    entry = payload.get("entry")
    tp = payload.get("tp")
    sl = payload.get("sl")
    mode = payload.get("mode")
    leverage = payload.get("leverage")
    score = payload.get("score")
    reasons = payload.get("reasons", [])
    danger = payload.get("danger_zone")

    lines = []
    lines.append(f"🔥 *{mode} SIGNAL* — *{symbol}/USDT*")
    lines.append(f"*Entry:* `{entry}`")
    lines.append(f"*TP:* `{tp}`    *SL:* `{sl}`")
    lines.append(f"*Leverage:* `{leverage}x`")
    lines.append(f"*Score:* `{score:.1f}`")
    if reasons:
        lines.append(f"*Reason:* {', '.join(reasons)}")
    if danger:
        lines.append(f"*Danger Zone:* `{danger}`")
    # quick code block with TP/SL for copy
    lines.append("\n```")
    lines.append(f"TP:{tp}\nSL:{sl}")
    lines.append("```")
    return "\n".join(lines)

def send_telegram_payload(payload: Dict[str, Any], token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    text = format_telegram_message(payload)
    tkn = token or TELEGRAM_TOKEN
    cid = chat_id or TELEGRAM_CHAT_ID
    if not tkn or not cid:
        log.warning("Telegram token/chat_id not set. Skipping send.")
        return False
    url = f"https://api.telegram.org/bot{tkn}/sendMessage"
    body = {"chat_id": cid, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=body, timeout=10)
        if r.status_code == 200:
            log.info("Telegram sent for %s", payload.get("symbol"))
            return True
        else:
            log.error("Telegram error %s %s", r.status_code, r.text)
            return False
    except Exception:
        log.exception("Telegram post failed")
        return False


# -------------------------
# High-level evaluate + emit
# -------------------------
def evaluate_symbol_and_emit(exchange: 'ccxt.Exchange', symbol: str, timeframe: str = "1m", limit: int = 300, pair_suffix: str = "/USDT", send_telegram_flag: bool = True) -> Optional[Dict[str, Any]]:
    """
    Evaluate one symbol: compute indicators, score, decide mode & tp/sl, check cooldowns, emit telegram, set cooldown.
    Returns payload dict if emitted, else None.
    """
    try:
        df = get_ohlcv_sample(exchange, symbol, timeframe=timeframe, limit=limit, pair_suffix=pair_suffix)
    except Exception as e:
        log.error("OHLCV fetch failed for %s: %s", symbol, e)
        return None

    try:
        info = compute_signal_score(df)
    except Exception:
        log.exception("Scoring failed for %s", symbol)
        return None

    score = info.get("score", 0.0)
    mode, _ = decide_mode_and_leverage(score)
    if mode == "NO_TRADE":
        log.debug("No trade for %s (score %.2f)", symbol, score)
        return None

    if is_in_cooldown(symbol, mode):
        log.debug("Cooldown active %s mode %s", symbol, mode)
        return None

    tp_sl = compute_tp_sl(df, mode, score)
    danger = compute_danger_zone(df)

    payload = {
        "symbol": symbol,
        "entry": tp_sl["entry"],
        "tp": tp_sl["tp"],
        "sl": tp_sl["sl"],
        "mode": mode,
        "leverage": tp_sl["leverage"],
        "score": score,
        "reasons": info.get("reasons", []),
        "indicators": info,
        "danger_zone": danger,
        "timestamp": time.time()
    }

    # send telegram (if configured)
    telegram_ok = False
    if send_telegram_flag:
        telegram_ok = send_telegram_payload(payload)
    # persist cooldown & history even on dry-run if you want — here we set only if sent
    if telegram_ok:
        set_cooldown(symbol, mode)
    else:
        # still set soft cooldown to avoid spam if not sent
        # set shorter soft cooldown (e.g., 1/3 of mode cd)
        soft = max(60, int(COOLDOWNS.get(mode, 300) / 3))
        data = _load_cooldown_storage()
        now = time.time()
        data.setdefault("last_signal", {})[symbol] = now
        data.setdefault("blocks", {})[symbol] = now + soft
        _save_cooldown_storage(data)
    payload["telegram_ok"] = telegram_ok
    log.info("Evaluate result for %s: mode=%s score=%.2f sent=%s", symbol, mode, score, telegram_ok)
    return payload


def run_scan_and_emit(exchange: 'ccxt.Exchange', symbols: List[str], timeframe: str = "1m", limit: int = 300, pair_suffix: str = "/USDT", send_telegram_flag: bool = True, max_emit: int = 3) -> List[Dict[str, Any]]:
    """
    Scan list of symbols and emit signals (respecting cooldowns). Returns list of payloads emitted or attempted.
    max_emit limits how many signals will be emitted in one run (safety).
    """
    emitted = []
    for s in symbols:
        if len(emitted) >= max_emit:
            log.info("Reached max_emit=%d, stopping scan", max_emit)
            break
        try:
            res = evaluate_symbol_and_emit(exchange, s, timeframe=timeframe, limit=limit, pair_suffix=pair_suffix, send_telegram_flag=send_telegram_flag)
            if res:
                emitted.append(res)
                # small safety sleep between emits
                time.sleep(0.5)
        except Exception:
            log.exception("run_scan_and_emit error for %s", s)
    return emitted


# -------------------------
# Quick self-test runner (dry-run)
# -------------------------
if __name__ == "__main__":
    log.info("helpers PART2 quick dry-run test start")
    try:
        ex = get_exchange()
        coins = load_coins()
        sample = coins[:8]
        out = run_scan_and_emit(ex, sample, send_telegram_flag=False, max_emit=5)
        log.info("Dry-run emitted (count) %d", len(out))
    except Exception:
        log.exception("helpers PART2 quick test failed")