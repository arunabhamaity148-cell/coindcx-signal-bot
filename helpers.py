# helpers.py — FINAL CLEAN (Redis-free, MTF, liquidity, buy/sell, emoji messages)
from __future__ import annotations
import os, time, json, math, logging, requests
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
try:
    import ccxt
except Exception:
    ccxt = None

# env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1","true","yes")
EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "binance")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")

SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "1"))

MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "85"))
THRESH_QUICK = float(os.getenv("THRESH_QUICK", "92"))
THRESH_MID = float(os.getenv("THRESH_MID", "85"))
THRESH_TREND = float(os.getenv("THRESH_TREND", "72"))

MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", "250000"))  # on quote currency units
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.004"))  # 0.4%

COOLDOWN_JSON = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")

# TTLs (seconds)
TTL_QUICK = int(os.getenv("COOLDOWN_QUICK_S", "1800"))   # 30m
TTL_MID   = int(os.getenv("COOLDOWN_MID_S", "1800"))     # 30m (user wanted 30m same-coin)
TTL_TREND = int(os.getenv("COOLDOWN_TREND_S", "3600"))   # 60m

# indicators
EMA_FAST = 20
EMA_SLOW = 50
EMA_LONG = 200

# logging
LOG = logging.getLogger("helpers")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(os.getenv("HELPERS_LOG_LEVEL", "INFO"))

# -----------------------------
# Cooldown (local JSON only)
# -----------------------------
class CooldownManager:
    def __init__(self):
        self._local_map: Dict[str,int] = {}
        self._load_local()
        LOG.info("CooldownManager: using local JSON fallback (%s)", COOLDOWN_JSON)

    def _load_local(self):
        try:
            with open(COOLDOWN_JSON, "r") as f:
                data = json.load(f)
            self._local_map = {k:int(v) for k,v in data.items()}
        except Exception:
            self._local_map = {}

    def _save_local(self):
        try:
            with open(COOLDOWN_JSON + ".tmp","w") as f:
                json.dump(self._local_map, f)
            os.replace(COOLDOWN_JSON + ".tmp", COOLDOWN_JSON)
        except Exception as e:
            LOG.warning("CooldownManager: failed to save local json: %s", e)

    def _ttl_for_mode(self, mode: str) -> int:
        m = mode.lower()
        if m == "quick": return TTL_QUICK
        if m == "mid": return TTL_MID
        return TTL_TREND

    def set_cooldown(self, key: str, mode: str) -> bool:
        ttl = self._ttl_for_mode(mode)
        now = int(time.time())
        exp = self._local_map.get(key, 0)
        if exp <= now:
            self._local_map[key] = now + ttl
            self._save_local()
            return True
        return False

    def is_cooled(self, key: str) -> bool:
        now = int(time.time())
        exp = self._local_map.get(key)
        if not exp: return False
        if exp <= now:
            self._local_map.pop(key, None)
            self._save_local()
            return False
        return True

_cd_mgr = CooldownManager()

def cooldown_key_for(pair: str, mode: str) -> str:
    return f"{pair.upper()}::{mode.upper()}"

# -----------------------------
# Exchange helpers
# -----------------------------
def get_exchange(api_keys: bool = False) -> "ccxt.Exchange":
    if ccxt is None:
        raise RuntimeError("ccxt not installed")
    ex_name = EXCHANGE_NAME.lower()
    try:
        ex_cls = getattr(ccxt, ex_name, ccxt.binance)
        ex = ex_cls({'enableRateLimit': True})
    except Exception:
        ex = ccxt.binance({'enableRateLimit': True})
    try:
        ex.load_markets()
    except Exception:
        LOG.debug("exchange.load_markets failed/ignored")
    return ex

def normalize_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if "/" not in s:
        s = f"{s}/{QUOTE_ASSET}"
    return s

# -----------------------------
# OHLCV fetch + df helper
# -----------------------------
def df_from_ohlcv(ohlcv) -> Optional[pd.DataFrame]:
    if not ohlcv: return None
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    if df.empty: return None
    return df

def fetch_ohlcv_sync(ex: "ccxt.Exchange", symbol: str, timeframe: str="1m", limit: int=200, retries:int=3) -> Optional[pd.DataFrame]:
    s = normalize_symbol(symbol)
    last_exc = None
    for attempt in range(retries):
        try:
            data = ex.fetch_ohlcv(s, timeframe=timeframe, limit=limit)
            time.sleep(0.2)
            return df_from_ohlcv(data)
        except Exception as e:
            last_exc = e
            time.sleep(0.3 * (attempt+1))
    LOG.debug("fetch_ohlcv failed for %s: %s", s, last_exc)
    return None

# -----------------------------
# Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/length, adjust=False).mean()
    rs = up/(down+1e-12)
    return 100 - (100/(1+rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean().bfill()

# -----------------------------
# Orderbook & liquidity
# -----------------------------
def fetch_orderbook_safe(ex: "ccxt.Exchange", symbol: str, limit: int = 50) -> dict:
    try:
        ob = ex.fetch_order_book(normalize_symbol(symbol), limit=limit)
        time.sleep(0.15)
        return ob or {}
    except Exception:
        return {}

def orderbook_imbalance_from_ob(ob: dict, depth_levels:int=12) -> float:
    bids = ob.get('bids', []); asks = ob.get('asks', [])
    bid_vol = sum([float(x[1]) for x in bids[:depth_levels]]) if bids else 0.0
    ask_vol = sum([float(x[1]) for x in asks[:depth_levels]]) if asks else 0.0
    total = bid_vol + ask_vol + 1e-12
    return (bid_vol - ask_vol) / total

def liquidity_score(ob: dict) -> float:
    # simple liquidity score: (top 5 depth total size) / (avg of top 20 size)
    bids = ob.get('bids', [])
    asks = ob.get('asks', [])
    top = (sum([float(x[1]) for x in bids[:5]]) + sum([float(x[1]) for x in asks[:5]])) / 2.0 if (bids or asks) else 0.0
    deeper = (sum([float(x[1]) for x in bids[:20]]) + sum([float(x[1]) for x in asks[:20]])) / 2.0 if (bids or asks) else 1.0
    if deeper == 0: return 0.0
    score = min(1.0, top / (deeper + 1e-12))
    return round(score,3)

def calc_spread_from_ob(ob: dict) -> float:
    try:
        bid = float(ob['bids'][0][0])
        ask = float(ob['asks'][0][0])
        return abs(ask - bid)/ask
    except Exception:
        return 999.0

# -----------------------------
# Scoring (MTF + liquidity + volume + OBI + spread)
# -----------------------------
def compute_score_and_reasons(df_1m: pd.DataFrame, df_5m: pd.DataFrame, ob: dict, spread: float) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    if df_1m is None or df_1m.empty:
        return 0.0, ["insufficient_data"]
    close1 = df_1m['close']
    score = 40.0

    # MTF EMA alignment: 1m + 5m both confirm => stronger
    e20_1 = ema(close1, EMA_FAST).iloc[-1]
    e50_1 = ema(close1, EMA_SLOW).iloc[-1]
    e20_5 = ema(df_5m['close'], EMA_FAST).iloc[-1] if df_5m is not None else e20_1
    e50_5 = ema(df_5m['close'], EMA_SLOW).iloc[-1] if df_5m is not None else e50_1

    # EMA alignment scoring (MTF)
    if e20_1 > e50_1:
        score += 10; reasons.append("EMA20>50_1m")
    else:
        reasons.append("EMA20<=50_1m")
    if e20_5 > e50_5:
        score += 8; reasons.append("EMA20>50_5m")
    else:
        reasons.append("EMA20<=50_5m")

    # MACD-ish on 1m
    macd_val = (ema(close1,12) - ema(close1,26)).iloc[-1]
    if macd_val > 0:
        score += 10; reasons.append("MACD_pos")
    else:
        reasons.append("MACD_neg")

    # RSI on 1m
    r = float(rsi(close1).iloc[-1])
    if 40 < r < 70:
        score += 6; reasons.append("RSI_ok")
    elif r <= 40:
        score += 2; reasons.append("RSI_low")
    else:
        score += 1; reasons.append("RSI_high")

    # Volume spike (compare to 20-window on 1m)
    vol = df_1m['volume']; vol_avg = vol.rolling(20, min_periods=1).mean().iloc[-1] or 1.0
    if vol.iloc[-1] > vol_avg * 1.6:
        score += 12; reasons.append("Vol_spike")
    else:
        reasons.append("Vol_ok")

    # orderbook imbalance and liquidity
    ob_imb = orderbook_imbalance_from_ob(ob)
    if ob_imb > 0.55:
        score += 6; reasons.append("OB_buy_pressure")
    elif ob_imb < -0.55:
        score += 6; reasons.append("OB_sell_pressure")

    liq = liquidity_score(ob)
    if liq > 0.55:
        score += 8; reasons.append("Liquidity_ok")
    else:
        reasons.append("Liquidity_low")

    # spread
    if spread <= MAX_SPREAD_PCT:
        score += 6; reasons.append("Spread_ok")
    else:
        reasons.append("Spread_high")

    score = max(0.0, min(100.0, score))
    return round(score,1), reasons

# -----------------------------
# Signal builder (BUY + SELL, symmetric)
# -----------------------------
def build_signal_from_df(symbol: str, df_1m: pd.DataFrame, df_5m: pd.DataFrame, ob: dict, score: float, reasons: List[str]) -> Dict[str,Any]:
    last = df_1m.iloc[-1]
    entry = float(last['close'])
    atr_val = float(atr(df_1m).iloc[-1]) if not atr(df_1m).empty else 0.0

    # direction by EMA cross + orderbook imbalance
    e20 = ema(df_1m['close'], EMA_FAST).iloc[-1]
    e50 = ema(df_1m['close'], EMA_SLOW).iloc[-1]
    ob_imb = orderbook_imbalance_from_ob(ob)

    direction = "BUY"
    if e20 < e50 or ob_imb < -0.45:
        direction = "SELL"

    mode = "MID"
    if score >= THRESH_QUICK: mode = "QUICK"
    elif score >= THRESH_MID: mode = "MID"
    else: mode = "TREND"

    if mode == "QUICK":
        tp_off = atr_val * 1.5
        sl_off = atr_val * 1.0
    elif mode == "TREND":
        tp_off = atr_val * 3.0
        sl_off = atr_val * 2.0
    else:
        tp_off = atr_val * 2.0
        sl_off = atr_val * 1.4

    if direction == "BUY":
        tp = entry + tp_off
        sl = entry - sl_off
    else:
        tp = entry - tp_off
        sl = entry + sl_off

    # suggested leverage (only suggestion)
    lev = "25x" if mode=="MID" else ("50x" if mode=="QUICK" else "20x")

    danger = (round(entry - atr_val*1.1,8), round(entry + atr_val*1.1,8))
    return {
        "pair": normalize_symbol(symbol),
        "mode": mode,
        "score": score,
        "entry": round(entry,8),
        "tp": round(tp,8),
        "sl": round(sl,8),
        "atr": round(atr_val,8),
        "direction": direction,
        "reasons": reasons,
        "danger": danger,
        "lev_suggest": lev,
        "ts": int(time.time())
    }

# -----------------------------
# Formatter & Telegram sender
# -----------------------------
def format_signal_message(sig: Dict[str,Any]) -> str:
    em = "🔥" if sig['mode']=="MID" else ("⚡" if sig['mode']=="QUICK" else "🚀")
    dir_emoji = "⬆️" if sig['direction']=="BUY" else "⬇️"
    dz_low, dz_high = sig.get("danger", (0,0))
    reason = ", ".join(sig.get("reasons", []))
    return (
        f"{em} <b>{dir_emoji} {sig['direction']} SIGNAL — {sig['mode']}</b>\n"
        f"Pair: <b>{sig['pair']}</b>\n"
        f"Entry: <code>{sig['entry']}</code>\n"
        f"TP: <code>{sig['tp']}</code>   SL: <code>{sig['sl']}</code>\n"
        f"Leverage: <b>{sig['lev_suggest']}</b>\n"
        f"Score: <b>{sig['score']}</b>\n"
        f"ATR: {sig['atr']}\n"
        f"⚠️ Danger Zone: <code>{dz_low}</code> - <code>{dz_high}</code>\n"
        f"Reason: {reason}\n"
        f"Time: {pd.to_datetime(sig['ts'], unit='s').strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

def send_telegram_message(msg: str) -> bool:
    if NOTIFY_ONLY or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        LOG.info("Telegram preview (NOT SENT):\n%s", msg)
        return True
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        LOG.warning("Telegram send failed: %s %s", r.status_code, r.text)
        return False
    except Exception as e:
        LOG.warning("Telegram send exception: %s", e)
        return False

# -----------------------------
# Top-level analyzer
# -----------------------------
def analyze_coin(ex: "ccxt.Exchange", symbol: str) -> Optional[Dict[str,Any]]:
    try:
        norm = normalize_symbol(symbol)
        # quick market existence
        try:
            ex.load_markets()
            if norm not in ex.markets:
                LOG.debug("Symbol not in markets: %s", norm)
                return None
        except Exception:
            pass

        df1 = fetch_ohlcv_sync(ex, norm, timeframe="1m", limit=200)
        df5 = fetch_ohlcv_sync(ex, norm, timeframe="5m", limit=200)

        if df1 is None or df1.empty:
            return None

        # volume filter: use 1m last volume scaled to quote if exchange uses quote differently — assume raw
        last_vol = float(df1['volume'].iloc[-1]) if 'volume' in df1.columns and not df1['volume'].empty else 0.0
        if last_vol < MIN_24H_VOLUME:
            LOG.debug("%s skip: low vol %.1f", norm, last_vol); return None

        ob = fetch_orderbook_safe(ex, norm, limit=50)
        spread = calc_spread_from_ob(ob)
        if spread > MAX_SPREAD_PCT:
            LOG.debug("%s skip: spread %.6f > max %.6f", norm, spread, MAX_SPREAD_PCT); return None

        score, reasons = compute_score_and_reasons(df1, df5, ob, spread)
        if score < MIN_SIGNAL_SCORE:
            LOG.debug("%s skip: score %.1f < min %.1f", norm, score, MIN_SIGNAL_SCORE); return None

        sig = build_signal_from_df(norm, df1, df5, ob, score, reasons)
        return sig
    except Exception as e:
        LOG.exception("analyze_coin error %s: %s", symbol, e)
        return None