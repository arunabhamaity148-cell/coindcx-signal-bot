# ============================
# helpers.py — FINAL CLEAN v3
# ArunBot PRO (Redis-free)
# ============================

from __future__ import annotations
import os, time, json, math, logging, requests
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
try:
    import ccxt
except Exception:
    ccxt = None


# --------------------------
# ENV + CONFIG
# --------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1","true","yes")

EXCHANGE_NAME  = os.getenv("EXCHANGE_NAME", "binance")
QUOTE_ASSET    = os.getenv("QUOTE_ASSET", "USDT")

SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "1"))

# Score thresholds
MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "85"))
THRESH_QUICK = float(os.getenv("THRESH_QUICK", "92"))
THRESH_MID   = float(os.getenv("THRESH_MID", "85"))
THRESH_TREND = float(os.getenv("THRESH_TREND", "72"))

# Filters
MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", "250000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.004"))   # 0.4%

# Cooldowns
COOLDOWN_JSON = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")
TTL_QUICK = int(os.getenv("COOLDOWN_QUICK_S", "1800"))
TTL_MID   = int(os.getenv("COOLDOWN_MID_S", "1800"))
TTL_TREND = int(os.getenv("COOLDOWN_TREND_S", "3600"))

# Indicators
EMA_FAST = 20
EMA_SLOW = 50
EMA_LONG = 200


# --------------------------
# LOGGER
# --------------------------
LOG = logging.getLogger("helpers")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(os.getenv("HELPERS_LOG_LEVEL", "INFO"))


# --------------------------
# Cooldown Manager (LOCAL JSON)
# --------------------------
class CooldownManager:
    def __init__(self):
        self._local_map: Dict[str, int] = {}
        self._load_local()
        LOG.info("CooldownManager: using local JSON fallback (%s)", COOLDOWN_JSON)

    def _load_local(self):
        try:
            with open(COOLDOWN_JSON, "r") as f:
                data = json.load(f)
            self._local_map = {k: int(v) for k, v in data.items()}
        except Exception:
            self._local_map = {}

    def _save_local(self):
        try:
            with open(COOLDOWN_JSON + ".tmp", "w") as f:
                json.dump(self._local_map, f)
            os.replace(COOLDOWN_JSON + ".tmp", COOLDOWN_JSON)
        except Exception as e:
            LOG.warning("CooldownManager save failed: %s", e)

    def _ttl_for_mode(self, mode: str) -> int:
        m = mode.lower()
        if m == "quick": return TTL_QUICK
        if m == "mid":   return TTL_MID
        return TTL_TREND

    def set_cooldown(self, key: str, mode: str) -> bool:
        ttl = self._ttl_for_mode(mode)
        now = int(time.time())
        expires = self._local_map.get(key, 0)
        if expires <= now:
            self._local_map[key] = now + ttl
            self._save_local()
            return True
        return False

    def is_cooled(self, key: str) -> bool:
        now = int(time.time())
        exp = self._local_map.get(key, 0)
        if exp <= now:
            if key in self._local_map:
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
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    if df.empty:
        return None
    return df

def fetch_ohlcv_sync(ex: "ccxt.Exchange", symbol: str, timeframe: str="1m", limit: int=200, retries:int=3):
    symbol = normalize_symbol(symbol)
    last_exc = None
    for attempt in range(retries):
        try:
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            time.sleep(0.25)
            return df_from_ohlcv(data)
        except Exception as e:
            last_exc = e
            time.sleep(0.35 * (attempt+1))
    LOG.debug("fetch_ohlcv failed for %s: %s", symbol, last_exc)
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
    high = df["high"]; low = df["low"]; close = df["close"]
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean().bfill()

# -----------------------------
# Orderbook / Spread / Liquidity
# -----------------------------
def fetch_orderbook_safe(ex: "ccxt.Exchange", symbol: str, limit: int = 50):
    try:
        ob = ex.fetch_order_book(normalize_symbol(symbol), limit=limit)
        time.sleep(0.15)
        return ob or {}
    except Exception:
        return {}

def orderbook_imbalance_from_ob(ob: dict, depth_levels:int=12) -> float:
    bids = ob.get("bids", []); asks = ob.get("asks", [])
    bid_vol = sum(float(x[1]) for x in bids[:depth_levels]) if bids else 0.0
    ask_vol = sum(float(x[1]) for x in asks[:depth_levels]) if asks else 0.0
    total = bid_vol + ask_vol + 1e-12
    return (bid_vol - ask_vol) / total

def liquidity_score(ob: dict) -> float:
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    top = (sum(float(x[1]) for x in bids[:5]) + sum(float(x[1]) for x in asks[:5])) / 2.0 if (bids or asks) else 0.0
    deeper = (sum(float(x[1]) for x in bids[:20]) + sum(float(x[1]) for x in asks[:20])) / 2.0 if (bids or asks) else 1.0
    return round(min(1.0, top / (deeper + 1e-12)), 3)

def calc_spread_from_ob(ob: dict) -> float:
    try:
        bid = float(ob["bids"][0][0])
        ask = float(ob["asks"][0][0])
        return abs(ask - bid)/ask
    except Exception:
        return 999
# -----------------------------
# SUPPORT / RESISTANCE (Fake Signal Blocker)
# -----------------------------
def find_sr_levels(df: pd.DataFrame, lookback: int = 60, sensitivity: float = 0.003):
    """
    খুব simple কিন্তু powerful S/R detection:
    - recent highs → resistance
    - recent lows → support
    """
    if df is None or df.empty:
        return None, None

    closes = df['close'].iloc[-lookback:]
    highs = df['high'].iloc[-lookback:]
    lows  = df['low'].iloc[-lookback:]

    resistance = highs.max()
    support    = lows.min()

    return float(support), float(resistance)


def sr_filter(entry: float, support: float, resistance: float) -> bool:
    """
    Fake breakout avoid:
    - Entry যদি resistance-এর খুব কাছে → BUY avoid
    - Entry যদি support-এর খুব কাছে → SELL avoid
    """
    if support is None or resistance is None:
        return True   # allow

    zone = (resistance - support) * 0.06  # 6% of range

    near_res = abs(entry - resistance) < zone
    near_sup = abs(entry - support) < zone

    # BUY হলে resistance-এর কাছে হওয়া যাবে না
    # SELL হলে support-এর কাছে হওয়া যাবে না
    return not (near_res or near_sup)


# -----------------------------
# GAINERS / LIQUIDITY FILTER
# -----------------------------
def is_top_volume_or_gainer(symbol: str, ex: "ccxt.Exchange", top_n: int = 25) -> bool:
    """
    এই ফিল্টার ensure করে high-quality coin আগে scan হয়।
    Low volume coin direct skip হবে।
    """
    try:
        tickers = ex.fetch_tickers()
        # শুধু USDT pair
        usdt_pairs = {k: v for k, v in tickers.items() if k.endswith("/USDT")}
        
        # volume অনুযায়ী সাজাও
        sorted_vol = sorted(usdt_pairs.items(), key=lambda x: x[1].get("quoteVolume", 0), reverse=True)
        top_list = [p[0] for p in sorted_vol[:top_n]]

        return symbol in top_list
    except Exception:
        return True  # fallback allow


# -----------------------------
# MAIN SCORING SYSTEM
# -----------------------------
def compute_score_and_reasons(df_1m, df_5m, ob, spread) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    if df_1m is None or df_1m.empty:
        return 0.0, ["insufficient_data"]

    close1 = df_1m['close']
    score = 40.0  # base

    # ---------- 1. MTF EMA ----------
    e20_1 = ema(close1, 20).iloc[-1]
    e50_1 = ema(close1, 50).iloc[-1]
    e20_5 = ema(df_5m['close'], 20).iloc[-1] if df_5m is not None else e20_1
    e50_5 = ema(df_5m['close'], 50).iloc[-1] if df_5m is not None else e50_1

    if e20_1 > e50_1:
        score += 10; reasons.append("EMA20>50_1m")
    else:
        reasons.append("EMA20<=50_1m")

    if e20_5 > e50_5:
        score += 8; reasons.append("EMA20>50_5m")
    else:
        reasons.append("EMA20<=50_5m")

    # ---------- 2. MACD ----------
    macd_val = (ema(close1, 12) - ema(close1, 26)).iloc[-1]
    if macd_val > 0:
        score += 10; reasons.append("MACD_pos")
    else:
        reasons.append("MACD_neg")

    # ---------- 3. RSI ----------
    r = float(rsi(close1).iloc[-1])
    if 40 < r < 70:
        score += 6; reasons.append("RSI_ok")
    elif r <= 40:
        score += 2; reasons.append("RSI_low")
    else:
        score += 1; reasons.append("RSI_high")

    # ---------- 4. Volume Spike ----------
    vol = df_1m['volume']
    vol_avg = vol.rolling(20).mean().iloc[-1] or 1
    if vol.iloc[-1] > vol_avg * 1.6:
        score += 12; reasons.append("Vol_spike")
    else:
        reasons.append("Vol_ok")

    # ---------- 5. Orderbook Imbalance ----------
    ob_imb = orderbook_imbalance_from_ob(ob)
    if ob_imb > 0.55:
        score += 6; reasons.append("OB_buy_pressure")
    elif ob_imb < -0.55:
        score += 6; reasons.append("OB_sell_pressure")

    # ---------- 6. Liquidity ----------
    liq = liquidity_score(ob)
    if liq > 0.55:
        score += 8; reasons.append("Liquidity_ok")
    else:
        reasons.append("Liquidity_low")

    # ---------- 7. Spread ----------
    if spread <= MAX_SPREAD_PCT:
        score += 6; reasons.append("Spread_ok")
    else:
        reasons.append("Spread_high")

    score = min(100, max(0, score))
    return round(score, 1), reasons


# -----------------------------
# SIGNAL BUILDER (BUY + SELL)
# -----------------------------
def build_signal_from_df(symbol, df_1m, df_5m, ob, score, reasons):
    last = df_1m.iloc[-1]
    entry = float(last['close'])
    atr_val = float(atr(df_1m).iloc[-1])

    e20 = ema(df_1m['close'], 20).iloc[-1]
    e50 = ema(df_1m['close'], 50).iloc[-1]
    ob_imb = orderbook_imbalance_from_ob(ob)

    direction = "BUY"
    if e20 < e50 or ob_imb < -0.45:
        direction = "SELL"

    # classification
    if score >= THRESH_QUICK: mode = "QUICK"
    elif score >= THRESH_MID: mode = "MID"
    else: mode = "TREND"

    # SL/TP dynamic
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

    lev = "25x" if mode=="MID" else ("50x" if mode=="QUICK" else "20x")

    # S/R
    support, resistance = find_sr_levels(df_1m)
    danger = (round(support or 0, 6), round(resistance or 0, 6))

    return {
        "pair": normalize_symbol(symbol),
        "mode": mode,
        "score": score,
        "entry": round(entry,6),
        "tp": round(tp,6),
        "sl": round(sl,6),
        "atr": round(atr_val,6),
        "direction": direction,
        "reasons": reasons,
        "danger": danger,
        "lev_suggest": lev,
        "ts": int(time.time())
    }
# -----------------------------
# FORMATTER (Emoji + HTML)
# -----------------------------
def format_signal_message(sig: Dict[str,Any]) -> str:
    em = "🔥" if sig["mode"] == "MID" else ("⚡" if sig["mode"] == "QUICK" else "🚀")
    dir_emoji = "⬆️" if sig["direction"] == "BUY" else "⬇️"

    support, resistance = sig.get("danger", (0, 0))
    reason = ", ".join(sig.get("reasons", []))

    return (
        f"{em} <b>{dir_emoji} {sig['direction']} SIGNAL — {sig['mode']}</b>\n"
        f"Pair: <b>{sig['pair']}</b>\n"
        f"Entry: <code>{sig['entry']}</code>\n"
        f"TP: <code>{sig['tp']}</code>   SL: <code>{sig['sl']}</code>\n"
        f"Leverage: <b>{sig['lev_suggest']}</b>\n"
        f"Score: <b>{sig['score']}</b>\n"
        f"ATR: {sig['atr']}\n"
        f"⚠️ S/R Zone: <code>{support}</code> - <code>{resistance}</code>\n"
        f"Reason: {reason}\n"
        f"Time: {pd.to_datetime(sig['ts'], unit='s').strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )


# -----------------------------
# TELEGRAM SENDER
# -----------------------------
def send_telegram_message(msg: str) -> bool:
    if NOTIFY_ONLY or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        LOG.info("Telegram preview (NOT SENT):\n%s", msg)
        return True

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

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
# FINAL analyze_coin() WRAPPER
# -----------------------------
def analyze_coin(ex: "ccxt.Exchange", symbol: str) -> Optional[Dict[str,Any]]:
    try:
        symbol = normalize_symbol(symbol)

        # ----------------- 1) Top Volume / Gainers Filter -----------------
        if not is_top_volume_or_gainer(symbol, ex, top_n=25):
            LOG.debug("%s skip: not in top volume/gainers list", symbol)
            return None

        # ----------------- 2) Fetch OHLCV -----------------
        df1 = fetch_ohlcv_sync(ex, symbol, timeframe="1m", limit=200)
        df5 = fetch_ohlcv_sync(ex, symbol, timeframe="5m", limit=200)

        if df1 is None or df1.empty:
            return None

        # ----------------- 3) Volume Filter -----------------
        last_vol = float(df1["volume"].iloc[-1])
        if last_vol < MIN_24H_VOLUME:
            LOG.debug("%s skip: low volume %.1f", symbol, last_vol)
            return None

        # ----------------- 4) Orderbook + Spread -----------------
        ob = fetch_orderbook_safe(ex, symbol, limit=50)
        spread = calc_spread_from_ob(ob)
        if spread > MAX_SPREAD_PCT:
            LOG.debug("%s skip: spread %.6f > %.6f", symbol, spread, MAX_SPREAD_PCT)
            return None

        # ----------------- 5) Scoring -----------------
        score, reasons = compute_score_and_reasons(df1, df5, ob, spread)
        if score < MIN_SIGNAL_SCORE:
            LOG.debug("%s skip: score %.1f < %.1f", symbol, score, MIN_SIGNAL_SCORE)
            return None

        # ----------------- 6) Build Signal -----------------
        sig = build_signal_from_df(symbol, df1, df5, ob, score, reasons)

        # ----------------- 7) S/R FILTER -----------------
        support, resistance = find_sr_levels(df1)
        sig["danger"] = (support, resistance)

        if not sr_filter(sig["entry"], support, resistance):
            LOG.debug("%s skip: S/R fake zone blocked", symbol)
            return None

        # ----------------- 8) Cooldown Check -----------------
        key = cooldown_key_for(sig["pair"], sig["mode"])
        if _cd_mgr.is_cooled(key):
            LOG.debug("%s skip: cooldown active for mode=%s", sig['pair'], sig["mode"])
            return None

        ok = _cd_mgr.set_cooldown(key, sig["mode"])
        if not ok:
            LOG.debug("%s skip: cooldown refusal", symbol)
            return None

        return sig

    except Exception as e:
        LOG.exception("analyze_coin error %s: %s", symbol, e)
        return None