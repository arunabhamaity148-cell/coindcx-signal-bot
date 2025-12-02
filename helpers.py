# helpers.py — FINAL (stable, defensive, fallback scoring)
# Drop this file into the project. Exports:
# now_ts, human_time, esc, calc_tp_sl, ensemble_score, CACHE,
# compute_ema_from_closes, atr, rsi_from_closes
import os
import time
import json
import html
import hashlib
import random
import asyncio
from typing import Dict, List, Optional

# -------------------------
# Utils / time / small help
# -------------------------
def now_ts() -> int:
    return int(time.time())

def human_time(ts: int = None) -> str:
    if ts is None:
        ts = now_ts()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def esc(x) -> str:
    return html.escape("" if x is None else str(x))

# -------------------------
# TP/SL rules (mode-based)
# -------------------------
TP_SL_RULES = {
    "quick": {"tp_pct": 1.6, "sl_pct": 1.0},
    "mid":   {"tp_pct": 2.0, "sl_pct": 1.0},
    "trend": {"tp_pct": 4.0, "sl_pct": 1.5},
}

def calc_tp_sl(price: float, mode: str):
    cfg = TP_SL_RULES.get(mode, {"tp_pct": 2.0, "sl_pct": 1.0})
    tp = price * (1 + cfg["tp_pct"] / 100.0)
    sl = price * (1 - cfg["sl_pct"] / 100.0)
    return round(tp, 8), round(sl, 8)

# -------------------------
# Small TTL in-memory cache
# -------------------------
class SimpleCache:
    def __init__(self):
        self.store = {}  # key -> (expiry, value)

    def get(self, key):
        rec = self.store.get(key)
        if not rec:
            return None
        exp, val = rec
        if exp < time.time():
            try:
                del self.store[key]
            except KeyError:
                pass
            return None
        return val

    def set(self, key, value, ttl_seconds: int):
        self.store[key] = (time.time() + ttl_seconds, value)

    def make_key(self, *parts) -> str:
        raw = "|".join([str(p) for p in parts])
        return hashlib.sha256(raw.encode()).hexdigest()

CACHE = SimpleCache()

# -------------------------
# Indicators (lightweight)
# -------------------------
def sma(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    if len(values) < period:
        return sum(values) / len(values)
    return sum(values[-period:]) / period

def compute_ema_from_closes(closes: List[float], period: int) -> float:
    if not closes:
        return 0.0
    if len(closes) < period:
        return sma(closes, period)
    seed = sum(closes[:period]) / period
    ema_val = seed
    k = 2 / (period + 1)
    for price in closes[period:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val

def rsi_from_closes(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        if ch > 0:
            gains.append(ch)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(ch))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(ohlc: List[List[float]], period: int = 14) -> float:
    if len(ohlc) < 2:
        return 0.0
    trs = []
    for i in range(1, len(ohlc)):
        high = ohlc[i][2]
        low = ohlc[i][3]
        prev_close = ohlc[i-1][4]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        return 0.0
    if len(trs) < period:
        return sum(trs) / len(trs)
    seed = sum(trs[:period]) / period
    atr_val = seed
    k = 2 / (period + 1)
    for tr in trs[period:]:
        atr_val = tr * k + atr_val * (1 - k)
    return atr_val

# -------------------------
# Logic descriptions (kept for prompts)
# -------------------------
LOGIC_DESC = {
    "HTF_EMA_1h_15m": "1h vs 15m EMA alignment and slope",
    "HTF_Structure_Break": "Breakout of HTF support/resistance",
    "Imbalance_FVG": "Fair-value gap / imbalance near price",
    "Vol_Sweep_1m": "1m volume spike vs baseline",
    "ATR_Expansion": "ATR expansion confirms breakout",
    "Price_Compression": "Tight range pre-breakout",
    "Final_Signal_Score": "Final combined AI decision/confidence"
}
def logic_descriptions_text() -> str:
    return "\n".join([f"{k}: {v}" for k, v in LOGIC_DESC.items()])

# -------------------------
# Simple scoring heuristics (fallback)
# -------------------------
def heuristic_score_from_metrics(symbol: str, price: float, metrics: Dict) -> Dict:
    """
    Return a fallback score dict when OpenAI fails.
    Uses HTF EMA alignment, rsi, spread, volume heuristics.
    """
    score = 0
    reason_parts = []

    price_val = metrics.get("price", price)
    ema1h = metrics.get("ema_1h_50", 0)
    ema15 = metrics.get("ema_15m_50", 0)
    rsi1m = metrics.get("rsi_1m", 50)
    spread = metrics.get("spread_pct", 999)
    vol = metrics.get("vol_1m", 0)

    # EMA alignment
    if price_val and ema15 and ema1h:
        if price_val > ema15 > ema1h:
            score += 30
            reason_parts.append("HTF+ microEMA bullish")
        elif price_val < ema15 < ema1h:
            score += 30
            reason_parts.append("HTF+ microEMA bearish")
        else:
            reason_parts.append("no EMA alignment")

    # RSI bias
    if rsi1m > 65:
        score += 10
        reason_parts.append("RSI high")
    elif rsi1m < 35:
        score += 10
        reason_parts.append("RSI low")
    else:
        score += 5

    # Spread and volume
    if spread < 0.1:
        score += 10
    else:
        reason_parts.append("spread wide")

    if vol and vol > 1000:
        score += 10

    # ATR normalization + small random (stable)
    try:
        atr_val = metrics.get("atr_1h", 0) or metrics.get("atr", 0)
        if atr_val and atr_val > 0:
            score += min(10, int(10 * (1 / (1 + atr_val))))
    except Exception:
        pass

    # normalize and clamp
    score = max(0, min(100, int(round(score + random.uniform(-2, 2)))))

    # decide mode
    if score >= 85:
        mode = "trend"
    elif score >= 60:
        mode = "mid"
    else:
        mode = "quick"

    reason = " | ".join(reason_parts[:3]) or "heuristic"
    return {"score": score, "mode": mode, "reason": reason}

# -------------------------
# OpenAI sync caller (defensive)
# -------------------------
def _call_openai_sync(prompt: str, model: Optional[str] = None, max_tokens: int = 300) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.responses.create(model=model_name, input=prompt, temperature=0, max_output_tokens=max_tokens)
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
        # fallback: try to convert to dict string
        try:
            return json.dumps(resp.to_dict())
        except Exception:
            return str(resp)
    except Exception:
        return ""

# -------------------------
# Single-GPT scoring with caching + fallback
# -------------------------
async def ai_score_symbol_with_memory_and_tools(symbol: str, price: float, metrics: Dict, prefs: Dict, ttl: int = 120) -> Optional[Dict]:
    fingerprint = f"{symbol}|{round(price,6)}|{round(metrics.get('rsi_1m',50),2)}|{round(metrics.get('spread_pct',0),4)}"
    key = CACHE.make_key("ai", fingerprint)
    cached = CACHE.get(key)
    if cached:
        return cached

    # Minimal strict filters (keep relaxed to reduce false-0)
    spread = metrics.get("spread_pct", 999)
    if spread is None:
        spread = 999

    # Build prompt but keep small arrays
    small_metrics = {k: (v[-20:] if isinstance(v, list) else v) for k, v in metrics.items()}
    metrics_text = json.dumps(small_metrics, default=str)
    prefs_text = json.dumps(prefs or {}, default=str)
    logic_text = logic_descriptions_text()

    prompt = (
        "You are a concise expert crypto futures signal scorer. Return SINGLE-LINE JSON only:\n"
        '{"score":int,"mode":"quick|mid|trend","reason":"max 8 words"}\n\n'
        f"Symbol: {symbol}\nPrice: {price}\nMetrics: {metrics_text}\nPreferences: {prefs_text}\n\n"
        f"Logic short: {logic_text}\n\nRules: be concise, exact JSON only."
    )

    # Try OpenAI (sync call in thread)
    try:
        raw = await asyncio.to_thread(_call_openai_sync, prompt, os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 250)
        parsed = None
        if raw:
            try:
                parsed = json.loads(raw.strip())
            except Exception:
                # extract first JSON object if model prepended text
                import re
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = None
        if parsed and isinstance(parsed, dict) and {"score", "mode", "reason"} <= set(parsed.keys()):
            # sanitize
            out = {
                "score": int(parsed.get("score", 0)),
                "mode": str(parsed.get("mode", "quick")),
                "reason": str(parsed.get("reason", ""))[:120]
            }
            # clamp
            out["score"] = max(0, min(100, out["score"]))
            CACHE.set(key, out, ttl_seconds=ttl)
            return out
    except Exception:
        # fall through to fallback
        pass

    # fallback heuristic
    fallback = heuristic_score_from_metrics(symbol, price, metrics)
    CACHE.set(key, fallback, ttl_seconds=ttl)
    return fallback

# -------------------------
# Ensemble scoring (calls the single scorer n times)
# -------------------------
async def ensemble_score(symbol: str, price: float, metrics: Dict, prefs: Dict, n: int = 3) -> Dict:
    scores = []
    for _ in range(max(1, n)):
        sc = await ai_score_symbol_with_memory_and_tools(symbol, price, metrics, prefs)
        if sc:
            scores.append(sc)
    if not scores:
        return {"score": 0, "mode": "quick", "reason": "ensemble_fail"}
    score_vals = [s["score"] for s in scores]
    final_score = int(round(sum(score_vals) / len(score_vals)))
    modes = [s["mode"] for s in scores]
    # median-like choose
    final_mode = sorted(modes)[len(modes)//2] if modes else "quick"
    reason = scores[0].get("reason", "ok") if scores else "ok"
    return {"score": final_score, "mode": final_mode, "reason": reason}

# -------------------------
# End
# -------------------------