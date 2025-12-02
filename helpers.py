# helpers.py — STABLE FINAL
# - Provides indicators, caching, TP/SL, AI scoring (sync call wrapper), ensemble
# - Designed to work with the main.py you pasted (ensemble_score, calc_tp_sl, CACHE, indicators)
# - Minimal external deps (openai only if OPENAI_API_KEY present). Robust parsing & caching.

import os
import time
import json
import html
import hashlib
import random
import asyncio
from typing import Dict, List, Optional

# -------------------------
# utils
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
# TP / SL rules
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
# simple TTL cache (in-memory)
# -------------------------
class SimpleCache:
    def __init__(self):
        self.store: Dict[str, tuple] = {}
    def get(self, key: str):
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
    def set(self, key: str, value, ttl_seconds: int):
        self.store[key] = (time.time() + ttl_seconds, value)
    def make_key(self, *parts) -> str:
        raw = "|".join([str(p) for p in parts])
        return hashlib.sha256(raw.encode()).hexdigest()

CACHE = SimpleCache()

# -------------------------
# Indicators: sma, ema, rsi, atr
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
# logic descriptions (kept for prompt building)
# -------------------------
LOGIC_DESC = {
    "HTF_EMA_1h_15m": "1h vs 15m EMA alignment and slope",
    "HTF_EMA_1h_4h": "1h vs 4h EMA confluence",
    "HTF_Structure_Break": "Breakout of HTF support/resistance",
    "Imbalance_FVG": "Fair-value gap / imbalance near price",
    "Trend_Continuation": "Current trend likely to continue",
    "Trend_Exhaustion": "Momentum dying out at push extremes",
    "Micro_Pullback": "Small retracement into value for entry",
    "Wick_Strength": "Strong wick rejection candle present",
    "Sweep_Reversal": "Liquidity sweep then reversal",
    "Vol_Sweep_1m": "1m volume spike vs baseline",
    "Vol_Sweep_5m": "5m volume spike vs baseline",
    "Delta_Divergence_1m": "Trade-delta divergence on 1m",
    "Orderbook_Wall_Shift": "Large wall shift in orderbook",
    "Liquidity_Wall": "Liquidity wall near price",
    "ADR_DayRange": "Price vs daily range (ADR)",
    "ATR_Expansion": "ATR expansion confirms breakout",
    "Price_Compression": "Tight range pre-breakout",
    "Taker_Pressure": "Taker dominance in trades",
    "Partial_Exit_Logic": "Partial take-profit logic",
    "BTC_Risk_Filter_L1": "BTC calm / volatility filter L1",
    "Final_Signal_Score": "Final combined AI decision/confidence"
}

def logic_descriptions_text() -> str:
    return "\n".join([f"{k}: {v}" for k, v in LOGIC_DESC.items()])

# -------------------------
# relaxed filters (default)
# -------------------------
def spread_filter(metrics: Dict) -> bool:
    # allow slightly wider spread (percent)
    return metrics.get("spread_pct", 999) < 0.12

def htf_alignment_score(metrics: Dict) -> int:
    ema1h = metrics.get("ema_1h_50", 0)
    ema15 = metrics.get("ema_15m_50", 0)
    price = metrics.get("price", 0)
    if price and ema15 and ema1h:
        if price > ema15 > ema1h:
            return 20
        if price < ema15 < ema1h:
            return 20
    return 0

# -------------------------
# memory & feedback helpers
# -------------------------
def get_memory(symbol: str) -> List[dict]:
    return CACHE.get(f"memory_{symbol}") or []

def append_memory(symbol: str, entry: dict, max_items: int = 5):
    mem = get_memory(symbol)
    mem.append(entry)
    mem = mem[-max_items:]
    CACHE.set(f"memory_{symbol}", mem, ttl_seconds=86400)

def get_trade_pnl(symbol: str, tp: float, sl: float, mode: str) -> float:
    # placeholder: return small random feedback to help prompt context
    return random.uniform(-1.0, 1.0)

# -------------------------
# OpenAI sync caller (safe)
# -------------------------
def _call_openai_sync(prompt: str, model: Optional[str] = None, max_tokens: int = 300) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.responses.create(
            model=model_name,
            input=prompt,
            temperature=0,
            max_output_tokens=max_tokens
        )
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
        # fallback to dict serialization
        try:
            return json.dumps(resp.to_dict())
        except Exception:
            return str(resp)
    except Exception:
        # OpenAI not available or failed — fallback to a simple heuristic scorer
        return ""

# -------------------------
# single-call scorer (relaxed, robust parse)
# returns dict {"score":int,"mode":str,"reason":str} or None
# -------------------------
async def ai_score_symbol_with_memory_and_tools(symbol: str, price: float, metrics: Dict, prefs: Dict, ttl: int = 120) -> Optional[Dict]:
    fingerprint = f"{symbol}|{round(price,6)}|{round(metrics.get('rsi_1m',50),2)}|{round(metrics.get('spread_pct',0),4)}"
    key = CACHE.make_key("ai", fingerprint)
    cached = CACHE.get(key)
    if cached:
        return cached

    # base from HTF alignment
    base = htf_alignment_score(metrics)

    memory = get_memory(symbol)
    pnl = get_trade_pnl(symbol, calc_tp_sl(price, "mid")[0], calc_tp_sl(price, "mid")[1], "mid")

    small = {k: (v[-20:] if isinstance(v, list) else v) for k, v in metrics.items()}
    metrics_text = json.dumps(small, default=str)
    prefs_text = json.dumps(prefs, default=str)
    logic_text = logic_descriptions_text()

    system = (
        "You are a senior quant. Use evidence from metrics only. "
        "Return SINGLE-LINE JSON: {\"score\":int, \"mode\":\"quick|mid|trend\", \"reason\":\"max 8 words\"}."
    )

    prompt = (
        system + "\n\n"
        f"Symbol: {symbol}\nPrice: {price}\n"
        f"Base score: {base}\nMemory: {memory}\nLast pnl approx: {pnl:.2f}\n\n"
        f"Metrics: {metrics_text}\n\nLogic descriptions:\n{logic_text}\n\n"
        "List 3 pros and 3 cons briefly then RETURN EXACT single-line JSON only."
    )

    raw = ""
    try:
        raw = await asyncio.to_thread(_call_openai_sync, prompt, os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 300)
    except Exception:
        raw = ""

    parsed = None
    if raw:
        # Try direct JSON parse, then try to extract JSON substring
        try:
            parsed = json.loads(raw.strip())
        except Exception:
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

    # If OpenAI failed or parsed is invalid, fallback to heuristic scorer
    if not parsed or not isinstance(parsed, dict) or not {"score", "mode", "reason"} <= set(parsed.keys()):
        # Heuristic fallback: use base + small adjustments
        score = base
        # rsi bias
        rsi = metrics.get("rsi_1m", 50)
        if rsi > 65:
            score += 8
        elif rsi < 35:
            score += 2
        # volume bias
        vol = metrics.get("vol_1m", 0)
        if vol and vol > 0:
            score += 0  # keep neutral unless external volume baseline is provided
        # clamp
        score = max(0, min(100, int(round(score))))
        # mode selection
        mode = "quick"
        if score > 85:
            mode = "trend"
        elif score > 70:
            mode = "mid"
        reason = "heuristic fallback"
        out = {"score": score, "mode": mode, "reason": reason}
        CACHE.set(key, out, ttl_seconds=ttl)
        # store memory minimally
        append_memory(symbol, out)
        return out

    # Normalize/validate parsed
    try:
        s = int(parsed.get("score", 0))
    except Exception:
        s = 0
    m = parsed.get("mode", "quick")
    if m not in ("quick", "mid", "trend"):
        m = "quick"
    reason = str(parsed.get("reason", ""))[:120]
    out = {"score": max(0, min(100, s)), "mode": m, "reason": reason}
    CACHE.set(key, out, ttl_seconds=ttl)
    append_memory(symbol, out)
    return out

# -------------------------
# ensemble: call single scorer n times and average
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
    mode_vals = [s["mode"] for s in scores]
    final_mode = sorted(mode_vals)[len(mode_vals)//2]
    reason = scores[0].get("reason", "") if scores else "ok"
    return {"score": final_score, "mode": final_mode, "reason": reason}

# -------------------------
# End of helpers.py
# -------------------------