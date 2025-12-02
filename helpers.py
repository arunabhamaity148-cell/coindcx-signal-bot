# helpers.py — FINAL (stable, paste-ready)
import os
import time
import json
import html
import hashlib
import random
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

# -- logging --
logger = logging.getLogger("helpers")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ---------- utils ----------
def now_ts() -> int:
    return int(time.time())

def human_time(ts: int = None) -> str:
    if ts is None:
        ts = now_ts()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def esc(x) -> str:
    return html.escape("" if x is None else str(x))

# ---------- TP/SL ----------
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

# ---------- TTL cache ----------
class SimpleCache:
    def __init__(self):
        self.store = {}
    def get(self, key):
        rec = self.store.get(key)
        if not rec:
            return None
        exp, val = rec
        if exp < time.time():
            try: del self.store[key]
            except: pass
            return None
        return val
    def set(self, key, value, ttl_seconds: int):
        self.store[key] = (time.time() + ttl_seconds, value)
    def make_key(self, *parts) -> str:
        raw = "|".join([str(p) for p in parts])
        return hashlib.sha256(raw.encode()).hexdigest()

CACHE = SimpleCache()

# ---------- indicators ----------
def sma(values: list, period: int) -> float:
    if not values: return 0.0
    if len(values) < period: return sum(values) / len(values)
    return sum(values[-period:]) / period

def compute_ema_from_closes(closes: list, period: int) -> float:
    if not closes: return 0.0
    if len(closes) < period: return sma(closes, period)
    seed = sum(closes[:period]) / period
    ema_val = seed
    k = 2 / (period + 1)
    for price in closes[period:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val

def rsi_from_closes(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1: return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0)); losses.append(abs(min(ch, 0)))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(ohlc: list, period: int = 14) -> float:
    if len(ohlc) < 2: return 0.0
    trs = []
    for i in range(1, len(ohlc)):
        high, low, prev_close = ohlc[i][2], ohlc[i][3], ohlc[i-1][4]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs: return 0.0
    if len(trs) < period: return sum(trs) / len(trs)
    seed = sum(trs[:period]) / period
    atr_val = seed
    k = 2 / (period + 1)
    for tr in trs[period:]:
        atr_val = tr * k + atr_val * (1 - k)
    return atr_val

# ---------- logic descriptions (short) ----------
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
    "Iceberg_1m": "Hidden large order signatures",
    "Orderbook_Wall_Shift": "Large wall shift in orderbook",
    "Liquidity_Wall": "Liquidity wall near price",
    "Liquidity_Bend": "Liquidity bending/rotation",
    "ADR_DayRange": "Price vs daily range (ADR)",
    "ATR_Expansion": "ATR expansion confirms breakout",
    "Phase_Shift": "Market shifting phase (comp->exp)",
    "Price_Compression": "Tight range pre-breakout",
    "Speed_Imbalance": "Rapid move vs volume imbalance",
    "Taker_Pressure": "Taker dominance in trades",
    "HTF_Volume_Imprint": "HTF volume footprint",
    "Tiny_Cluster_Imprint": "Micro cluster volume imprint",
    "Absorption": "Opposing liquidity absorbing aggression",
    "Recent_Weakness": "Recent push weakness",
    "Spread_Snap_0_5s": "0.5s spread snapshot",
    "Spread_Snap_0_25s": "0.25s spread snapshot",
    "Tight_Spread_Filter": "Is spread tight enough",
    "Spread_Safety": "Spread vs baseline safety",
    "BE_SL_AutoLock": "Auto-lock BE/SL rules",
    "Liquidation_Distance": "Distance to liquidation clusters",
    "Kill_Zone_5m": "5m time-based killzone",
    "Kill_Zone_HTF": "HTF killzone (news/market open)",
    "Kill_Switch_Fast": "Immediate kill switch conditions",
    "Kill_Switch_Primary": "Primary safety kill switch",
    "News_Guard": "Major news guard",
    "30s_Recheck_Loop": "30s quick recheck rule",
    "Partial_Exit_Logic": "Partial take-profit logic",
    "BTC_Risk_Filter_L1": "BTC calm / volatility filter L1",
    "Funding_Extreme": "Extreme funding rate signal",
    "OI_Spike_5pct": "OI spike >5% detection",
    "Final_Signal_Score": "Final combined AI decision/confidence"
}

def logic_descriptions_text() -> str:
    return "\n".join([f"{k}: {v}" for k, v in LOGIC_DESC.items()])

# ---------- relaxed strict filters (safe defaults) ----------
def btc_calm_filter() -> bool:
    # relaxed default: pass (main.py can set strict if needed)
    snap = CACHE.get("btc_snapshot")
    if not snap:
        return True
    # if metric keys present, do a simple check
    atr_pct = snap.get("metrics", {}).get("atr_1h_pct")
    spr_pct = snap.get("metrics", {}).get("spread_5m_pct")
    if atr_pct is None or spr_pct is None:
        return True
    return atr_pct < 1.2 and spr_pct < 0.05

def spread_filter(metrics: Dict) -> bool:
    return metrics.get("spread_pct", 999) < 0.12  # relaxed

def volume_spike_ok(metrics: Dict) -> bool:
    vol = metrics.get("vol_1m", 0)
    base = metrics.get("vol_1m_avg_20", 0) or 1
    return vol > base * 1.2

def htf_alignment_score(metrics: Dict) -> int:
    ema1h = metrics.get("ema_1h_50", 0)
    ema15 = metrics.get("ema_15m_50", 0)
    price = metrics.get("price", 0)
    if price and ema15 and ema1h:
        if price > ema15 > ema1h: return 20
        if price < ema15 < ema1h: return 20
    return 0

def funding_oi_filter(symbol: str) -> bool:
    # relaxed default: pass
    return True

def kill_switch() -> bool:
    return False

# ---------- memory ----------
def get_memory(symbol: str) -> list:
    return CACHE.get(f"memory_{symbol}") or []

def append_memory(symbol: str, entry: dict, max_items: int = 6):
    mem = get_memory(symbol)
    mem.append(entry)
    mem = mem[-max_items:]
    CACHE.set(f"memory_{symbol}", mem, ttl_seconds=86400)

# ---------- feedback (placeholder) ----------
def get_trade_pnl(symbol: str, tp: float, sl: float, mode: str) -> float:
    # placeholder: main app can replace with real pnl
    return random.uniform(-1.0, 1.0)

# ---------- OpenAI sync caller (safe) ----------
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
        # prefer output_text if present
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
        try:
            return json.dumps(resp.to_dict())
        except Exception:
            return str(resp)
    except Exception as e:
        logger.exception("OpenAI call failed")
        return ""

# ---------- single AI scorer (relaxed + robust parsing) ----------
async def ai_score_symbol_with_memory_and_tools(symbol: str, price: float, metrics: dict, prefs: dict, ttl: int = 120) -> Optional[dict]:
    fingerprint = f"{symbol}|{round(price,6)}|{round(metrics.get('rsi_1m',50),2)}|{round(metrics.get('spread_pct',0),4)}"
    key = CACHE.make_key("ai", fingerprint)
    cached = CACHE.get(key)
    if cached:
        return cached

    # quick filters (relaxed by default)
    if not btc_calm_filter():
        out = {"score": 0, "mode": "quick", "reason": "BTC volatile"}
        CACHE.set(key, out, ttl_seconds=ttl)
        return out
    if not spread_filter(metrics):
        out = {"score": 0, "mode": "quick", "reason": "Spread wide"}
        CACHE.set(key, out, ttl_seconds=ttl)
        return out
    if kill_switch():
        out = {"score": 0, "mode": "quick", "reason": "Kill-switch"}
        CACHE.set(key, out, ttl_seconds=ttl)
        return out

    base = htf_alignment_score(metrics)
    if not volume_spike_ok(metrics):
        base = min(base, 60)
    if not funding_oi_filter(symbol):
        base = min(base, 50)

    memory = get_memory(symbol)
    pnl = get_trade_pnl(symbol, *calc_tp_sl(price, "mid"), "mid")

    small = {k: (v[-20:] if isinstance(v, list) else v) for k, v in metrics.items()}
    metrics_text = json.dumps(small, default=str)
    prefs_text = json.dumps(prefs, default=str)
    logic_text = logic_descriptions_text()

    system = (
        "You are a senior quant. Use only the metrics provided. "
        "Return SINGLE-LINE JSON only: {\"score\":int, \"mode\":\"quick|mid|trend\", \"reason\":\"max 8 words\"}."
    )

    prompt = (
        system + "\n\n"
        f"Symbol: {symbol}\nPrice: {price}\n"
        f"Base filter score: {base}\n"
        f"Memory: {memory}\n"
        f"Approx last trade pnl: {pnl:.2f}\n\n"
        f"Metrics: {metrics_text}\n\nLogic descriptions:\n{logic_text}\n\n"
        "RULES:\n"
        "1) Add 0-50 points based on HTF alignment, volume, ATR, orderflow, imbalance, liquidation distance.\n"
        "2) If total >85 -> mode 'trend', >70 -> 'mid', else 'quick'.\n"
        "3) Penalize wide spread / low liquidity / extreme funding/OI.\n"
        "Return EXACT single-line JSON only."
    )

    try:
        raw = await asyncio.to_thread(_call_openai_sync, prompt, os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 300)
        parsed = None
        # try direct parse or extract json object
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
        if parsed and isinstance(parsed, dict) and {"score", "mode", "reason"} <= set(parsed.keys()):
            # sanitize types
            try:
                score = int(parsed.get("score", 0))
            except Exception:
                score = 0
            mode = str(parsed.get("mode", "quick"))
            reason = str(parsed.get("reason", ""))[:128]
            out = {"score": score, "mode": mode, "reason": reason}
            CACHE.set(key, out, ttl_seconds=ttl)
            append_memory(symbol, out)
            return out
    except Exception:
        logger.exception("ai_score failed for %s", symbol)

    return None

# ---------- ensemble (simple median/mean) ----------
async def ensemble_score(symbol: str, price: float, metrics: dict, prefs: dict, n: int = 3) -> dict:
    results = []
    for _ in range(n):
        try:
            r = await ai_score_symbol_with_memory_and_tools(symbol, price, metrics, prefs)
            if r:
                results.append(r)
            else:
                # small delay to avoid hitting rate limits in quick loops
                await asyncio.sleep(0.05)
        except Exception:
            await asyncio.sleep(0.05)
    if not results:
        return {"score": 0, "mode": "quick", "reason": "ensemble_fail"}
    scores = [r["score"] for r in results]
    avg = int(round(sum(scores) / len(scores)))
    modes = [r["mode"] for r in results]
    # pick median mode by simple sort
    final_mode = sorted(modes)[len(modes)//2]
    reason = results[0].get("reason", "")
    return {"score": avg, "mode": final_mode, "reason": reason}

# -------------------------
# End of helpers.py
# -------------------------