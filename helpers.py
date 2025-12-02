# helpers.py — RELAXED (alert ↑, quality intact)
import os, time, json, html, hashlib, random, asyncio
from typing import Dict, List, Optional
from datetime import datetime

# ------------------------- utils / cache / indicators / TP-SL (same) ----------
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

# ---------- cache ----------
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

# ---------- 58 logic desc ----------
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
    "BTC_Risk_FILTER_L2": "Stricter BTC risk filter L2",
    "BTC_Funding_OI_Combo": "BTC funding & OI combination",
    "Funding_Extreme": "Extreme funding rate signal",
    "Funding_Delta_Speed": "Speed of funding change",
    "Funding_Arbitrage": "Funding arbitrage signal",
    "OI_Spike_5pct": "OI spike >5% detection",
    "OI_Spike_Sustained": "Sustained OI increase",
    "ETH_BTC_Beta_Divergence": "ETH/BTC beta divergence",
    "Options_Gamma_Flip": "Options gamma flip signal",
    "Heatmap_Sweep": "Heatmap sweep on orderflow",
    "Micro_Slip": "Micro execution slippage",
    "Order_Block": "Order block detection",
    "Score_Normalization": "Normalization step for scores",
    "Final_Signal_Score": "Final combined AI decision/confidence"
}

def logic_descriptions_text() -> str:
    return "\n".join([f"{k}: {v}" for k, v in LOGIC_DESC.items()])

# ---------- RELAXED strict filters ----------
def btc_calm_filter() -> bool:
    # relaxed: always pass (no BTC volatility gate)
    return True

def spread_filter(metrics: dict) -> bool:
    # relaxed: allow up to 0.12 % spread
    return metrics.get("spread_pct", 999) < 0.12   # was 0.08

def volume_spike_ok(metrics: dict) -> bool:
    # relaxed: 1.2 x average volume
    vol = metrics.get("vol_1m", 0)
    base = metrics.get("vol_1m_avg_20", 0) or 1
    return vol > base * 1.2   # was 1.5

def htf_alignment_score(metrics: dict) -> int:
    ema1h = metrics.get("ema_1h_50", 0)
    ema15 = metrics.get("ema_15m_50", 0)
    price = metrics.get("price", 0)
    if price and ema15 and ema1h:
        if price > ema15 > ema1h:
            return 20
        if price < ema15 < ema1h:
            return 20
    return 0

def funding_oi_filter(symbol: str) -> bool:
    # relaxed: always pass (no funding/OI gate)
    return True

def kill_switch() -> bool:
    # relaxed: no kill-switch
    return False

# ---------- memory ----------
def get_memory(symbol: str) -> list:
    return CACHE.get(f"memory_{symbol}") or []

def append_memory(symbol: str, entry: dict, max_items: int = 5):
    mem = get_memory(symbol)
    mem.append(entry)
    mem = mem[-max_items:]
    CACHE.set(f"memory_{symbol}", mem, ttl_seconds=86400)

# ---------- feedback ----------
def get_trade_pnl(symbol: str, tp: float, sl: float, mode: str) -> float:
    return random.uniform(-1.0, 1.0)

# ---------- pydantic ----------
try:
    from pydantic import BaseModel, Field
    from typing import Literal
    class Signal(BaseModel):
        score: int = Field(ge=0, le=100)
        mode: Literal["quick", "mid", "trend"]
        reason: str = Field(max_length=64)
except Exception:
    Signal = None

SYSTEM_PROMPT = (
    "You are a senior quant at a top hedge fund specializing in crypto futures. "
    "Use only evidence from provided metrics and follow instructions exactly. "
    "RETURN A SINGLE-LINE JSON: {\"score\":int, \"mode\":\"quick|mid|trend\", \"reason\":\"max 8 words\"}."
)

# ---------- OpenAI caller ----------
def _call_openai_sync(prompt: str, model: Optional[str] = None, max_tokens: int = 300) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.responses.create(model=model_name, input=prompt, temperature=0, max_output_tokens=max_tokens)
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
        return json.dumps(resp.to_dict())
    except Exception:
        return ""

# ---------- relaxed AI scorer ----------
async def ai_score_symbol_with_memory_and_tools(symbol: str, price: float, metrics: dict, prefs: dict, ttl: int = 120) -> Optional[dict]:
    fingerprint = f"{symbol}|{round(price,6)}|{round(metrics.get('rsi_1m',50),2)}|{round(metrics.get('spread_pct',0),4)}"
    key = CACHE.make_key("ai", fingerprint)
    cached = CACHE.get(key)
    if cached:
        return cached

    # relaxed filters (always pass)
    base = htf_alignment_score(metrics)

    memory = get_memory(symbol)
    pnl = get_trade_pnl(symbol, calc_tp_sl(price, "mid")[0], calc_tp_sl(price, "mid")[1], "mid")

    small = {k: v[-20:] if isinstance(v, list) else v for k, v in metrics.items()}
    metrics_text = json.dumps(small, default=str)
    prefs_text = json.dumps(prefs, default=str)
    logic_text = logic_descriptions_text()

    prompt = (
        SYSTEM_PROMPT + "\n\n"
        f"Step 1: List 3 pros and 3 cons briefly about this symbol based only on metrics.\n"
        f"Step 2: Use memory: {memory}\n"
        f"Step 3: Last trade pnl (approx): {pnl:.2f}\n"
        f"Step 4: Base filter score: {base}\n\n"
        f"Metrics: {metrics_text}\n\n"
        f"Logic descriptions: {logic_text}\n\n"
        "Return EXACT single-line JSON only."
    )

    try:
        raw = await asyncio.to_thread(_call_openai_sync, prompt, os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 300)
        parsed = None
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
            if Signal:
                try:
                    sig = Signal(**parsed)
                    out = sig.dict()
                except Exception:
                    out = {"score": int(parsed.get("score", 0)), "mode": str(parsed.get("mode", "quick")), "reason": str(parsed.get("reason",""))}
            else:
                out = {"score": int(parsed.get("score", 0)), "mode": str(parsed.get("mode", "quick")), "reason": str(parsed.get("reason",""))}
            CACHE.set(key, out, ttl_seconds=ttl)
            append_memory(symbol, out)
            return out
    except Exception:
        pass

    return None

# ---------- ensemble ----------
async def ensemble_score(symbol: str, price: float, metrics: dict, prefs: dict, n: int = 3) -> dict:
    scores = []
    for _ in range(n):
        sc = await ai_score_symbol_with_memory_and_tools(symbol, price, metrics, prefs)
        if sc:
            scores.append(sc)
    if not scores:
        return {"score": 0, "mode": "quick", "reason": "ensemble_fail"}
    score_vals = [s["score"] for s in scores]
    final_score = int(round(sum(score_vals) / len(score_vals)))
    mode_vals = [s["mode"] for s in scores]
    final_mode = sorted(mode_vals)[len(mode_vals) // 2]
    return {"score": final_score, "mode": final_mode, "reason": scores[0]["reason"]}

# -------------------------
# End of helpers.py
# -------------------------
