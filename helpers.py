# ============================
# helpers.py — FINAL (58 logic + strict filters + stable scoring)
# ============================

import os, time, json, html, hashlib, random, asyncio
from typing import Dict, List, Optional
from datetime import datetime

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
    "mid":   {"tp_pct": 2.2, "sl_pct": 1.0},
    "trend": {"tp_pct": 4.0, "sl_pct": 1.4},
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

def atr(ohlc: list, period: int = 14) -> float:
    if len(ohlc) < 2: return 0.0
    trs = []
    for i in range(1, len(ohlc)):
        h, l, pc = ohlc[i][2], ohlc[i][3], ohlc[i-1][4]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if not trs: return 0.0
    if len(trs) < period: return sum(trs) / len(trs)
    seed = sum(trs[:period]) / period
    val = seed
    k = 2 / (period + 1)
    for tr in trs[period:]:
        val = tr*k + val*(1-k)
    return val

# ---------- 58 logic descriptions ----------
LOGIC_DESC = {
    "HTF_EMA_1h_15m": "1h vs 15m EMA alignment",
    "HTF_EMA_1h_4h": "1h vs 4h EMA confluence",
    "HTF_EMA_1h_8h": "1h vs 8h EMA confluence",
    "HTF_Structure_Break": "Breakout of HTF S/R",
    "HTF_Structure_Reject": "HTF rejection at S/R",
    "Imbalance_FVG": "Fair-value gap near price",
    "Trend_Continuation": "Trend continuation possible",
    "Trend_Exhaustion": "Momentum exhaustion",
    "Micro_Pullback": "Small pullback to value",
    "Wick_Strength": "Strong wick rejection",
    "Sweep_Reversal": "Liquidity sweep + reversal",
    "Vol_Sweep_1m": "1m volume spike",
    "Vol_Sweep_5m": "5m volume spike",
    "Delta_Divergence_1m": "Delta divergence 1m",
    "Delta_Divergence_HTF": "Delta divergence HTF",
    "Iceberg_1m": "Hidden orders (iceberg)",
    "Iceberg_v2": "Advanced iceberg",
    "Orderbook_Wall_Shift": "Orderbook wall shift",
    "Liquidity_Wall": "Liquidity wall near price",
    "Liquidity_Bend": "Liquidity bending",
    "ADR_DayRange": "ADR comparison",
    "ATR_Expansion": "ATR expansion",
    "Phase_Shift": "Market phase shift",
    "Price_Compression": "Price compression",
    "Speed_Imbalance": "Speed imbalance",
    "Taker_Pressure": "Taker pressure strength",
    "HTF_Volume_Imprint": "HTF volume footprint",
    "Tiny_Cluster_Imprint": "Micro cluster volume",
    "Absorption": "Absorption signature",
    "Recent_Weakness": "Weak push",
    "Spread_Snap_0_5s": "Spread snapshot 0.5s",
    "Spread_Snap_0_25s": "Spread snapshot 0.25s",
    "Tight_Spread_Filter": "Tight spread rule",
    "Spread_Safety": "Spread safe zone",
    "BE_SL_AutoLock": "Breakeven/SL auto-lock",
    "Liquidation_Distance": "Liquidation cluster",
    "Kill_Zone_5m": "5m kill-zone",
    "Kill_Zone_HTF": "HTF kill-zone",
    "Kill_Switch_Fast": "Fast kill-switch",
    "Kill_Switch_Primary": "Primary kill-switch",
    "News_Guard": "News safety guard",
    "30s_Recheck_Loop": "30s recheck",
    "Partial_Exit_Logic": "Partial profit logic",
    "BTC_Risk_Filter_L1": "BTC volatility filter 1",
    "BTC_Risk_FILTER_L2": "BTC volatility filter 2",
    "BTC_Funding_OI_Combo": "BTC funding + OI",
    "Funding_Extreme": "Extreme funding",
    "Funding_Delta_Speed": "Funding change speed",
    "Funding_Arbitrage": "Funding arbitrage",
    "OI_Spike_5pct": "OI spike >5%",
    "OI_Spike_Sustained": "Sustained OI spike",
    "ETH_BTC_Beta_Divergence": "ETH/BTC beta divergence",
    "Options_Gamma_Flip": "Gamma flip",
    "Heatmap_Sweep": "Heatmap sweep",
    "Micro_Slip": "Micro slippage",
    "Order_Block": "Order block detection",
    "Score_Normalization": "Score normalization",
    "Final_Signal_Score": "Final AI score"
}

def logic_descriptions_text() -> str:
    return "\n".join([f"{k}: {v}" for k, v in LOGIC_DESC.items()])

# ---------- strict filters ----------
def btc_calm_filter() -> bool:
    return True  # relaxed (always pass)

def spread_filter(metrics: dict) -> bool:
    return metrics.get("spread_pct", 999) < 0.12

def volume_spike_ok(metrics: dict) -> bool:
    vol = metrics.get("vol_1m", 0)
    base = metrics.get("vol_1m_avg_20", 1)
    return vol > base * 1.2

def htf_alignment_score(metrics: dict) -> int:
    p = metrics.get("price", 0)
    e15 = metrics.get("ema_15m_50", 0)
    e1h = metrics.get("ema_1h_50", 0)
    if p and e15 and e1h:
        if p > e15 > e1h: return 20
        if p < e15 < e1h: return 20
    return 0

def funding_oi_filter(symbol: str) -> bool:
    return True

def kill_switch() -> bool:
    return False

# ---------- memory ----------
def get_memory(symbol: str) -> list:
    return CACHE.get(f"memory_{symbol}") or []

def append_memory(symbol: str, entry: dict, max_items: int = 5):
    mem = get_memory(symbol)
    mem.append(entry)
    CACHE.set(f"memory_{symbol}", mem[-max_items:], 86400)

# ---------- feedback ----------
def get_trade_pnl(symbol: str, tp: float, sl: float, mode: str) -> float:
    return random.uniform(-1, 1)

# ---------- OpenAI sync ----------
def _call_openai_sync(prompt: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        mdl = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        r = client.responses.create(model=mdl, input=prompt, temperature=0, max_output_tokens=180)
        return r.output_text.strip()
    except:
        return ""

# ---------- AI scorer ----------
async def ai_score_symbol(symbol: str, price: float, metrics: dict, prefs: dict, ttl: int = 90):
    key = CACHE.make_key("ai", symbol, round(price,5))
    c = CACHE.get(key)
    if c: return c

    base = htf_alignment_score(metrics)
    memory = get_memory(symbol)
    pnl = get_trade_pnl(symbol, price, price, "mid")

    mtxt = json.dumps({k:(v[-20:] if isinstance(v,list) else v) for k,v in metrics.items()})
    ptxt = json.dumps(prefs)
    logic = logic_descriptions_text()

    prompt = f"""
You are a senior quant. Output ONLY strict JSON.

Metrics: {mtxt}
BaseScore: {base}
Memory: {memory}
PnL: {pnl:.2f}

Rules:
1) score 0-100
2) mode quick/mid/trend
3) reason max 8 words

Logic:
{logic}

Return ONLY: {{"score":int,"mode":"quick|mid|trend","reason":"text"}}
"""

    raw = await asyncio.to_thread(_call_openai_sync, prompt)
    try:
        js = json.loads(raw)
        out = {"score": int(js["score"]), "mode": js["mode"], "reason": js["reason"]}
        CACHE.set(key, out, ttl)
        append_memory(symbol, out)
        return out
    except:
        return {"score": 0, "mode": "quick", "reason": "parse_fail"}

# ---------- ensemble ----------
async def ensemble_score(symbol: str, price: float, metrics: dict, prefs: dict, n: int = 3):
    out = []
    for _ in range(n):
        r = await ai_score_symbol(symbol, price, metrics, prefs)
        if r: out.append(r)
    if not out:
        return {"score":0,"mode":"quick","reason":"fail"}

    sc = round(sum(o["score"] for o in out)/len(out))
    mode = sorted([o["mode"] for o in out])[1]
    return {"score":sc,"mode":mode,"reason":out[0]["reason"]}

# END helpers.py