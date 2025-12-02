# ============================
# helpers.py — FINAL (Stable + Full 58 Logic + Strong AI Scoring)
# ============================

import os, time, json, html, hashlib, random, asyncio
from typing import Optional

# ---------- utils ----------
def now_ts(): return int(time.time())

def human_time(ts=None):
    return time.strftime("%Y-%m-%d %H:%M:%S",
                         time.localtime(ts or now_ts()))

def esc(x): return html.escape("" if x is None else str(x))


# ---------- TP/SL ----------
TP_SL = {
    "quick": {"tp": 1.8, "sl": 1.0},
    "mid":   {"tp": 2.2, "sl": 1.1},
    "trend": {"tp": 3.5, "sl": 1.2},
}

def calc_tp_sl(price, mode):
    r = TP_SL.get(mode, TP_SL["mid"])
    tp = price * (1 + r["tp"] / 100)
    sl = price * (1 - r["sl"] / 100)
    return round(tp,8), round(sl,8)


# ---------- CACHE ----------
class Cache:
    def __init__(self):
        self.s = {}

    def get(self, key):
        v = self.s.get(key)
        if not v: return None
        exp,val = v
        if exp < time.time():
            self.s.pop(key, None)
            return None
        return val

    def set(self, key, val, ttl):
        self.s[key] = (time.time()+ttl, val)

    def make(self, *p):
        return hashlib.sha256("|".join([str(x) for x in p]).encode()).hexdigest()

CACHE = Cache()


# ---------- Indicators ----------
def sma(v,p):
    return sum(v[-p:])/p if len(v)>=p else (sum(v)/len(v) if v else 0)

def ema(v,p):
    if not v: return 0
    if len(v)<p: return sma(v,p)
    seed=sum(v[:p])/p
    out=seed
    k=2/(p+1)
    for x in v[p:]:
        out = x*k + out*(1-k)
    return out

def rsi_from_closes(c, period=14):
    if len(c)<period+1: return 50
    g,l=[],[]
    for i in range(1,len(c)):
        ch=c[i]-c[i-1]
        g.append(max(ch,0))
        l.append(abs(min(ch,0)))
    ag=sum(g[-period:])/period
    al=sum(l[-period:])/period
    if al==0: return 100
    rs=ag/al
    return 100-(100/(1+rs))

def atr(ohlc, p=14):
    if len(ohlc)<2: return 0
    trs=[]
    for i in range(1,len(ohlc)):
        h,l,pc = ohlc[i][2], ohlc[i][3], ohlc[i-1][4]
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    if len(trs)<p: return sum(trs)/len(trs)
    seed=sum(trs[:p])/p
    x=seed; k=2/(p+1)
    for t in trs[p:]:
        x = t*k + x*(1-k)
    return x


# ---------- 58 logic description ----------
# (AI prompt boosting — MUST HAVE)
LOGIC_DESC = """
HTF_EMA_1h_15m, HTF_EMA_1h_4h, HTF_Structure_Break, Imbalance_FVG,
Trend_Continuation, Trend_Exhaustion, Micro_Pullback, Wick_Strength,
Sweep_Reversal, Vol_Sweep_1m, Vol_Sweep_5m, Delta_Divergence_1m,
Iceberg_1m, Orderbook_Wall_Shift, Liquidity_Wall, Liquidity_Bend,
ADR_DayRange, ATR_Expansion, Phase_Shift, Price_Compression,
Speed_Imbalance, Taker_Pressure, HTF_Volume_Imprint,
Tiny_Cluster_Imprint, Absorption, Recent_Weakness, Spread_Snap_0_5s,
Spread_Snap_0_25s, Tight_Spread_Filter, Spread_Safety,
Liquidation_Distance, Kill_Zone_5m, Kill_Zone_HTF, News_Guard,
BTC_Risk_Filter_L1, BTC_Risk_Filter_L2, BTC_Funding_OI_Combo,
Funding_Extreme, Funding_Delta_Speed, OI_Spike_5pct,
OI_Spike_Sustained, ETH_BTC_Beta_Divergence, Options_Gamma_Flip,
Heatmap_Sweep, Micro_Slip, Order_Block, Score_Normalization,
Final_Signal_Score
"""


# ---------- MEMORY ----------
def mem_get(sym):
    return CACHE.get("m_"+sym) or []

def mem_add(sym, x):
    m = mem_get(sym)
    m.append(x)
    m = m[-5:]
    CACHE.set("m_"+sym, m, 86400)


# ---------- OPENAI CALLER ----------
def call_openai(prompt, model):
    try:
        from openai import OpenAI
        cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        r = cli.responses.create(
            model=model,
            input=prompt,
            temperature=0,
            max_output_tokens=150
        )
        if hasattr(r,"output_text"):
            return r.output_text.strip()
        return str(r)
    except:
        return "{}"


# ---------- AI SCORE (Strong Version) ----------
async def ai_single(symbol, price, metrics, prefs):
    fp = f"{symbol}|{round(price,5)}|{round(metrics.get('rsi_1m',50),2)}"
    key = CACHE.make("ai", fp)
    c = CACHE.get(key)
    if c: return c

    memory = mem_get(symbol)

    prompt = f"""
You are a senior quant working for a world-class high-frequency crypto desk.
You MUST analyse the metrics EXACTLY and return **ONE LINE JSON** ONLY.

Metrics: {json.dumps(metrics)}
Memory: {memory}
Logic: {LOGIC_DESC}

RULES:
1) score = 0–100 (higher = stronger signal)
2) mode = quick | mid | trend
3) reason = max 6 words
4) DO NOT write anything except JSON.

Return JSON only:
{{"score":int, "mode":"quick|mid|trend", "reason":"text"}}
"""

    raw = await asyncio.to_thread(call_openai, prompt, os.getenv("OPENAI_MODEL","gpt-4o-mini"))

    # try clean JSON
    try:
        j = json.loads(raw.strip())
    except:
        import re
        m = re.search(r"\{.*\}", raw)
        j = json.loads(m.group(0)) if m else {"score":0,"mode":"quick","reason":"fail"}

    out = {
        "score": int(j.get("score",0)),
        "mode": j.get("mode","quick"),
        "reason": j.get("reason","")
    }

    CACHE.set(key, out, 40)
    mem_add(symbol, out)
    return out


# ---------- ENSEMBLE ----------
async def ensemble_score(symbol, price, metrics, prefs, n=3):
    L=[]
    for _ in range(n):
        r = await ai_single(symbol, price, metrics, prefs)
        L.append(r)

    if not L:
        return {"score":0,"mode":"quick","reason":"no_score"}

    score = int(sum([x["score"] for x in L]) / len(L))
    mode  = sorted([x["mode"] for x in L])[1]
    reason = L[0]["reason"]

    return {"score":score, "mode":mode, "reason":reason}