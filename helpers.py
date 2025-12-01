# helpers.py — FINAL 10/10 VERSION
import time, html, json, hashlib, math
from typing import Dict, List

# ---------- TIME UTIL ----------
def now_ts():
    return int(time.time())

def human_time(ts=None):
    if ts is None:
        ts = now_ts()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def esc(x):
    return html.escape("" if x is None else str(x))

# ---------- TP/SL RULE ----------
TP_SL_RULES = {
    "quick": {"tp_pct": 1.8, "sl_pct": 1.0},
    "mid":   {"tp_pct": 2.2, "sl_pct": 1.0},
    "trend": {"tp_pct": 4.0, "sl_pct": 1.4},
}

def calc_tp_sl(price, mode):
    cfg = TP_SL_RULES.get(mode, {"tp_pct":2.0,"sl_pct":1.0})
    tp = price*(1+cfg["tp_pct"]/100)
    sl = price*(1-cfg["sl_pct"]/100)
    return round(tp,8), round(sl,8)

# ---------- SIMPLE TTL CACHE ----------
class SimpleCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        rec = self.store.get(key)
        if not rec: return None
        exp, val = rec
        if exp < time.time():
            del self.store[key]
            return None
        return val

    def set(self, key, val, ttl):
        self.store[key] = (time.time()+ttl, val)

    def make_key(self, *parts):
        raw = "|".join([str(p) for p in parts])
        return hashlib.sha256(raw.encode()).hexdigest()

CACHE = SimpleCache()

# ---------- INDICATORS ----------
def sma(v, n):
    if not v: return 0
    if len(v)<n: return sum(v)/len(v)
    return sum(v[-n:])/n

def ema(v, n):
    if not v: return 0
    if len(v)<n: return sma(v,n)
    k=2/(n+1)
    e=sma(v[:n],n)
    for x in v[n:]:
        e=x*k+e*(1-k)
    return e

def rsi(cl, n=14):
    if len(cl)<n+1: return 50
    gains=[]; losses=[]
    for i in range(1,len(cl)):
        ch=cl[i]-cl[i-1]
        if ch>0: gains.append(ch); losses.append(0)
        else: gains.append(0); losses.append(-ch)
    ag=sum(gains[-n:])/n
    al=sum(losses[-n:])/n
    if al==0: return 100
    rs=ag/al
    return 100-(100/(1+rs))

def atr(ohlc, n=14):
    if len(ohlc)<2: return 0
    tr=[]
    for i in range(1,len(ohlc)):
        h=ohlc[i][2]; l=ohlc[i][3]; pc=ohlc[i-1][4]
        tr.append(max(h-l, abs(h-pc), abs(l-pc)))
    if len(tr)<n:
        return sum(tr)/len(tr)
    e=sum(tr[:n])/n
    k=2/(n+1)
    for x in tr[n:]:
        e=x*k+e*(1-k)
    return e

# ---------- LOGIC DESCRIPTIONS ----------
LOGIC_TEXT = """
HTF_EMA_1h_15m: 1h vs 15m EMA trend alignment
HTF_EMA_1h_4h: 1h vs 4h EMA trend match
HTF_Structure_Break: Breakout confirmation
Imbalance_FVG: Fair value gap / imbalance
Trend_Continuation: Ongoing trend strength
Trend_Exhaustion: Reversal probability
Liquidity_Wall: Nearby liquidity
Orderbook_Wall_Shift: OB wall shift
Vol_Sweep_1m: 1m volume impulse
ATR_Expansion: Volatility breakout
Spread_Safety: Spread must be acceptable
BTC_Risk_Filter_L1: BTC must be stable
Final_Signal_Score: Combined decision
""".strip()

# ---------- PROMPT BUILDER ----------
def build_ai_prompt(symbol, price, metrics, prefs):

    # shrink arrays (for cost)
    small={}
    for k,v in metrics.items():
        if isinstance(v,list):
            small[k]=v[-20:]
        else:
            small[k]=v

    return f"""
You are a professional futures signal scorer.

Return ONLY valid single-line JSON:
{{"score":0-100,"mode":"quick|mid|trend","reason":"4-12 words"}}

Symbol: {symbol}
Price: {price}
Metrics: {json.dumps(small)}
Preferences: {json.dumps(prefs)}

Logic Descriptions:
{LOGIC_TEXT}

Rules:
1) If BTC is unstable → score = 0
2) Penalize high spread / low liquidity
3) Prefer QUICK for momentum scalps
4) MID for stable continuation
5) TREND for multi-timeframe alignment
6) Output ONLY JSON, nothing else
""".strip()