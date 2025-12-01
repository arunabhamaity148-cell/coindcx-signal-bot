import os, time, json, asyncio, random, hashlib, aiohttp
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------
# Load ENV
# -----------------------
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CYCLE_TIME = int(os.getenv("CYCLE_TIME", "20"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "1800"))
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "78"))

SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
    "DOTUSDT","MATICUSDT","LTCUSDT","LINKUSDT","FILUSDT","ATOMUSDT","ETCUSDT","OPUSDT",
    "ICPUSDT","APTUSDT","NEARUSDT","INJUSDT","ARBUSDT","SUIUSDT","AAVEUSDT","EOSUSDT",
    "CRVUSDT","RUNEUSDT","XMRUSDT","FTMUSDT","SNXUSDT","DYDXUSDT","GMTUSDT","HBARUSDT",
    "THETAUSDT","AXSUSDT","FLOWUSDT","KAVAUSDT","ZILUSDT","GALAUSDT","MTLUSDT",
    "CHZUSDT","RNDRUSDT","SANDUSDT","MANAUSDT","1INCHUSDT","COMPUSDT","KLAYUSDT",
    "TOMOUSDT","VETUSDT","BLURUSDT","STRKUSDT"
]

# -----------------------
# OpenAI Client
# -----------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# Cooldown
# -----------------------
cooldown = {}

# -----------------------
# Metric mock
# (No CCXT → AI-only scoring)
# -----------------------
def fetch_price(symbol):
    return random.uniform(50, 70000)

# -----------------------
# Build AI Prompt (58 LOGICS)
# -----------------------
LOGIC_LIST = """
HTF_EMA_1h_15m, HTF_EMA_1h_4h, HTF_EMA_1h_8h, HTF_Structure_Break, HTF_Structure_Reject,
Imbalance_FVG, Trend_Continuation, Trend_Exhaustion, Micro_Pullback, Wick_Strength,
Sweep_Reversal, Vol_Sweep_1m, Vol_Sweep_5m, Delta_Divergence_1m, Delta_Divergence_HTF,
Iceberg_1m, Iceberg_v2, Orderbook_Wall_Shift, Liquidity_Wall, Liquidity_Bend,
ADR_DayRange, ATR_Expansion, Phase_Shift, Price_Compression, Speed_Imbalance,
Taker_Pressure, HTF_Volume_Imprint, Tiny_Cluster_Imprint, Absorption, Recent_Weakness,
Spread_Snap_0_5s, Spread_Snap_0_25s, Tight_Spread_Filter, Spread_Safety,
BE_SL_AutoLock, Liquidation_Distance, Kill_Zone_5m, Kill_Zone_HTF,
Kill_Switch_Fast, Kill_Switch_Primary, News_Guard, 30s_Recheck_Loop,
Partial_Exit_Logic, BTC_Risk_Filter_L1, BTC_Risk_Filter_L2,
BTC_Funding_OI_Combo, Funding_Extreme, Funding_Delta_Speed,
Funding_Arbitrage, OI_Spike_5pct, OI_Spike_Sustained,
ETH_BTC_Beta_Divergence, Options_Gamma_Flip, Heatmap_Sweep,
Micro_Slip, Order_Block, Score_Normalization, Final_Signal_Score
"""

def build_prompt(symbol, price):
    return f"""
You are an expert quant trading signal engine.
Evaluate FUTURES entry based on 58 logics:

{LOGIC_LIST}

Return STRICT JSON ONLY:
{{
 "score": 0-100,
 "mode": "quick" | "mid" | "trend",
 "reason": "short reason"
}}

Symbol: {symbol}
Price: {price}

Rules:
- If BTC not calm → score = 0.
- Trend alignment boosts score.
- Spread/funding/liquidity issues reduce score.
- Mode rules:
  quick = scalp, mid = momentum, trend = HTF alignment.
- Output JSON only.
"""

# -----------------------
# AI Call
# -----------------------
async def ai_score(symbol, price):
    prompt = build_prompt(symbol, price)

    def call_api():
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Only JSON response allowed."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=150
        )
        return resp.choices[0].message.content

    raw = await asyncio.to_thread(call_api)

    try:
        return json.loads(raw)
    except:
        # Try to extract JSON
        import re
        m = re.search(r"\{.*\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return None
        return None

# -----------------------
# TP/SL Calculation
# -----------------------
def calc_tp_sl(price, mode):
    if mode == "quick":
        tp = price * 1.016
        sl = price * 0.99
    elif mode == "mid":
        tp = price * 1.02
        sl = price * 0.99
    else:
        tp = price * 1.04
        sl = price * 0.985
    return round(tp, 2), round(sl, 2)

# -----------------------
# TEXT message ( NO HTML )
# -----------------------
def build_msg(symbol, price, score, mode, reason, tp, sl):
    return f"""
🔥 AI SIGNAL ({mode.upper()})  
Symbol: {symbol}  
Price: {price}  

Score: {score}  
Reason: {reason}  

🎯 TP: {tp}  
🛡️ SL: {sl}  

Time: {time.strftime('%H:%M:%S')}  
"""

# -----------------------
# Telegram Sender
# -----------------------
async def send(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}

    async with aiohttp.ClientSession() as s:
        await s.post(url, data=payload)

# -----------------------
# MAIN LOOP
# -----------------------
async def run():
    print("AI Bot Running...")

    while True:
        for sym in SYMBOLS:

            # Cooldown check
            if cooldown.get(sym, 0) > time.time():
                continue

            price = fetch_price(sym)
            res = await ai_score(sym, price)

            if not res:
                continue

            score = res.get("score", 0)
            mode = res.get("mode", "quick")
            reason = res.get("reason", "")

            if score >= SCORE_THRESHOLD:
                tp, sl = calc_tp_sl(price, mode)
                msg = build_msg(sym, price, score, mode, reason, tp, sl)

                await send(msg)

                cooldown[sym] = time.time() + COOLDOWN_SECONDS
                print("SENT:", sym, score, mode)

        await asyncio.sleep(CYCLE_TIME)

# -----------------------
# Start
# -----------------------
if __name__ == "__main__":
    asyncio.run(run())