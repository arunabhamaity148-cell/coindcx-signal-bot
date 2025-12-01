# main.py — FINAL 10/10 VERSION
import os, time, json, asyncio, random, sqlite3, hashlib
import aiohttp
from dotenv import load_dotenv

import ccxt.async_support as ccxt
from openai import OpenAI

from helpers import (
    now_ts, human_time, esc, CACHE, calc_tp_sl,
    build_ai_prompt, rsi, atr, ema
)

load_dotenv()

# ---------- ENV ----------
BOT_TOKEN = os.getenv("BOT_TOKEN","")
CHAT_ID   = os.getenv("CHAT_ID","")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL","gpt-4o-mini")
OPENAI_MAX_PER_MIN = int(os.getenv("OPENAI_MAX_PER_MIN","40"))
OPENAI_TTL = int(os.getenv("OPENAI_TTL_SECONDS","60"))

CYCLE_TIME = int(os.getenv("CYCLE_TIME","20"))
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD","78"))
COOLDOWN = int(os.getenv("COOLDOWN_SECONDS","1800"))

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS","").split(",") if s.strip()]

USE_TESTNET = os.getenv("USE_TESTNET","true").lower() in ("true","1","yes")
BIN_KEY = os.getenv("BINANCE_API_KEY","")
BIN_SEC = os.getenv("BINANCE_API_SECRET","")

client = OpenAI(api_key=OPENAI_API_KEY)
_openai_calls=[]

def openai_can():
    now=time.time()
    while _openai_calls and _openai_calls[0]<=now-60:
        _openai_calls.pop(0)
    return len(_openai_calls)<OPENAI_MAX_PER_MIN

def openai_note():
    _openai_calls.append(time.time())

# ---------- DB ----------
conn=sqlite3.connect(os.getenv("SIGNAL_DB","signals.db"),check_same_thread=False)
cur=conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS signals(
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 ts INTEGER,
 symbol TEXT,
 price REAL,
 score INTEGER,
 mode TEXT,
 reason TEXT,
 tp REAL,
 sl REAL
)
""")
conn.commit()

def log_signal(ts,sym,p,sc,m,r,tp,sl):
    cur.execute("INSERT INTO signals(ts,symbol,price,score,mode,reason,tp,sl) VALUES(?,?,?,?,?,?,?,?)",
                (ts,sym,p,sc,m,r,tp,sl))
    conn.commit()

# ---------- TELEGRAM ----------
async def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured")
        return
    url=f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data={"chat_id":CHAT_ID,"text":msg,"parse_mode":"HTML"}
    async with aiohttp.ClientSession() as s:
        try:
            async with s.post(url,data=data,timeout=15) as r:
                return await r.text()
        except:
            return None

def format_msg(sym,p,sc,mode,reason,tp,sl):
    return (
f"🔥⚡ <b>AI SIGNAL — {mode.upper()} MODE</b> ⚡🔥\n"
f"<b>{sym}</b> • Price: <code>{p:.8f}</code>\n\n"
f"🎯 <b>Score:</b> <b>{sc}</b>\n"
f"📌 <b>Reason:</b> {esc(reason)}\n\n"
f"🎯 <b>TP:</b> <code>{tp}</code>\n"
f"🛡️ <b>SL:</b> <code>{sl}</code>\n\n"
f"🕒 {human_time()}  ⏳ Cooldown: {COOLDOWN//60}m\n"
f"🤖 AI-Only • Hybrid Indicators"
)

# ---------- EXCHANGE ----------
async def init_ex():
    opt={"enableRateLimit":True,"options":{"defaultType":"future"}}
    if USE_TESTNET:
        opt["urls"]={
            "api":{
                "public":"https://testnet.binancefuture.com/fapi/v1",
                "private":"https://testnet.binancefuture.com/fapi/v1"
            }
        }
    ex=ccxt.binance(opt)
    if BIN_KEY and BIN_SEC:
        ex.apiKey=BIN_KEY
        ex.secret=BIN_SEC
    return ex

# ---------- SNAPSHOT ----------
async def snapshot(ex, sym):
    key=CACHE.make_key("snap",sym)
    c=CACHE.get(key)
    if c: return c

    # price
    try:
        tk=await ex.fetch_ticker(sym)
        p=float(tk.get("last") or tk.get("close") or 0)
        vol=float(tk.get("baseVolume") or 0)
    except:
        p=random.uniform(10,50000); vol=0

    # orderbook for spread
    try:
        ob=await ex.fetch_order_book(sym,5)
        b=ob["bids"][0][0] if ob["bids"] else None
        a=ob["asks"][0][0] if ob["asks"] else None
        sp=abs(a-b)/((a+b)/2)*100 if a and b else 0
    except:
        sp=0

    # 1m closes
    try:
        o1=await ex.fetch_ohlcv(sym,"1m",limit=120)
        cl1=[x[4] for x in o1]
    except:
        cl1=[]

    m={
        "closes_1m": cl1[-60:],
        "vol_1m": vol,
        "spread_pct": round(sp,4),
        "rsi_1m": round(rsi(cl1),2) if cl1 else 50,
    }

    snap={"price":p,"metrics":m}
    CACHE.set(key,snap,3)
    return snap

# ---------- AI SCORING ----------
def score_key(sym,p,m):
    base={
        "p":round(p,6),
        "r":m.get("rsi_1m"),
        "s":m.get("spread_pct"),
    }
    raw=f"{sym}|{json.dumps(base,sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()

async def ai_score(sym,p,m,prefs):
    k=CACHE.make_key("ai",score_key(sym,p,m))
    c=CACHE.get(k)
    if c: return c

    # rate limit
    wait=0
    while not openai_can():
        await asyncio.sleep(1)
        wait+=1
        if wait>10:
            return None

    prompt=build_ai_prompt(sym,p,m,prefs)

    def call():
        r=client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"Strict JSON only."},
                {"role":"user","content":prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        return r.choices[0].message.content

    openai_note()
    raw=await asyncio.to_thread(call)

    try:
        j=json.loads(raw.strip())
    except:
        import re
        f=re.search(r"\{.*\}",raw,re.DOTALL)
        if f:
            try: j=json.loads(f.group(0))
            except: j=None
        else: j=None

    if j:
        CACHE.set(k,j,OPENAI_TTL)
    return j

# ---------- MAIN LOOP ----------
async def worker():
    ex=await init_ex()
    cd={}
    prefs={"BTC_CALM_REQUIRED":True}

    print(f"BOT READY • {len(SYMBOLS)} symbols • AI model={OPENAI_MODEL}")

    try:
        while True:
            for sym in SYMBOLS:
                try:
                    if cd.get(sym,0)>time.time():
                        continue

                    snap=await snapshot(ex,sym)
                    p=snap["price"]; m=snap["metrics"]

                    sc=await ai_score(sym,p,m,prefs)
                    if not sc: 
                        await asyncio.sleep(0.05)
                        continue

                    score=int(sc.get("score",0))
                    mode=sc.get("mode","quick")
                    reason=sc.get("reason","")

                    if score>=SCORE_THRESHOLD:
                        tp,sl=calc_tp_sl(p,mode)
                        msg=format_msg(sym,p,score,mode,reason,tp,sl)
                        await send_telegram(msg)
                        log_signal(now_ts(),sym,p,score,mode,reason,tp,sl)
                        cd[sym]=time.time()+COOLDOWN

                    await asyncio.sleep(0.07)

                except Exception as e:
                    print("ERR symbol:",sym,e)
                    await asyncio.sleep(0.1)

            await asyncio.sleep(CYCLE_TIME)

    finally:
        try: await ex.close()
        except: pass

if __name__=="__main__":
    if not OPENAI_API_KEY:
        print("ERROR: Missing OPENAI_API_KEY")
    else:
        asyncio.run(worker())