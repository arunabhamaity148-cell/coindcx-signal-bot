# main.py — AI-Only Scoring System
import os
import asyncio
import json
import random
import aiohttp
import time
from dotenv import load_dotenv
from openai import OpenAI

from helpers import (
    now_ts, human_time, esc, calc_tp_sl, build_ai_prompt
)

load_dotenv()

# -------------------
# ENV Config
# -------------------
BOT_TOKEN        = os.getenv("BOT_TOKEN")
CHAT_ID          = os.getenv("CHAT_ID")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
MODEL_NAME       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CYCLE_TIME       = int(os.getenv("CYCLE_TIME", "20"))
SCORE_THRESHOLD  = int(os.getenv("SCORE_THRESHOLD", "78"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "1800"))

SYMBOLS = [
    s.strip() for s in os.getenv("SYMBOLS",
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,OPUSDT,LINKUSDT,"
    "MATICUSDT,APTUSDT,NEARUSDT,INJUSDT,SUIUSDT,FILUSDT,ATOMUSDT,ETCUSDT,RNDRUSDT,FTMUSDT,"
    "ARBUSDT,AAVEUSDT,CRVUSDT,EOSUSDT,CHZUSDT,SANDUSDT,MANAUSDT,THETAUSDT,HBARUSDT,COMPUSDT,"
    "DYDXUSDT,LTCUSDT,ICPUSDT,RUNEUSDT,XMRUSDT,VETUSDT,1INCHUSDT,GALAUSDT,ZILUSDT,FLOWUSDT,"
    "KAVAUSDT,MTLUSDT,TOMOUsdt,OPUSDT,LDOUSDT,BLURUSDT,STRKUSDT,ZRXUSDT"
    ).split(",")
    if s.strip()
]

PREFERENCES = {
    "BTC_CALM_REQUIRED": True,
    "TP_SL": {
        "quick": {"tp_pct":1.6,"sl_pct":1.0},
        "mid":   {"tp_pct":2.0,"sl_pct":1.0},
        "trend": {"tp_pct":4.0,"sl_pct":1.5}
    }
}

client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------
# Cooldown
# -------------------
class Cooldown:
    def __init__(self):
        self.store = {}

    def ok(self, symbol):
        return self.store.get(symbol, 0) < time.time()

    def set(self, symbol):
        self.store[symbol] = time.time() + COOLDOWN_SECONDS


# -------------------
# Telegram Sender
# -------------------
async def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    async with aiohttp.ClientSession() as s:
        try:
            async with s.post(url, data=payload, timeout=10) as r:
                return await r.json()
        except:
            return None


# -------------------
# Market Snapshot (mock)
# -------------------
async def fetch_market_snapshot(sym):
    # Later: replace with ccxt actual data fetch
    price = random.uniform(10, 50000)
    metrics = {
        "vol_1m": round(random.uniform(100,50000),2),
        "vol_5m": round(random.uniform(200,90000),2),
        "atr": round(random.uniform(0.5,6.0),2),
        "rsi": round(random.uniform(15,85),1),
        "oi_1h_change_pct": round(random.uniform(-5,10),2),
        "spread_pct": round(random.uniform(0.01,0.5),3),
        "funding_rate": round(random.uniform(-0.001,0.001),6)
    }
    return price, metrics


# -------------------
# AI scoring function
# -------------------
async def ai_score(symbol, price, metrics):
    prompt = build_ai_prompt(symbol, price, metrics, PREFERENCES)

    def sync_call():
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return STRICT JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        try:
            return r.choices[0].message.content
        except:
            return ""

    raw = await asyncio.to_thread(sync_call)

    try:
        return json.loads(raw)
    except:
        # Try extracting JSON substring
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return None
        return None


# -------------------
# Build Message
# -------------------
def build_msg(symbol, price, score, mode, reason, tp, sl):
    return (
        f"🔥 <b>{mode.upper()} SIGNAL</b> 🔥\n"
        f"<b>{symbol}</b>\n"
        f"Price: {price:.2f}\n"
        f"Score: <b>{score}</b>\n"
        f"Reason: {esc(reason)}\n"
        f"TP: <code>{tp}</code> | SL: <code>{sl}</code>\n"
        f"Time: {human_time()}\n"
        f"<i>AI-Scored • Cooldown: {int(COOLDOWN_SECONDS/60)}m</i>"
    )


# -------------------
# Main Loop
# -------------------
async def main_loop():
    cd = Cooldown()
    print(f"AI-Only Bot Started ({len(SYMBOLS)} coins)…")

    while True:
        for sym in SYMBOLS:

            if not cd.ok(sym):
                continue

            price, metrics = await fetch_market_snapshot(sym)

            result = await ai_score(sym, price, metrics)
            if not result:
                continue

            score = result.get("score", 0)
            mode  = result.get("mode", "quick")
            reason= result.get("reason", "")

            if score >= SCORE_THRESHOLD:
                tp, sl = calc_tp_sl(price, mode)
                msg = build_msg(sym, price, score, mode, reason, tp, sl)
                await send_telegram(msg)
                print(f"[SENT] {sym} | {score} | {mode} | {reason}")
                cd.set(sym)

            await asyncio.sleep(0.15)

        await asyncio.sleep(CYCLE_TIME)


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY missing in .env")
    else:
        asyncio.run(main_loop())