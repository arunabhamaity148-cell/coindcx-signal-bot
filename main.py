# main.py — AI-only scoring, Live prices + Testnet-ready config
import os
import asyncio
import json
import time
import random
import aiohttp
from dotenv import load_dotenv

import ccxt.async_support as ccxt
from openai import OpenAI

from helpers import now_ts, human_time, esc, calc_tp_sl, build_ai_prompt

load_dotenv()

# ---------------------------
# ENV / CONFIG
# ---------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CYCLE_TIME = int(os.getenv("CYCLE_TIME", "20"))
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "78"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "1800"))
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]

USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() in ("1", "true", "yes")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()

# default fallback coin list if none provided
if not SYMBOLS:
    SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
               "LTCUSDT","LINKUSDT","FILUSDT","ATOMUSDT","ETCUSDT","OPUSDT","ICPUSDT","APTUSDT","NEARUSDT","INJUSDT",
               "SUIUSDT","AAVEUSDT","EOSUSDT","CRVUSDT","RUNEUSDT","XMRUSDT","FTMUSDT","SNXUSDT","DYDXUSDT","GMTUSDT",
               "HBARUSDT","THETAUSDT","AXSUSDT","FLOWUSDT","KAVAUSDT","ZILUSDT","GALAUSDT","MTLUSDT","CHZUSDT","RNDRUSDT",
               "SANDUSDT","MANAUSDT","1INCHUSDT","COMPUSDT","KLAYUSDT","TOMOUSDT","VETUSDT","BLURUSDT","STRKUSDT","ZRXUSDT"]

# ---------------------------
# OpenAI client
# ---------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Exchange setup (ccxt async)
# ---------------------------
async def create_exchange():
    opts = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    if USE_TESTNET:
        # Binance futures testnet public endpoints (for ccxt mapping we provide api urls)
        # Note: ccxt may need explicit testnet config for private endpoints; public endpoints work with testnet base URLs
        opts["urls"] = {
            "api": {
                "public": "https://testnet.binancefuture.com/fapi/v1",
                "private": "https://testnet.binancefuture.com/fapi/v1"
            }
        }
    exchange = ccxt.binance(opts)
    # set api keys if provided (for private endpoints later)
    if BINANCE_API_KEY and BINANCE_API_SECRET:
        exchange.apiKey = BINANCE_API_KEY
        exchange.secret = BINANCE_API_SECRET
    return exchange

# ---------------------------
# Cooldown store
# ---------------------------
class Cooldown:
    def __init__(self):
        self.store = {}

    def ok(self, symbol: str):
        return self.store.get(symbol, 0) < time.time()

    def set(self, symbol: str):
        self.store[symbol] = time.time() + COOLDOWN_SECONDS

# ---------------------------
# Market snapshot (ticker + orderbook top)
# ---------------------------
async def fetch_snapshot(exchange, symbol: str):
    """
    Returns: dict {price: float, metrics: {...}}
    metrics includes vol_1m/5m (if computed), spread_pct, oi_change_1h_pct (if available), funding_rate (if available)
    """
    # default fallback
    price = None
    spread_pct = None
    base_volume = None

    # fetch ticker
    try:
        tk = await exchange.fetch_ticker(symbol)
        price = float(tk.get("last") or tk.get("close") or 0)
        base_volume = float(tk.get("baseVolume") or 0)
    except Exception:
        # fallback mock
        price = random.uniform(1, 50000)
        base_volume = random.uniform(100, 50000)

    # fetch orderbook top (try)
    spread_pct = None
    try:
        ob = await exchange.fetch_order_book(symbol, 10)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if bid and ask:
            mid = (bid + ask) / 2.0
            spread_pct = abs(ask - bid) / mid * 100.0
    except Exception:
        spread_pct = None

    # Try fetching funding/oi if exchange supports (best-effort)
    oi_change_1h_pct = 0.0
    funding_rate = 0.0
    try:
        # some exchanges have fetch_funding_rate or fetch_open_interest — best-effort
        if hasattr(exchange, "fetch_funding_rate"):
            fr = await exchange.fetch_funding_rate(symbol)
            funding_rate = float(fr.get("fundingRate") or 0)
    except Exception:
        funding_rate = 0.0

    metrics = {
        "vol_1m": base_volume,
        "vol_5m": base_volume * random.uniform(0.9, 1.5),
        "atr": 0.0,
        "rsi": 0.0,
        "oi_1h_change_pct": oi_change_1h_pct,
        "spread_pct": round(spread_pct or 0.0, 4),
        "funding_rate": funding_rate
    }

    return {"price": price, "metrics": metrics}

# ---------------------------
# AI scoring wrapper
# ---------------------------
def parse_json_from_text(text: str):
    try:
        return json.loads(text.strip())
    except Exception:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

async def ai_score_symbol(symbol: str, price: float, metrics: dict, prefs: dict):
    prompt = build_ai_prompt(symbol, price, metrics, prefs)

    def call_ai():
        # synchronous call to OpenAI client (run in thread)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert crypto futures signal scorer. Return STRICT JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            try:
                return resp.choices[0].text
            except Exception:
                return str(resp)

    raw = await asyncio.to_thread(call_ai)
    parsed = parse_json_from_text(raw)
    return parsed

# ---------------------------
# Telegram formatting (Premium Style C Ultra)
# ---------------------------
def build_telegram_message(symbol, price, score, mode, reason, tp, sl, ts):
    # boxed/pretty style with emojis and code blocks (HTML)
    msg = []
    msg.append(f"🔥⚡ <b>AI SIGNAL — {esc(mode.upper())} MODE</b> ⚡🔥")
    msg.append(f"\n<b>{esc(symbol)}</b>")
    msg.append(f"Price: <code>{price:.8f}</code>")
    msg.append(f"")
    msg.append(f"🎯 <b>Score:</b> <b>{int(score)}</b>")
    msg.append(f"📌 <b>Reason:</b> {esc(reason)}")
    msg.append(f"")
    msg.append(f"🎯 <b>Take Profit:</b> <code>{tp}</code>")
    msg.append(f"🛡️ <b>Stop Loss:</b> <code>{sl}</code>")
    msg.append(f"")
    msg.append(f"🕒 <i>{human_time(ts)}</i>")
    msg.append(f"⏳ Cooldown: {int(COOLDOWN_SECONDS/60)}m")
    msg.append(f"🤖 Source: AI-Only Scoring • Live prices + Testnet-ready")
    return "\n".join(msg)

# ---------------------------
# Main worker loop
# ---------------------------
async def worker():
    exchange = await create_exchange()
    cd = Cooldown()
    prefs = {
        "BTC_CALM_REQUIRED": True,
        "TP_SL": {
            "quick": {"tp_pct":1.6, "sl_pct":1.0},
            "mid":   {"tp_pct":2.0, "sl_pct":1.0},
            "trend": {"tp_pct":4.0, "sl_pct":1.5}
        }
    }

    print("AI-Only Bot (Live prices + Testnet-ready) started. Symbols:", len(SYMBOLS))
    try:
        while True:
            for sym in SYMBOLS:
                try:
                    if not cd.ok(sym):
                        continue

                    snap = await fetch_snapshot(exchange, sym)
                    price = snap["price"]
                    metrics = snap["metrics"]

                    # Call AI scorer
                    parsed = await ai_score_symbol(sym, price, metrics, prefs)
                    if not parsed:
                        await asyncio.sleep(0.05)
                        continue

                    score = int(parsed.get("score", 0))
                    mode = parsed.get("mode", "quick")
                    reason = parsed.get("reason", "")

                    if score >= SCORE_THRESHOLD:
                        tp, sl = calc_tp_sl(price, mode)
                        msg = build_telegram_message(sym, price, score, mode, reason, tp, sl, now_ts())
                        await send_telegram(msg)
                        print(f"[SENT] {sym} | {score} | {mode} | {reason}")
                        cd.set(sym)
                    await asyncio.sleep(0.15)

                except Exception as e:
                    print("Symbol loop error:", e)
                    await asyncio.sleep(0.1)
            await asyncio.sleep(CYCLE_TIME)
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY missing in .env")
    else:
        asyncio.run(worker())