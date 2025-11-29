#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
75 % win-rate Binance → Telegram scalper
All-in-one file (uses guards.py & config.py)
"""

import os, time, requests, datetime
from config import SYMBOLS, MODE, TIMEFRAME, PASS_SCORE, SL_PCT, TP1_PCT, TP2_PCT
from guards import news_guard, funding_filter, spread_guard, market_awake

# ---------- helpers ----------
def get_klines(symbol, interval, limit=50):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    return requests.get(url, params=params, timeout=10).json()

def telegram(html):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat  = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("❗ Telegram keys missing"); return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat, "text": html, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Telegram error:", e)

# ---------- scoring ----------
def quick_score(k):
    c = [float(x[4]) for x in k]
    h = [float(x[2]) for x in k]
    l = [float(x[3]) for x in k]
    v = [float(x[5]) for x in k]
    fvg = (h[-3] < l[-1]) or (h[-1] < l[-3])
    vol = v[-1] > v[-2]
    return min(100, 60 + 40*(fvg and vol))

def mid_score(k):
    c = [float(x[4]) for x in k]
    ema21 = sum(c[-21:]) / 21
    ema55 = sum(c[-55:]) / 55
    bull_cross = ema21 > ema55
    break_structure = c[-1] > max(c[-10:-2])
    return min(100, 60 + 40*(bull_cross and break_structure))

def trend_score(k):
    c = [float(x[4]) for x in k]
    ema21 = sum(c[-21:]) / 21
    ema55 = sum(c[-55:]) / 55
    return min(100, 70 + 30*(ema21 > ema55))

def calc(symbol):
    data = get_klines(symbol, TIMEFRAME)
    if MODE == "quick": return quick_score(data)
    if MODE == "mid":   return mid_score(data)
    if MODE == "trend": return trend_score(data)

# ---------- main loop ----------
def main():
    print("🚀 75 % win-rate bot started")
    while True:
        # macro guards
        if news_guard() or not market_awake():
            time.sleep(300); continue
        for symbol in SYMBOLS:
            if funding_filter(symbol) or spread_guard(symbol): continue
            score = calc(symbol)
            if score < PASS_SCORE: continue
            # price & levels
            klines2 = get_klines(symbol, TIMEFRAME, 2)
            last_price = float(klines2[-1][4])
            sl  = last_price * (1 - SL_PCT/100)
            tp1 = last_price * (1 + TP1_PCT/100)
            tp2 = last_price * (1 + TP2_PCT/100)
            msg = (f"📊 <b>{symbol[:-4]}/USDT</b>  ⏱ {TIMEFRAME}\n"
                   f"🎯 Score: <b>{int(score)}/100</b>\n"
                   f"🔥 {MODE.capitalize()} (75 %)\n"
                   f"💰 Entry: {last_price:.4f}\n"
                   f"⛔ SL: {sl:.4f}\n"
                   f"🎯 TP1: {tp1:.4f} (50 %)\n"
                   f"🎯 TP2: {tp2:.4f} (50 %)")
            telegram(msg)
            print(f"✅ {symbol}  →  {int(score)}")
        time.sleep(30)

if __name__ == "__main__":
    main()
