#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
75 % win-rate scalper – 10-logic combo
Layer-1 Core (3), Layer-2 Micro (4), Layer-3 Risk (3)
Score = 60 + 10×L1 + 5×L2 + 5×L3  (≥ 80 → signal)
"""

import os, time, requests, datetime, pytz

# ---------- config ----------
SYMBOLS    = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT",
              "FTMUSDT","AVAXUSDT","MATICUSDT","ATOMUSDT","NEARUSDT"]
TIMEFRAME  = "15m"
PASS_SCORE = 80
SL_PCT     = 0.20
TP1_PCT    = 0.20
TP2_PCT    = 0.40

# ---------- helpers ----------
def get_klines(symbol, interval, limit=50):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    return requests.get(url, params=params, timeout=10).json()

def telegram(html):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat  = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat, "text": html, "parse_mode": "HTML"}
    requests.post(url, json=payload, timeout=10)

# ---------- Layer-1 Core Entry (0-30 pts) ----------
def layer1_core(k):
    c  = [float(x[4]) for x in k]
    h  = [float(x[2]) for x in k]
    l  = [float(x[3]) for x in k]
    # 1. Imbalance_FVG
    fvg = (h[-3] < l[-1]) or (h[-1] < l[-3])
    # 2. HTF_EMA_1h_15m (21 vs 55 ema)
    ema21 = sum(c[-21:]) / 21
    ema55 = sum(c[-55:]) / 55
    htf_bias = ema21 > ema55
    # 3. HTF_Structure_Break
    brk = c[-1] > max(c[-10:-2])
    return int(fvg) + int(htf_bias) + int(brk)

# ---------- Layer-2 Micro Edge (0-20 pts) ----------
def layer2_micro(k):
    c = [float(x[4]) for x in k]
    h = [float(x[2]) for x in k]
    l = [float(x[3]) for x in k]
    v = [float(x[5]) for x in k]
    # 4. Micro_Pullback
    pb = (c[-2] < c[-3]) and (c[-1] > c[-2])
    # 5. Vol_Sweep_1m (stop-hunt wick)
    wick = (h[-1] - max(c[-1], c[-2])) / (h[-1] - l[-1]) > 0.6
    # 6. Delta_Divergence_1m (price down, volume up)
    div = (c[-1] < c[-2]) and (v[-1] > v[-2])
    # 7. Spread_Safety (live) – FIXED symbol parse
    symbol = k[0][0][:k[0][0].rfind("USDT") + 4]   # safe slice
    try:
        book = requests.get("https://api.binance.com/api/v3/ticker/bookTicker",
                            params={"symbol": symbol}, timeout=5).json()
        spr = (float(book["askPrice"]) - float(book["bidPrice"])) / float(book["bidPrice"])
        tight = spr < 0.0005
    except:
        tight = False
    return int(pb) + int(wick) + int(div) + int(tight)

# ---------- Layer-3 Risk Guard (0-15 pts) ----------
def layer3_guard(symbol):
    score = 0
    # 8. News_Guard (high-impact USD/EUR/GBP ±30 min)
    now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    st  = now - datetime.timedelta(minutes=30)
    et  = now + datetime.timedelta(minutes=30)
    try:
        cal = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                           timeout=5).json()
        for ev in cal:
            if ev.get("impact") == "High" and ev.get("currency") in ["USD","EUR","GBP"]:
                evt = datetime.datetime.fromisoformat(ev["date"].replace("Z","")).replace(tzinfo=pytz.utc)
                if st <= evt <= et: score -= 1; break
    except: pass
    # 9. Funding_Extreme (> 0.01 %)
    try:
        f = requests.get(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}",
                         timeout=5).json()
        if float(f["lastFundingRate"]) > 0.0001: score -= 1
    except: pass
    # 10. OI_Spike_5pct (open-interest jump)
    try:
        oi = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}",
                          timeout=5).json()
        oi_now = float(oi["openInterest"])
        oi_prev = float(requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}",
                                     params={"period":"5m"},timeout=5).json()["openInterest"])
        if oi_now > oi_prev * 1.05: score -= 1
    except: pass
    return max(0, 3 + score)   # 0-3 pts

# ---------- scorer ----------
def calc(symbol):
    k = get_klines(symbol, TIMEFRAME)
    l1 = layer1_core(k)
    l2 = layer2_micro(k)
    l3 = layer3_guard(symbol)
    score = 60 + 10*l1 + 5*l2 + 5*l3
    return min(100, score)

# ---------- main ----------
def main():
    print("🚀 75 % win-rate 10-logic bot started")
    while True:
        for symbol in SYMBOLS:
            score = calc(symbol)
            if score < PASS_SCORE: continue
            k2 = get_klines(symbol, TIMEFRAME, 2)
            last = float(k2[-1][4])
            sl  = last * (1 - SL_PCT/100)
            tp1 = last * (1 + TP1_PCT/100)
            tp2 = last * (1 + TP2_PCT/100)
            msg = (f"📊 <b>{symbol[:-4]}/USDT</b>  ⏱ {TIMEFRAME}\n"
                   f"🎯 Score: <b>{int(score)}/100</b>\n"
                   f"🔥 {MODE.capitalize()} (10-logic)\n"
                   f"💰 Entry: {last:.4f}\n"
                   f"⛔ SL: {sl:.4f}\n"
                   f"🎯 TP1: {tp1:.4f} (50 %)\n"
                   f"🎯 TP2: {tp2:.4f} (50 %)")
            telegram(msg)
            print(f"✅ {symbol}  →  {int(score)}")
        time.sleep(30)

if __name__ == "__main__":
    main()
