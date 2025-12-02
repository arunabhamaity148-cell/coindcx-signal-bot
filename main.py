import os, asyncio, aiohttp, time, hmac, hashlib
from dotenv import load_dotenv
load_dotenv()
from helpers import (
    final_score, calc_tp_sl, cooldown_ok, update_cd,
    trade_side, format_signal, auto_leverage, ema,
    calc_exhaustion, calc_fvg, calc_ob
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("BINANCE_API_KEY")
API_SEC = os.getenv("BINANCE_SECRET")
INTERVAL = 60
MODE_THRESH = {"quick": 55, "mid": 62, "trend": 70}


WATCHLIST = [
"BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
"LINKUSDT","DOTUSDT","MATICUSDT","TRXUSDT","LTCUSDT","ATOMUSDT","FILUSDT","INJUSDT",
"OPUSDT","ARBUSDT","AAVEUSDT","SANDUSDT","DYDXUSDT","NEARUSDT","FTMUSDT","RUNEUSDT",
"PYTHUSDT","TIAUSDT","SEIUSDT","RNDRUSDT","WLDUSDT","MINAUSDT","JTOUSDT","GALAUSDT",
"SUIUSDT","FLOWUSDT","CHZUSDT","CTSIUSDT","EGLDUSDT","APTUSDT","LDOUSDT","MKRUSDT",
"CRVUSDT","SKLUSDT","ENSUSDT","XLMUSDT","XTZUSDT","CFXUSDT","KAVAUSDT","PEPEUSDT","SHIBUSDT"
]

async def send(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as s:
        await s.post(url, json={"chat_id": CHAT, "text": msg, "parse_mode": "HTML"})

def sign(query_string):
    return hmac.new(API_SEC.encode(), query_string.encode(), hashlib.sha256).hexdigest()

async def retry_get(session, url, params=None, headers=None, max_retry=5):
    for attempt in range(max_retry):
        try:
            async with session.get(url, params=params, headers=headers, timeout=10) as r:
                data = await r.json()
                if isinstance(data, dict) and data.get("code"):
                    print("Binance error", data, "retry", attempt + 1)
                    await asyncio.sleep(2 ** attempt)
                    continue
                return data
        except Exception as e:
            print("Retry ex", e, attempt + 1)
            await asyncio.sleep(2 ** attempt)
    return None

async def get_ema(session, symbol, interval, length=20):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": str(length)}
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    k = await retry_get(session, url, params, headers)
    if not isinstance(k, list) or len(k) < length:
        return None
    try:
        closes = [float(x[4]) for x in k]
    except (ValueError, TypeError, IndexError):
        return None
    return ema(closes, length)

async def get_btc_1h_change(session):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": "2"}
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    k = await retry_get(session, url, params, headers)
    if not isinstance(k, list) or len(k) < 2:
        return 0
    try:
        return (float(k[-1][4]) - float(k[-2][4])) / float(k[-2][4]) * 100
    except Exception:
        return 0

async def get_spread(session, symbol):
    url = "https://api.binance.com/api/v3/ticker/bookTicker"
    params = {"symbol": symbol}
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    d = await retry_get(session, url, params, headers)
    if not isinstance(d, dict):
        return 0
    try:
        bid = float(d.get("bidPrice", 0))
        ask = float(d.get("askPrice", 0))
        return (ask - bid) / ask if ask else 0
    except Exception:
        return 0

async def get_depth_imbalance(session, symbol):
    url = "https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol, "limit": "20"}
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    d = await retry_get(session, url, params, headers)
    if not isinstance(d, dict):
        return False
    try:
        bids = d.get("bids", [])[:5]
        asks = d.get("asks", [])[:5]
        bid_vol = sum(float(q) for _, q in bids)
        ask_vol = sum(float(q) for _, q in asks)
        ratio = bid_vol / ask_vol if ask_vol else 1
        return ratio > 3 or ratio < 0.33
    except Exception:
        return False

async def get_data(session, symbol):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1m", "limit": "30"}
        headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
        kl = await retry_get(session, url, params, headers)
        if not isinstance(kl, list) or len(kl) < 3:
            return None
        try:
            price = float(kl[-1][4])
            prev = float(kl[-2][4])
            vol_1m = float(kl[-1][5])
        except Exception as e:
            print("Bad kline", symbol, e)
            return None
        ema_15m = await get_ema(session, symbol, "15m", 20) or price
        ema_1h = await get_ema(session, symbol, "1h", 20) or price
        ema_4h = await get_ema(session, symbol, "4h", 20) or price
        ema_8h = await get_ema(session, symbol, "4h", 40) or price
        spread = await get_spread(session, symbol)
        btc_ch = await get_btc_1h_change(session)
        liq_wall = await get_depth_imbalance(session, symbol)

        live = {
            "price": price, "ema_15m": ema_15m, "ema_1h": ema_1h, "ema_4h": ema_4h, "ema_8h": ema_8h,
            "trend": "bull_break" if price >= prev else "reject", "trend_strength": abs(price - prev) / price,
            "micro_pb": price > prev and price > ema_15m,
            "exhaustion": calc_exhaustion(kl[-1]), "fvg": calc_fvg(kl), "orderblock": calc_ob(kl),
            "vol_1m": vol_1m, "vol_5m": sum(float(x[5]) for x in kl),
            "adr_ok": True, "atr_expanding": True,
            "spread": spread, "btc_calm": abs(btc_ch) < 1.5, "kill_primary": False,
            "wick_ratio": 1.0, "liq_wall": liq_wall, "liq_bend": False,
            "wall_shift": False, "speed_imbalance": False, "absorption": False,
            "liquidation_sweep": False, "slippage": False, "liq_dist": 0.4, "taker_pressure": False
        }
        return live, price, prev
    except Exception as e:
        print("get_data ex", symbol, e)
        return None

async def handle_symbol(session, symbol):
    data = await get_data(session, symbol)
    if not data:
        print("Skip no data", symbol)
        return
    live, price, prev = data
    if live["spread"] > 0.05 or not live["btc_calm"]:
        print("Risk skip", symbol)
        return
    side = trade_side(price, prev, live["ema_1h"])
    if not side:
        print("No dir", symbol)
        return
    for mode in ("quick", "mid", "trend"):
        score = final_score(live)
        if score >= MODE_THRESH[mode] and cooldown_ok(symbol):
            tp, sl = calc_tp_sl(price, mode)
            lev = auto_leverage(score, mode)
            update_cd(symbol)
            await send(format_signal(symbol, side, mode, price, score, tp, sl, live["liq_dist"], lev))
            print("Signal", symbol, side, mode, score, lev)

async def scanner():
    print("🚀 REAL-30-LOGIC AUTH-BOT RUNNING")
    async with aiohttp.ClientSession() as s:
        while True:
            await asyncio.gather(*[handle_symbol(s, sym) for sym in WATCHLIST])
            await asyncio.sleep(INTERVAL)

if __name__ == "__main__":
    asyncio.run(scanner())
