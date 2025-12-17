"""
CoinDCX-ONLY Edge Bot  (TDS-hedged, no external exchange)
Runs on spot + perpetual inside CoinDCX
"""

import asyncio, os, time, math, logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import aiohttp, numpy as np
from collections import deque

API_KEY    = os.getenv("COINDCX_API_KEY")
SECRET     = os.getenv("COINDCX_SECRET")
TG_BOT     = os.getenv("TG_BOT")
TG_CHAT    = os.getenv("TG_CHAT")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ---------- CONFIG ----------
COINS        = ["BTC", "ETH", "MATIC", "SOL", "XRP", "ADA"]
MIN_SCORE    = 55
SCAN_SEC     = 25
PRICE_DEQUE  = 120
SPREAD_DEQUE = 100
FUND_WARN    = 0.05          # 5 bps
MAX_SLIP     = 0.25/100      # 0.25 %
PARTICIP     = 0.15          # 15 % of top-3 volume
CAPITAL      = float(os.getenv("CAPITAL", 50000))
RISK         = 0.01          # 1 % per trade
# ----------------------------

class CoinDCXClient:
    def __init__(self, key: str, secret: str):
        self.key = key
        self.s = secret
        self.base = "https://api.coindcx.com"
        self.session : Optional[aiohttp.ClientSession] = None

    async def _sess(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        if self.session: await self.session.close()

    async def get(self, path: str, params: dict = None) -> dict:
        sess = await self._sess()
        async with sess.get(self.base + path, params=params) as r:
            return await r.json()

    async def post(self, path: str, payload: dict) -> dict:
        sess = await self._sess()
        async with sess.post(self.base + path, json=payload) as r:
            return await r.json()

    async def markets(self) -> List[str]:
        res = await self.get("/exchange/v1/markets")
        return [m for m in res if m.endswith(("INR", "INRT"))]

    async def ticker(self, market: str) -> dict:
        res = await self.get("/exchange/ticker", {"market": market})
        return res[0] if res else {}

    async def orderbook(self, market: str) -> dict:
        return await self.get("/exchange/v1/orderbook", {"market": market})

    async def funding(self, perp: str) -> float:
        res = await self.get("/exchange/v1/funding", {"market": perp})
        return float(res[0].get("funding_rate", 0)) if res else 0.0

    async def place(self, mkt: str, side: str, price: float, qty: float, ord_type: str) -> dict:
        body = {
            "market": mkt,
            "side": side,
            "order_type": ord_type,
            "price": price,
            "quantity": qty,
            "timestamp": int(time.time()*1000)
        }
        return await self.post("/exchange/v1/orders/create", body)

    async def pos(self, perp: str) -> dict:
        res = await self.get("/exchange/v1/positions", {"market": perp})
        return res[0] if res else {}

# ---------- DATA ----------
class Tracker:
    def __init__(self, sz: int):
        self.prices : Dict[str, deque] = {}
        self.vols   : Dict[str, deque] = {}
        self.spreads: Dict[str, deque] = {}
        self.books  : Dict[str, dict]  = {}
        self.sz = sz

    def add(self, m: str, p: float, v: float, book: dict):
        if m not in self.prices:
            self.prices[m] = deque(maxlen=self.sz)
            self.vols[m]   = deque(maxlen=self.sz)
            self.spreads[m]= deque(maxlen=self.sz)
        self.prices[m].append(p)
        self.vols[m].append(v)
        self.books[m] = book
        if book.get("bids") and book.get("asks"):
            b, a = book["bids"][0][0], book["asks"][0][0]
            self.spreads[m].append((a-b)/b)

    def ready(self, m: str, n: int) -> bool:
        return len(self.prices.get(m, [])) >= n

    def last(self, m: str) -> float:
        return self.prices[m][-1]

    def book(self, m: str) -> dict:
        return self.books.get(m, {})

    def spread_pct(self, m: str) -> float:
        return self.spreads[m][-1] if self.spreads.get(m) else 0.0

    def vol_spike(self, m: str) -> Tuple[bool, str]:
        if not self.ready(m, 30): return False, ""
        v = list(self.vols[m])
        cur, avg = v[-1], np.mean(v[-20:-1])
        if avg == 0: return False, ""
        if cur > avg * 3.5:
            if v[-2] < avg * 1.5:
                return True, "FADE"
        return False, ""

    def imb(self, m: str) -> float:
        b = self.books[m]
        if not b.get("bids") or not b.get("asks"): return 0.0
        bidV = sum(x[1] for x in b["bids"][:3])
        askV = sum(x[1] for x in b["asks"][:3])
        t = bidV + askV
        return (bidV - askV) / t if t else 0.0

    def top3_vol(self, m: str) -> float:
        b = self.books[m]
        if not b.get("bids"): return 0.0
        return sum(x[1]*x[0] for x in b["bids"][:3])

# ---------- ANALYTICS ----------
def score(m: str, tr: Tracker) -> Tuple[int, str]:
    s, direction = 0, "NONE"
    # 1. spread quality
    sp = tr.spread_pct(m)
    if sp < np.percentile(list(tr.spreads[m]), 25):
        s += 8
    # 2. volume spike fade
    fade, txt = tr.vol_spike(m)
    if fade:
        s += 12
        direction = "CONTRA"
    # 3. orderbook imbalance
    imb = tr.imb(m)
    if imb > 0.18:
        s += 10
        direction = "BUY"
    elif imb < -0.18:
        s += 10
        direction = "SELL"
    # 4. time
    hr = datetime.now().hour
    if 10 <= hr <= 11 or 20 <= hr <= 22:
        s += 6
    return s, direction

# ---------- RISK ----------
def qty(m: str, side: str, entry: float, tr: Tracker) -> float:
    risk_amt = CAPITAL * RISK
    atr = np.std(list(tr.prices[m])[-12:])
    sl_dist = max(0.005, 2.5 * atr)
    risk_qty = risk_amt / sl_dist
    # participation cap
    max_qty = PARTICIP * tr.top3_vol(m) / entry
    return min(risk_qty, max_qty)

# ---------- EXECUTION ----------
async def hedge_enter(client: CoinDCXClient, mkt_spot: str, side: str, qty_spot: float, entry: float):
    perp = mkt_spot.replace("INR", "USDT") + "-PERPETUAL"
    # funding check
    fund = await client.funding(perp)
    if abs(fund) > FUND_WARN and (
        (side == "BUY" and fund < 0) or (side == "SELL" and fund > 0)
    ):
        log.warning(f"skip {mkt_spot} fund={fund}")
        return
    # place both legs
    if side == "BUY":
        await client.place(mkt_spot, "buy",  entry, qty_spot, "limit")
        await client.place(perp,    "sell", entry, qty_spot, "market")
    else:
        await client.place(mkt_spot, "sell", entry, qty_spot, "limit")
        await client.place(perp,    "buy",  entry, qty_spot, "market")
    log.info(f"HEDGE {side} {mkt_spot} @{entry} Q={qty_spot}")

# ---------- NOTIFIER ----------
async def notify(text: str):
    if not TG_BOT: return
    url = f"https://api.telegram.org/bot{TG_BOT}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as s:
        await s.post(url, data=payload)

# ---------- BOT ----------
class EdgeBot:
    def __init__(self):
        self.cli = CoinDCXClient(API_KEY, SECRET)
        self.tr  = Tracker(PRICE_DEQUE)
        self.seen: set[str] = set()

    async def update(self):
        mkts = await self.cli.markets()
        for m in mkts:
            t = await self.cli.ticker(m)
            if not t: continue
            p = float(t["last_price"])
            v = float(t["volume"])
            b = await self.cli.orderbook(m)
            self.tr.add(m, p, v, b)

    async def scan(self):
        await self.update()
        for m in self.tr.prices:
            if not self.tr.ready(m, 30): continue
            scr, dir_ = score(m, self.tr)
            if scr < MIN_SCORE or dir_ == "NONE": continue
            key = f"{m}{dir_}{datetime.now().hour}"
            if key in self.seen: continue
            self.seen.add(key)
            entry = self.tr.last(m)
            q = qty(m, dir_, entry, self.tr)
            await hedge_enter(self.cli, m, dir_, q, entry)
            msg = f"🎯 *CoinDCX EDGE*\n{m}  {dir_}\nEntry ₹{entry:.2f}\nQty {q:.4f}\nScore {scr}"
            await notify(msg)

    async def run(self):
        log.info("CoinDCX EdgeBot start")
        await notify("🚀 Bot started")
        # warm-up
        for _ in range(3):
            await self.update()
            await asyncio.sleep(20)
        while True:
            try:
                await self.scan()
                await asyncio.sleep(SCAN_SEC)
            except Exception as e:
                log.exception("loop")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = EdgeBot()
    asyncio.run(bot.run())
