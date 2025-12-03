import asyncio, os, json, logging
from helpers import WS, Exchange, features, calc_tp_sl, position_size, send_telegram, ai_review_ml, regime, CFG

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

async def run():
    ws = WS(); asyncio.create_task(ws.run()); await asyncio.sleep(3)
    ex = Exchange()
    while True:
        for sym in CFG["pairs"]:
            f = await features(sym)
            if not f: continue
            reg = await regime(sym)
            score = round((max(f["rsi"], 50) - 50) / 5 + (f["imb"] + 1) * 2 + f["sweep"], 1)  # quick score
            side = "long" if score >= 7.5 else "short" if score <= 3.0 else "none"
            if side == "none": continue
            atr = 0.4  # dummy ATR (replace with redis ohlcv)
            tp, sl = calc_tp_sl(f["last"], atr, side)
            ai = await ai_review_ml(sym, "QUICK", side, score, f"rsi={f['rsi']:.1f},imb={f['imb']:.2f}", f["spread"] < 0.1, f["depth_usd"] > 1e6, True)
            if not ai["allow"]: continue
            qty = position_size(CFG["equity"], f["last"], sl)
            msg = f"🎯 <b>{sym}</b> {side.upper()}  score={score}  qty={qty:.3f}  TP={tp}  SL={sl}  conf={ai['confidence']}%"
            logging.info(msg); await send_telegram(msg)
            # paper-limit order (uncomment for live)
            # await ex.limit(sym, side, qty, f["last"])
        await asyncio.sleep(5)

if __name__ == "__main__":
    try: asyncio.run(run())
    except KeyboardInterrupt: logging.info("stopped")
