"""
CoinDCX Smart Trading Bot - Main Entry Point
"""
import logging
import time
import schedule
import requests
import pandas as pd
from telegram import Bot
from config import *
from signal_logic import analyze_market, can_send_signal, mark_signal_sent
from smart_logic import get_smart_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

bot = Bot(token=TELEGRAM_BOT_TOKEN)

def send_telegram(msg: str):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
        logging.info("✅ Message sent")
    except Exception as e:
        logging.error(f"❌ Telegram error: {e}")

def get_candles(market: str):
    url = "https://public.coindcx.com/market_data/candles"
    params = {"pair": market, "interval": CANDLE_INTERVAL, "limit": CANDLE_LIMIT}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        for col in ["close", "open", "high", "low", "volume"]:
            df[col] = df[col].astype(float)
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"❌ Fetch error {market}: {e}")
        return None

def process_market(market: str):
    logging.info(f"🔍 Analyzing {market}...")
    if not can_send_signal(market):
        logging.info(f"⏳ {market} in cooldown/limit")
        return
    df = get_candles(market)
    if df is None:
        return
    result = analyze_market(df, market)
    smart = None
    if ENABLE_SMART_LOGIC:
        try:
            smart = get_smart_signals(df, market)
        except Exception as e:
            logging.error(f"Smart logic error: {e}")
            smart = None
    if result and smart:
        final_score = (result['score'] * WEIGHT_REGULAR_SCORE + smart['smart_score'] * WEIGHT_SMART_SCORE)
    elif result:
        final_score = result['score']
    elif smart:
        final_score = smart['smart_score']
    else:
        logging.info(f"📊 {market}: No signal")
        return
    if final_score < MIN_SIGNAL_SCORE:
        logging.info(f"📊 {market}: Score {final_score:.0f} < threshold")
        return
    direction = smart['direction'] if smart else result['direction']
    emoji = "🟢" if direction == "LONG" else "🔴"
    lines = []
    lines.append(f"{emoji} *{direction} SIGNAL*")
    lines.append("")
    lines.append(f"💎 Market: `{market}`")
    if smart:
        lines.append(f"🧠 Smart: *{smart['smart_score']:.0f}/100*")
        if result:
            lines.append(f"📊 Technical: *{result['score']:.0f}/100*")
        lines.append(f"🎯 Combined: *{final_score:.0f}/100*")
        lines.append(f"⚡ Confidence: *{smart['confidence']}*")
        lines.append(f"📈 Regime: `{smart['regime']}`")
        lines.append("")
    else:
        lines.append(f"📊 Score: *{final_score:.0f}/100*")
        lines.append("")
    if result:
        lines.append(f"📍 Entry: `{result['entry']:.4f}`")
        lines.append(f"🛑 Stop Loss: `{result['sl']:.4f}`")
        lines.append(f"🎯 TP1 (50%): `{result['tp1']:.4f}`")
        lines.append(f"🎯 TP2 (50%): `{result['tp2']:.4f}`")
        lines.append(f"⚖️ R:R: `1:{result['rr']:.2f}`")
        lines.append("")
        lines.append(f"📊 *Indicators:*")
        lines.append(f"RSI: `{result['rsi']:.1f}`")
        lines.append(f"MACD: `{result['macd_hist']:.6f}`")
        lines.append(f"ADX: `{result['adx']:.1f}`")
        lines.append(f"EMA: `{result['ema_fast']:.2f}`/`{result['ema_slow']:.2f}`")
        if result.get('pattern'):
            lines.append(f"Pattern: `{result['pattern']}`")
        if result.get('divergence'):
            lines.append(f"Div: `{result['divergence']}`")
    if smart and smart.get('signals'):
        lines.append("")
        lines.append(f"🧠 *Smart Signals:*")
        for sig in smart['signals'][:6]:
            lines.append(f"• {sig}")
    lines.append("")
    lines.append(f"⏰ {time.strftime('%H:%M:%S IST')}")
    msg = "
".join(lines)
    send_telegram(msg)
    mark_signal_sent(market)
    logging.info(f"✅ {market}: {direction} signal sent (score: {final_score:.0f})")

def job():
    logging.info("=" * 60)
    logging.info("🚀 Market scan started")
    for market in MARKETS:
        try:
            process_market(market.strip())
            time.sleep(2)
        except Exception as e:
            logging.error(f"❌ Error {market}: {e}")
    logging.info("✅ Scan complete")
    logging.info("=" * 60)

def main():
    markets_str = ', '.join([f"`{m}`" for m in MARKETS[:4]])
    smart_status = "ON" if ENABLE_SMART_LOGIC else "OFF"
    startup_lines = []
    startup_lines.append("🤖 *CoinDCX Smart Bot Started*")
    startup_lines.append("")
    startup_lines.append(f"📊 Markets: `{len(MARKETS)}`")
    startup_lines.append(f"⏱️ Interval: `{CANDLE_INTERVAL}`")
    startup_lines.append(f"📈 Min Score: `{MIN_SIGNAL_SCORE}`")
    startup_lines.append(f"🔄 Check: Every `{CHECK_INTERVAL_MINUTES}min`")
    startup_lines.append(f"⏳ Cooldown: `{COOLDOWN_MINUTES}min`")
    startup_lines.append(f"📅 Daily Limit: `{MAX_SIGNALS_PER_DAY}`")
    startup_lines.append(f"🧠 Smart Logic: `{smart_status}`")
    startup_lines.append("")
    startup_lines.append(f"Pairs: {markets_str}")
    startup = "
".join(startup_lines)
    send_telegram(startup)
    logging.info("🤖 Bot initialized")
    job()
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()