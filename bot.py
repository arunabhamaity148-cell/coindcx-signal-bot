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

def send_telegram(msg):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
        logging.info("Message sent")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def get_candles(market):
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
        logging.error(f"Fetch error: {e}")
        return None

def build_message(market, direction, final_score, result, smart):
    emoji = "GREEN" if direction == "LONG" else "RED"
    parts = [emoji + " " + direction + " SIGNAL", "", "Market: " + market]
    if smart:
        parts.append("Smart: " + str(int(smart['smart_score'])) + "/100")
        if result:
            parts.append("Technical: " + str(int(result['score'])) + "/100")
        parts.append("Combined: " + str(int(final_score)) + "/100")
        parts.append("Confidence: " + smart['confidence'])
        parts.append("Regime: " + smart['regime'])
        parts.append("")
    else:
        parts.append("Score: " + str(int(final_score)) + "/100")
        parts.append("")
    if result:
        parts.append("Entry: " + str(round(result['entry'], 4)))
        parts.append("Stop Loss: " + str(round(result['sl'], 4)))
        parts.append("TP1: " + str(round(result['tp1'], 4)))
        parts.append("TP2: " + str(round(result['tp2'], 4)))
        parts.append("RR: 1:" + str(round(result['rr'], 2)))
        parts.append("")
        parts.append("RSI: " + str(round(result['rsi'], 1)))
        parts.append("MACD: " + str(round(result['macd_hist'], 6)))
        parts.append("ADX: " + str(round(result['adx'], 1)))
        if result.get('pattern'):
            parts.append("Pattern: " + result['pattern'])
    if smart and smart.get('signals'):
        parts.append("")
        parts.append("Smart Signals:")
        for sig in smart['signals'][:5]:
            parts.append("- " + sig)
    parts.append("")
    parts.append("Time: " + time.strftime('%H:%M:%S IST'))
    return "
".join(parts)

def process_market(market):
    logging.info(f"Analyzing {market}")
    if not can_send_signal(market):
        return
    df = get_candles(market)
    if df is None:
        return
    result = analyze_market(df, market)
    smart = None
    if ENABLE_SMART_LOGIC:
        try:
            smart = get_smart_signals(df, market)
        except:
            pass
    if result and smart:
        final_score = result['score'] * WEIGHT_REGULAR_SCORE + smart['smart_score'] * WEIGHT_SMART_SCORE
    elif result:
        final_score = result['score']
    elif smart:
        final_score = smart['smart_score']
    else:
        return
    if final_score < MIN_SIGNAL_SCORE:
        return
    direction = smart['direction'] if smart else result['direction']
    msg = build_message(market, direction, final_score, result, smart)
    send_telegram(msg)
    mark_signal_sent(market)
    logging.info(f"{market}: {direction} signal sent")

def job():
    logging.info("Scan started")
    for market in MARKETS:
        try:
            process_market(market.strip())
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error: {e}")
    logging.info("Scan complete")

def main():
    send_telegram("Bot Started - Markets: " + str(len(MARKETS)))
    logging.info("Bot initialized")
    job()
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()