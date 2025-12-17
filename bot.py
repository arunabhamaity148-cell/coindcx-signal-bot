import logging
import time
import schedule
import requests
import pandas as pd
from telegram import Bot
from config import *
from signal_logic import analyze_market, can_send_signal, mark_signal_sent

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TELEGRAM_BOT_TOKEN)

def send_msg(text):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        logging.error(e)

def get_data(market):
    try:
        r = requests.get("https://public.coindcx.com/market_data/candles", params={"pair": market, "interval": CANDLE_INTERVAL, "limit": CANDLE_LIMIT}, timeout=10)
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df.sort_values("time").reset_index(drop=True)
    except:
        return None

def check_market(market):
    if not can_send_signal(market):
        return
    df = get_data(market)
    if df is None:
        return
    result = analyze_market(df, market)
    if not result:
        return
    if result['score'] < MIN_SIGNAL_SCORE:
        return
    msg = result['direction'] + " SIGNAL
Market: " + market + "
Score: " + str(result['score']) + "
Entry: " + str(result['entry']) + "
SL: " + str(result['sl']) + "
TP1: " + str(result['tp1'])
    send_msg(msg)
    mark_signal_sent(market)
    logging.info("Signal sent for " + market)

def scan():
    logging.info("Scanning markets")
    for m in MARKETS:
        try:
            check_market(m.strip())
            time.sleep(2)
        except Exception as e:
            logging.error(e)

def run():
    send_msg("Bot started")
    scan()
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(scan)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    run()