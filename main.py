import ccxt
import pandas as pd
import time
from helpers import fetch_ohlcv, check_signal
import telegram

BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

bot = telegram.Bot(token=BOT_TOKEN)

exchange = ccxt.binance()

def send(msg):
    bot.sendMessage(chat_id=CHAT_ID, text=msg)

def load_symbols():
    df = pd.read_csv("coins.csv")
    return df["symbol"].tolist()

symbols = load_symbols()

send("🚀 Bot Started Successfully!")

while True:
    try:
        for symbol in symbols:
            df = fetch_ohlcv(exchange, symbol)
            if df is None:
                continue

            signal = check_signal(df)
            if signal:
                send(f"{signal} Signal detected on {symbol}")

        time.sleep(10)

    except Exception as e:
        send(f"Error: {e}")
        time.sleep(5)
