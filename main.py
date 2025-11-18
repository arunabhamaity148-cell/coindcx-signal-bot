import time
import logging
from helpers import (
    get_exchange,
    load_symbols,
    fetch_ohlcv_safe,
    evaluate_signal,
    build_message
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")

# Telegram
import requests
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": msg}
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram error: {e}")


def main():
    logging.info("Starting coindcx-signal-bot (MINIMAL ULTRA-STABLE VERSION)")

    exchange = get_exchange()
    symbols = load_symbols()

    logging.info(f"Loaded coins: {symbols}")

    while True:
        for symbol in symbols:
            logging.info(f"Checking: {symbol}")

            ohlcv = fetch_ohlcv_safe(exchange, symbol)

            if ohlcv:
                signal = evaluate_signal(symbol, ohlcv)

                if signal:
                    msg = build_message(signal)
                    send_telegram(msg)
                    logging.info(f"SIGNAL SENT FOR {symbol}")

            time.sleep(2)

        logging.info("Loop finished. Sleeping 5 seconds…")
        time.sleep(5)


if __name__ == "__main__":
    main()