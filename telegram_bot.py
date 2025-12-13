import requests

class TradingBotNotifier:
    def __init__(self, token, chat_id):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id

    def send(self, text):
        requests.post(self.url, json={
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        })

    def notify_trade(self, trade):
        msg = (
            f"📊 <b>{trade['symbol']}</b>\n"
            f"Side: {trade['side']}\n"
            f"Entry: {trade['entry']:.2f}\n"
            f"Logic: {trade['logic_score']:.1f}%"
        )
        self.send(msg)