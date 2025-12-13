import logging
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class InstitutionalScanner:
    def __init__(self, connector, logic_evaluator, config):
        self.cdx = connector
        self.logic = logic_evaluator
        self.min_score = config.get("min_signal_score", 65)
        self.symbols = connector.symbols
        self.timeframes = ["5m", "15m", "1h"]

    def scan_pair(self, symbol, timeframe):
        try:
            df = self.cdx.get_ohlcv(symbol, timeframe, 200)
            if df is None or len(df) < 60:
                return None

            orderbook = self.cdx.get_orderbook(symbol)
            if not orderbook:
                return None

            result = self.logic.evaluate_all_logics(
                df=df,
                orderbook=orderbook,
                funding_rate=0,
                oi_history=[0]*20,
                recent_trades=[],
                fear_greed_index=50,
                news_times=[],
                liquidation_clusters=[]
            )

            if not result["trade_allowed"]:
                return None

            if result["final_score"] < self.min_score:
                return None

            price = df["close"].iloc[-1]

            side = "BUY" if result["price_action"]["structure"] == "bullish" else "SELL"

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "side": side,
                "entry": price,
                "logic_score": result["final_score"],
                "timestamp": datetime.now()
            }

        except Exception as e:
            logger.error(f"{symbol} scan error: {e}")
            return None

    def scan_all(self):
        signals = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            tasks = []
            for s in self.symbols:
                for tf in self.timeframes:
                    tasks.append(ex.submit(self.scan_pair, s, tf))
            for t in tasks:
                r = t.result()
                if r:
                    signals.append(r)
        return signals