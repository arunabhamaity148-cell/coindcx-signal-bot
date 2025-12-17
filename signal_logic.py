from config import Config
from indicators import TechnicalIndicators
from patterns import CandlestickPatterns
from smart_logic import SmartMoneyLogic

class SignalGenerator:

    def __init__(self):
        self.config = Config()
        self.indicators = TechnicalIndicators()
        self.patterns = CandlestickPatterns()
        self.smart = SmartMoneyLogic()

    # ================= ANALYSIS =================
    def analyze_market(self, candles):
        if not candles or len(candles) < 100:
            return None

        candles = candles[-100:]
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]

        ema_9 = self.indicators.calculate_ema(closes, self.config.EMA_SHORT)
        ema_21 = self.indicators.calculate_ema(closes, self.config.EMA_MEDIUM)
        ema_50 = self.indicators.calculate_ema(closes, self.config.EMA_LONG)

        rsi = self.indicators.calculate_rsi(closes)
        macd, macd_sig, hist = self.indicators.calculate_macd(closes)
        atr = self.indicators.calculate_atr(highs, lows, closes)
        adx = self.indicators.calculate_adx(highs, lows, closes)

        return {
            "price": closes[-1],
            "ema_9": ema_9,
            "ema_21": ema_21,
            "ema_50": ema_50,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_sig,
            "macd_hist": hist,
            "atr": atr,
            "adx": adx
        }

    # ================= SCORE =================
    def calculate_score(self, a, direction):
        score = 0
        reasons = []

        if direction == "LONG":
            if a["ema_9"] > a["ema_21"] > a["ema_50"]:
                score += 20; reasons.append("EMA uptrend")
        else:
            if a["ema_9"] < a["ema_21"] < a["ema_50"]:
                score += 20; reasons.append("EMA downtrend")

        if a["rsi"]:
            if direction == "LONG" and 30 < a["rsi"] < 55:
                score += 15; reasons.append("RSI support")
            if direction == "SHORT" and 45 < a["rsi"] < 70:
                score += 15; reasons.append("RSI resistance")

        if a["macd"] and a["macd_signal"]:
            if direction == "LONG" and a["macd"] > a["macd_signal"]:
                score += 15; reasons.append("MACD bullish")
            if direction == "SHORT" and a["macd"] < a["macd_signal"]:
                score += 15; reasons.append("MACD bearish")

        if a["adx"] and a["adx"] > 20:
            score += 10; reasons.append("ADX strength")

        return score, reasons

    # ================= MTF CONFIRM =================
    def mtf_confirm(self, candles_15m, direction):
        closes = [c['close'] for c in candles_15m]
        ema_21 = self.indicators.calculate_ema(closes, 21)
        ema_50 = self.indicators.calculate_ema(closes, 50)

        if not ema_21 or not ema_50:
            return False

        return ema_21 > ema_50 if direction == "LONG" else ema_21 < ema_50

    # ================= FINAL SIGNAL =================
    def generate_signal(self, market, candles_5m, candles_15m):
        analysis = self.analyze_market(candles_5m)
        if not analysis or not analysis["atr"]:
            return None

        long_score, long_reasons = self.calculate_score(analysis, "LONG")
        short_score, short_reasons = self.calculate_score(analysis, "SHORT")

        if long_score < self.config.MIN_SIGNAL_SCORE and short_score < self.config.MIN_SIGNAL_SCORE:
            return None

        if long_score > short_score:
            direction, score, reasons = "LONG", long_score, long_reasons
        else:
            direction, score, reasons = "SHORT", short_score, short_reasons

        # 🔒 MTF FILTER
        if not self.mtf_confirm(candles_15m, direction):
            return None

        entry = analysis["price"]
        atr = analysis["atr"]

        if direction == "LONG":
            sl = entry - atr * 1.3
            tp = entry + atr * 2.3
        else:
            sl = entry + atr * 1.3
            tp = entry - atr * 2.3

        rr = abs(tp - entry) / abs(entry - sl)
        if rr < self.config.MIN_RR_RATIO:
            return None

        return {
            "market": market,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "score": score,
            "rr": rr,
            "reasons": reasons
        }