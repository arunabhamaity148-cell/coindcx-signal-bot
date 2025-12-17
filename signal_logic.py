from config import Config
from indicators import TechnicalIndicators
from patterns import CandlestickPatterns
from smart_logic import SmartMoneyLogic


class SignalGenerator:

    def __init__(self):
        self.config = Config()
        self.ind = TechnicalIndicators()
        self.patterns = CandlestickPatterns()
        self.smart = SmartMoneyLogic()

    # ---------------- 5m ANALYSIS ----------------
    def analyze_5m(self, candles):
        if len(candles) < 100:
            return None

        candles = candles[-100:]
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]

        return {
            "price": closes[-1],
            "ema9": self.ind.calculate_ema(closes, 9),
            "ema21": self.ind.calculate_ema(closes, 21),
            "ema50": self.ind.calculate_ema(closes, 50),
            "rsi": self.ind.calculate_rsi(closes),
            "macd": self.ind.calculate_macd(closes),
            "atr": self.ind.calculate_atr(highs, lows, closes),
            "adx": self.ind.calculate_adx(highs, lows, closes),
        }

    # ---------------- SCORE ----------------
    def score_signal(self, a, direction):
        score = 0
        reasons = []

        if direction == "LONG" and a["ema9"] > a["ema21"] > a["ema50"]:
            score += 20; reasons.append("EMA uptrend")

        if direction == "SHORT" and a["ema9"] < a["ema21"] < a["ema50"]:
            score += 20; reasons.append("EMA downtrend")

        if a["rsi"]:
            if direction == "LONG" and 30 < a["rsi"] < 55:
                score += 15; reasons.append("RSI support")
            if direction == "SHORT" and 45 < a["rsi"] < 70:
                score += 15; reasons.append("RSI resistance")

        macd, macd_sig, hist = a["macd"]
        if direction == "LONG" and macd > macd_sig:
            score += 15; reasons.append("MACD bullish")
        if direction == "SHORT" and macd < macd_sig:
            score += 15; reasons.append("MACD bearish")

        if a["adx"] and a["adx"] > 20:
            score += 10; reasons.append("ADX strength")

        return score, reasons

    # ---------------- 15m MTF ----------------
    def mtf_confirm(self, candles_15m, direction):
        closes = [c['close'] for c in candles_15m]
        ema21 = self.ind.calculate_ema(closes, 21)
        ema50 = self.ind.calculate_ema(closes, 50)

        if not ema21 or not ema50:
            return False

        return ema21 > ema50 if direction == "LONG" else ema21 < ema50

    # ---------------- FINAL SIGNAL ----------------
    def generate_signal(self, market, candles_5m, candles_15m):
        a = self.analyze_5m(candles_5m)
        if not a or not a["atr"]:
            return None

        long_score, long_r = self.score_signal(a, "LONG")
        short_score, short_r = self.score_signal(a, "SHORT")

        if long_score < self.config.MIN_SIGNAL_SCORE and short_score < self.config.MIN_SIGNAL_SCORE:
            return None

        if long_score > short_score:
            direction, score, reasons = "LONG", long_score, long_r
        else:
            direction, score, reasons = "SHORT", short_score, short_r

        if not self.mtf_confirm(candles_15m, direction):
            return None

        entry = a["price"]
        atr = a["atr"]

        sl = entry - atr * 1.3 if direction == "LONG" else entry + atr * 1.3
        tp = entry + atr * 2.3 if direction == "LONG" else entry - atr * 2.3

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