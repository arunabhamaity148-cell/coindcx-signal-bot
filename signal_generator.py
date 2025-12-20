import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from config import config
from indicators import Indicators
from trap_detector import TrapDetector
from coindcx_api import CoinDCXAPI
from news_guard import news_guard

class SignalGenerator:
    """
    Core signal generation engine
    Analyzes market data and generates high-quality trading signals
    """

    def __init__(self):
        self.signal_count = 0
        self.signals_today = []
        self.last_signal_time: Dict[str, datetime] = {}
        self.mode_config = config.MODE_CONFIG[config.MODE]
        self.last_reset_date = datetime.now().date()

    def _reset_daily_counters(self):
        """Reset signal counters at midnight"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.signal_count = 0
            self.signals_today = []
            self.last_reset_date = today
            print(f"🔄 Daily counters reset for {today}")

    def _check_cooldown(self, pair: str) -> bool:
        """Check if pair is in cooldown period"""
        if pair in self.last_signal_time:
            time_since_last = datetime.now() - self.last_signal_time[pair]
            if time_since_last < timedelta(minutes=config.COOLDOWN_MINUTES):
                return False  # Still in cooldown
        return True

    def _check_power_hours(self) -> bool:
        """Check if current time is in power hours (best trading times)"""
        now = datetime.now()
        current_hour = now.hour

        for start_hour, end_hour in config.POWER_HOURS:
            if start_hour <= current_hour < end_hour:
                return True

        return False

    def _calculate_entry_sl_tp(self, direction: str, current_price: float, atr: float) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""

        sl_multiplier = self.mode_config['atr_sl_multiplier']
        tp1_multiplier = self.mode_config['atr_tp1_multiplier']
        tp2_multiplier = self.mode_config['atr_tp2_multiplier']

        if direction == "LONG":
            entry = current_price
            sl = entry - (atr * sl_multiplier)
            tp1 = entry + (atr * tp1_multiplier)
            tp2 = entry + (atr * tp2_multiplier)
        else:  # SHORT
            entry = current_price
            sl = entry + (atr * sl_multiplier)
            tp1 = entry - (atr * tp1_multiplier)
            tp2 = entry - (atr * tp2_multiplier)

        return {
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'tp1': round(tp1, 2),
            'tp2': round(tp2, 2)
        }

    def _check_liquidation_safety(self, entry: float, sl: float) -> bool:
        """Ensure SL is far enough from liquidation price"""
        distance = abs(entry - sl) / entry
        return distance >= config.LIQUIDATION_BUFFER

    def _calculate_signal_score(self, indicators: Dict, trend_strength: str) -> int:
        """
        Calculate signal quality score (0-100)
        Higher score = better signal
        """
        score = 0

        # RSI score (max 20 points)
        rsi = indicators['rsi']
        if 35 < rsi < 65:
            score += 20
        elif 30 < rsi < 70:
            score += 15
        else:
            score += 10

        # ADX score (max 25 points)
        adx = indicators['adx']
        if adx > 40:
            score += 25
        elif adx > 30:
            score += 20
        elif adx > 25:
            score += 15
        else:
            score += 10

        # MACD momentum (max 20 points)
        if abs(indicators['macd_histogram']) > abs(indicators['prev_macd_histogram']):
            score += 20  # Momentum increasing
        else:
            score += 10

        # Volume surge (max 15 points)
        if indicators['volume_surge'] > 1.5:
            score += 15
        elif indicators['volume_surge'] > 1.2:
            score += 10
        else:
            score += 5

        # Multi-timeframe alignment (max 20 points)
        if trend_strength in ['STRONG_UP', 'STRONG_DOWN']:
            score += 20
        elif trend_strength in ['MODERATE_UP', 'MODERATE_DOWN']:
            score += 15
        else:
            score += 5

        return min(score, 100)

    def analyze(self, pair: str, candles: pd.DataFrame) -> Optional[Dict]:
        """
        Main analysis function with robust data validation
        
        Args:
            pair: Trading pair
            candles: Historical OHLCV data
        
        Returns:
            Signal dict or None
        """

        # Reset daily counters if needed
        self._reset_daily_counters()

        # ⚠️ NEWS GUARD CHECK - PRIORITY #1
        is_blocked, reason = news_guard.is_blocked()
        if is_blocked:
            return None

        # Check daily limit
        if self.signal_count >= config.MAX_SIGNALS_PER_DAY:
            return None

        # Check cooldown
        if not self._check_cooldown(pair):
            return None

        # ✅ CRITICAL: Check minimum data requirement
        min_required = 250  # Enough for EMA 200 + buffer
        if len(candles) < min_required:
            print(f"⚠️ {pair}: Need {min_required} candles, got {len(candles)}")
            return None

        try:
            # ✅ Clean data - remove NaN and infinite values
            candles = candles.replace([np.inf, -np.inf], np.nan)
            candles = candles.dropna()

            if len(candles) < min_required:
                print(f"⚠️ {pair}: After cleaning, only {len(candles)} candles remain")
                return None

            # ✅ Extract and validate series
            close = candles['close'].astype(float)
            high = candles['high'].astype(float)
            low = candles['low'].astype(float)
            volume = candles['volume'].astype(float)

            # Ensure no NaN in base data
            if close.isna().any() or high.isna().any() or low.isna().any():
                print(f"⚠️ {pair}: NaN values in OHLC data")
                return None

            # ✅ Calculate all indicators
            ema_fast = Indicators.ema(close, self.mode_config['ema_fast'])
            ema_slow = Indicators.ema(close, self.mode_config['ema_slow'])
            macd_line, signal_line, histogram = Indicators.macd(close)
            rsi = Indicators.rsi(close)
            adx, plus_di, minus_di = Indicators.adx(high, low, close)
            atr = Indicators.atr(high, low, close)
            volume_surge = Indicators.volume_surge(volume)

            # ✅ Helper function to safely extract last valid value
            def get_last_valid(series, name, default=None):
                """Get last valid non-NaN value"""
                valid = series.dropna()
                if len(valid) == 0:
                    if default is None:
                        raise ValueError(f"{name} has no valid values")
                    return default
                return float(valid.iloc[-1])

            def get_nth_last_valid(series, n, name, default=None):
                """Get nth-last valid non-NaN value"""
                valid = series.dropna()
                if len(valid) <= n:
                    if default is None:
                        raise ValueError(f"{name} has insufficient values")
                    return default
                return float(valid.iloc[-(n+1)])

            # ✅ Extract current values safely
            current_price = get_last_valid(close, "close")
            current_rsi = get_last_valid(rsi, "RSI")
            current_adx = get_last_valid(adx, "ADX")
            current_atr = get_last_valid(atr, "ATR")
            current_macd_hist = get_last_valid(histogram, "MACD histogram")
            prev_macd_hist = get_nth_last_valid(histogram, 1, "prev MACD histogram", default=0.0)
            current_volume_surge = get_last_valid(volume_surge, "volume surge", default=1.0)
            
            current_ema_fast = get_last_valid(ema_fast, "EMA fast")
            current_ema_slow = get_last_valid(ema_slow, "EMA slow")
            current_macd_line = get_last_valid(macd_line, "MACD line")
            current_signal_line = get_last_valid(signal_line, "Signal line")
            current_plus_di = get_last_valid(plus_di, "Plus DI")
            current_minus_di = get_last_valid(minus_di, "Minus DI")

            # ✅ Validate all critical values
            critical_values = [
                current_price, current_rsi, current_adx, current_atr,
                current_ema_fast, current_ema_slow
            ]
            
            if any(pd.isna(critical_values)) or current_price <= 0 or current_atr <= 0:
                print(f"⚠️ {pair}: Invalid indicator values detected")
                return None

            # Get ticker for spread check
            ticker = CoinDCXAPI.get_ticker(pair)
            bid = ticker.get('bid', current_price) if ticker else current_price
            ask = ticker.get('ask', current_price) if ticker else current_price

            # ✅ Trap detection
            traps = TrapDetector.check_all_traps(
                candles, bid, ask, current_rsi, current_adx, current_macd_hist
            )

            if TrapDetector.is_trapped(traps):
                trap_reasons = TrapDetector.get_trap_reasons(traps)
                print(f"⚠️ {pair} - Traps: {', '.join(trap_reasons)}")
                return None

            # ✅ Determine trend direction
            trend = None
            if (current_ema_fast > current_ema_slow and 
                current_macd_line > current_signal_line and
                current_plus_di > current_minus_di):
                trend = "LONG"
            elif (current_ema_fast < current_ema_slow and 
                  current_macd_line < current_signal_line and
                  current_minus_di > current_plus_di):
                trend = "SHORT"
            else:
                return None  # No clear trend

            # ✅ RSI filter
            if trend == "LONG" and current_rsi > 70:
                return None  # Overbought
            if trend == "SHORT" and current_rsi < 30:
                return None  # Oversold

            # ✅ ADX filter (minimum trend strength)
            if current_adx < config.MIN_ADX_STRENGTH:
                return None

            # ✅ Calculate entry/SL/TP
            levels = self._calculate_entry_sl_tp(trend, current_price, current_atr)

            # ✅ Liquidation safety check
            if not self._check_liquidation_safety(levels['entry'], levels['sl']):
                print(f"⚠️ {pair} - Liquidation risk high")
                return None

            # ✅ Multi-timeframe trend (optional, with error handling)
            mtf_trend = "UNKNOWN"
            try:
                c5m = CoinDCXAPI.get_candles(pair, '5m', 100)
                c15m = CoinDCXAPI.get_candles(pair, '15m', 100)
                c1h = CoinDCXAPI.get_candles(pair, '1h', 100)

                if not c5m.empty and not c15m.empty and not c1h.empty:
                    if len(c5m) >= 10 and len(c15m) >= 10 and len(c1h) >= 10:
                        mtf_trend = Indicators.mtf_trend(
                            c5m['close'], c15m['close'], c1h['close']
                        )
            except Exception as e:
                print(f"⚠️ {pair}: MTF analysis failed - {e}")

            # ✅ Calculate signal score
            indicators_data = {
                'rsi': current_rsi,
                'adx': current_adx,
                'macd_histogram': current_macd_hist,
                'prev_macd_histogram': prev_macd_hist,
                'volume_surge': current_volume_surge
            }

            score = self._calculate_signal_score(indicators_data, mtf_trend)

            # ✅ Minimum score filter
            min_score = 60 if config.MODE == 'TREND' else 50
            if score < min_score:
                return None

            # ✅ Build signal
            signal = {
                'pair': pair,
                'direction': trend,
                'entry': levels['entry'],
                'sl': levels['sl'],
                'tp1': levels['tp1'],
                'tp2': levels['tp2'],
                'leverage': self.mode_config['leverage'],
                'score': score,
                'rsi': round(current_rsi, 1),
                'adx': round(current_adx, 1),
                'mtf_trend': mtf_trend,
                'mode': config.MODE,
                'volume_surge': round(current_volume_surge, 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # ✅ Update counters
            self.signal_count += 1
            self.signals_today.append(signal)
            self.last_signal_time[pair] = datetime.now()

            return signal

        except Exception as e:
            print(f"❌ {pair} analysis error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_stats(self) -> Dict:
        """Get signal generator statistics"""
        return {
            'signals_today': self.signal_count,
            'signals_remaining': config.MAX_SIGNALS_PER_DAY - self.signal_count,
            'mode': config.MODE,
            'last_reset': self.last_reset_date.strftime('%Y-%m-%d')
        }