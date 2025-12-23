import pandas as pd
import csv
from datetime import datetime, timedelta
from typing import Dict, Optional
from config import config
from indicators import Indicators
from trap_detector import TrapDetector
from coindcx_api import CoinDCXAPI
from news_guard import news_guard

class SignalGenerator:
    """
    Multi-Mode Signal Generator
    Analyzes markets in QUICK (5m), MID (15m), and TREND (1h) timeframes
    """

    def __init__(self):
        self.signal_count = 0
        self.signals_today = []
        self.last_signal_time: Dict[str, datetime] = {}
        self.last_reset_date = datetime.now().date()
        
        # ✅ Track signals per mode
        self.mode_signal_count = {mode: 0 for mode in config.ACTIVE_MODES}

    def _reset_daily_counters(self):
        """Reset signal counters at midnight"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.signal_count = 0
            self.signals_today = []
            self.mode_signal_count = {mode: 0 for mode in config.ACTIVE_MODES}
            self.last_reset_date = today
            print(f"🔄 Daily counters reset for {today}")

    def _check_cooldown(self, pair: str, mode: str) -> bool:
        """Check if pair is in cooldown period (per mode)"""
        key = f"{pair}_{mode}"
        if key in self.last_signal_time:
            time_since_last = datetime.now() - self.last_signal_time[key]
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

    def _calculate_entry_sl_tp(self, direction: str, current_price: float, atr: float, mode_config: Dict) -> Dict:
        """Calculate entry, stop loss, and take profit levels using mode-specific multipliers"""

        sl_multiplier = mode_config['atr_sl_multiplier']
        tp1_multiplier = mode_config['atr_tp1_multiplier']
        tp2_multiplier = mode_config['atr_tp2_multiplier']

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

    def _check_liquidation_safety(self, entry: float, sl: float) -> tuple[bool, float]:
        """
        Ensure SL is far enough from liquidation price
        Returns: (is_safe, distance_percentage)
        """
        distance_pct = abs(entry - sl) / entry * 100
        MIN_DISTANCE = config.LIQUIDATION_BUFFER * 100
        is_safe = distance_pct >= MIN_DISTANCE
        return is_safe, distance_pct

    def _log_signal_performance(self, signal: Dict):
        """Log signal to CSV for performance tracking"""
        if not hasattr(config, 'TRACK_PERFORMANCE') or not config.TRACK_PERFORMANCE:
            return

        log_file = getattr(config, 'PERFORMANCE_LOG_FILE', 'signal_performance.csv')

        log_data = {
            'timestamp': signal.get('timestamp', datetime.now().isoformat()),
            'pair': signal['pair'],
            'direction': signal['direction'],
            'mode': signal.get('mode', 'UNKNOWN'),
            'timeframe': signal.get('timeframe', 'UNKNOWN'),
            'entry': signal['entry'],
            'sl': signal['sl'],
            'tp1': signal['tp1'],
            'tp2': signal['tp2'],
            'score': signal['score'],
            'rsi': signal['rsi'],
            'adx': signal['adx'],
            'leverage': signal['leverage']
        }

        try:
            # Check if file exists
            try:
                with open(log_file, 'r') as f:
                    file_exists = True
            except FileNotFoundError:
                file_exists = False

            # Write to CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_data)

            print(f"📝 Signal logged to {log_file}")
        except Exception as e:
            print(f"⚠️ Performance logging failed: {e}")

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

    def analyze(self, pair: str, candles: pd.DataFrame, mode: str = None) -> Optional[Dict]:
        """
        Main analysis function - now supports specific mode
        
        Args:
            pair: Trading pair
            candles: Historical OHLCV data
            mode: Analysis mode (QUICK/MID/TREND) - if None, uses config.MODE
        
        Returns:
            Signal dict or None
        """
        
        # ✅ Use provided mode or default
        if mode is None:
            mode = config.MODE
        
        # ✅ Get mode-specific configuration
        mode_config = config.MODE_CONFIG[mode]
        min_score = mode_config.get('min_score', config.MIN_SIGNAL_SCORE)
        min_adx = mode_config.get('min_adx', config.MIN_ADX_STRENGTH)

        # Reset daily counters if needed
        self._reset_daily_counters()

        # ⚠️ NEWS GUARD CHECK - PRIORITY #1
        is_blocked, reason = news_guard.is_blocked()
        if is_blocked:
            return None

        # Check daily limit (global)
        if self.signal_count >= config.MAX_SIGNALS_PER_DAY:
            return None
        
        # ✅ Check per-mode limit
        max_per_mode = config.MAX_SIGNALS_PER_DAY // len(config.ACTIVE_MODES)
        if self.mode_signal_count[mode] >= max_per_mode:
            return None

        # ✅ Check cooldown (per pair per mode)
        if not self._check_cooldown(pair, mode):
            return None

        # Check if enough data
        if len(candles) < 50:
            return None

        try:
            # Clean data - remove NaN
            candles = candles.dropna()

            if len(candles) < 50:
                return None

            # Calculate all indicators
            close = candles['close']
            high = candles['high']
            low = candles['low']
            volume = candles['volume']

            ema_fast = Indicators.ema(close, mode_config['ema_fast'])
            ema_slow = Indicators.ema(close, mode_config['ema_slow'])
            macd_line, signal_line, histogram = Indicators.macd(close)
            rsi = Indicators.rsi(close)
            adx, plus_di, minus_di = Indicators.adx(high, low, close)
            atr = Indicators.atr(high, low, close)
            volume_surge = Indicators.volume_surge(volume)

            # Drop NaN from indicators
            ema_fast = ema_fast.dropna()
            ema_slow = ema_slow.dropna()
            macd_line = macd_line.dropna()
            signal_line = signal_line.dropna()
            histogram = histogram.dropna()
            rsi = rsi.dropna()
            adx = adx.dropna()
            atr = atr.dropna()
            volume_surge = volume_surge.dropna()

            # Check if we have enough data after cleaning
            if len(rsi) < 2 or len(adx) < 2 or len(atr) < 2:
                return None

            # Current values with NaN check
            current_price = float(close.iloc[-1])
            current_rsi = float(rsi.iloc[-1])
            current_adx = float(adx.iloc[-1])
            current_atr = float(atr.iloc[-1])
            current_macd_hist = float(histogram.iloc[-1])
            prev_macd_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0.0
            current_volume_surge = float(volume_surge.iloc[-1]) if len(volume_surge) > 0 else 1.0

            # Validate values
            if any(pd.isna([current_price, current_rsi, current_adx, current_atr])):
                return None

            # Get ticker for spread check
            ticker = CoinDCXAPI.get_ticker(pair)
            bid = ticker['bid'] if ticker else current_price
            ask = ticker['ask'] if ticker else current_price

            # Trap detection
            traps = TrapDetector.check_all_traps(
                candles, bid, ask, current_rsi, current_adx, current_macd_hist
            )

            # Count traps
            trapped_count = sum(traps.values())
            trap_reasons = TrapDetector.get_trap_reasons(traps)

            # If 3+ traps triggered, skip (too risky)
            if trapped_count >= 3:
                return None

            # Determine trend direction
            trend = None
            if (ema_fast.iloc[-1] > ema_slow.iloc[-1] and 
                macd_line.iloc[-1] > signal_line.iloc[-1] and
                plus_di.iloc[-1] > minus_di.iloc[-1]):
                trend = "LONG"
            elif (ema_fast.iloc[-1] < ema_slow.iloc[-1] and 
                  macd_line.iloc[-1] < signal_line.iloc[-1] and
                  minus_di.iloc[-1] > plus_di.iloc[-1]):
                trend = "SHORT"
            else:
                return None

            # ✅ RSI filter with relaxed thresholds
            if trend == "LONG" and current_rsi > config.RSI_OVERBOUGHT:
                return None
            if trend == "SHORT" and current_rsi < config.RSI_OVERSOLD:
                return None

            # ✅ ADX filter (use mode-specific threshold)
            if current_adx < min_adx:
                return None

            # ✅ Calculate levels with mode-specific config
            levels = self._calculate_entry_sl_tp(trend, current_price, current_atr, mode_config)

            # ✅ Liquidation check
            is_safe, sl_distance_pct = self._check_liquidation_safety(levels['entry'], levels['sl'])

            if not is_safe:
                return None

            # If 1-2 traps, ask ChatGPT
            if trapped_count > 0:
                chatgpt_approved = False

                try:
                    from chatgpt_advisor import ChatGPTAdvisor
                    advisor = ChatGPTAdvisor()

                    signal_preview = {
                        'pair': pair,
                        'direction': trend,
                        'entry': levels['entry'],
                        'sl': levels['sl'],
                        'rsi': current_rsi,
                        'adx': current_adx,
                        'trapped_count': trapped_count,
                        'trap_reasons': trap_reasons
                    }

                    decision = advisor.validate_signal_with_traps(signal_preview)

                    if not decision.get('approved', False):
                        return None
                    else:
                        chatgpt_approved = True

                except Exception as e:
                    # Fallback: Auto-approve 1-2 traps if ChatGPT fails
                    if trapped_count <= 2:
                        chatgpt_approved = True
                    else:
                        return None

                if not chatgpt_approved:
                    return None

            # Get multi-timeframe trend (if possible)
            try:
                candles_5m = CoinDCXAPI.get_candles(pair, '5m', 50)
                candles_15m = CoinDCXAPI.get_candles(pair, '15m', 50)
                candles_1h = CoinDCXAPI.get_candles(pair, '1h', 50)

                if not candles_5m.empty and not candles_15m.empty and not candles_1h.empty:
                    mtf_trend = Indicators.mtf_trend(
                        candles_5m['close'],
                        candles_15m['close'],
                        candles_1h['close']
                    )
                else:
                    mtf_trend = "UNKNOWN"
            except:
                mtf_trend = "UNKNOWN"

            # Calculate signal score
            indicators_data = {
                'rsi': current_rsi,
                'adx': current_adx,
                'macd_histogram': current_macd_hist,
                'prev_macd_histogram': prev_macd_hist,
                'volume_surge': current_volume_surge
            }

            score = self._calculate_signal_score(indicators_data, mtf_trend)

            # ✅ Use mode-specific minimum score
            if score < min_score:
                return None

            # ✅ SUCCESS! Generate signal
            print(f"✅ {mode} MODE: {pair} SIGNAL APPROVED!")
            print(f"   Direction: {trend}")
            print(f"   Score: {score}/100")
            print(f"   Entry: ₹{levels['entry']:,.2f}")
            print(f"   SL: ₹{levels['sl']:,.2f} ({sl_distance_pct:.1f}% away)")
            print(f"   RSI: {current_rsi:.1f}, ADX: {current_adx:.1f}")

            # Build signal
            signal = {
                'pair': pair,
                'direction': trend,
                'entry': levels['entry'],
                'sl': levels['sl'],
                'tp1': levels['tp1'],
                'tp2': levels['tp2'],
                'leverage': mode_config['leverage'],
                'score': score,
                'rsi': round(current_rsi, 1),
                'adx': round(current_adx, 1),
                'mtf_trend': mtf_trend,
                'mode': mode,  # ✅ Added
                'timeframe': mode_config['timeframe'],  # ✅ Added
                'volume_surge': round(current_volume_surge, 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # ✅ LOG PERFORMANCE
            self._log_signal_performance(signal)

            # ✅ Update counters (global + per-mode)
            self.signal_count += 1
            self.mode_signal_count[mode] += 1
            self.signals_today.append(signal)
            self.last_signal_time[f"{pair}_{mode}"] = datetime.now()

            return signal

        except Exception as e:
            print(f"❌ Error in {mode} analysis for {pair}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_stats(self) -> Dict:
        """Get signal generator statistics"""
        return {
            'signals_today': self.signal_count,
            'signals_remaining': config.MAX_SIGNALS_PER_DAY - self.signal_count,
            'mode_breakdown': self.mode_signal_count,
            'mode': 'MULTI' if config.MULTI_MODE_ENABLED else config.MODE,
            'last_reset': self.last_reset_date.strftime('%Y-%m-%d')
        }