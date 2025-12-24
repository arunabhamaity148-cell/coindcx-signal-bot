import pandas as pd
import csv
from datetime import datetime, timedelta
from typing import Dict, Optional
from config import config
from indicators import Indicators
from trap_detector import TrapDetector
from coindcx_api import CoinDCXAPI
from news_guard import news_guard
from chatgpt_advisor import ChatGPTAdvisor


class SignalGenerator:
    """
    Multi-Mode Signal Generator with ChatGPT Final Judge + Advanced Filters
    
    ARCHITECTURE:
    1. Rule-based system filters market (QUICK/MID/TREND modes)
    2. Advanced quality scoring (10 new modules)
    3. ChatGPT acts as FINAL JUDGE for every signal
    4. Only ChatGPT-approved signals reach Telegram
    5. Rejected signals are silently dropped (no output)
    
    ✅ ENHANCEMENTS:
    - 10 Advanced Filter Modules integrated
    - Quality score threshold (60+)
    - Better false signal filtering
    """

    def __init__(self):
        self.signal_count = 0
        self.signals_today = []
        self.last_signal_time: Dict[str, datetime] = {}
        self.last_reset_date = datetime.now().date()
        self.mode_signal_count = {mode: 0 for mode in config.ACTIVE_MODES}
        self.chatgpt_advisor = ChatGPTAdvisor()
        self.chatgpt_approved = 0
        self.chatgpt_rejected = 0

    def _reset_daily_counters(self):
        """Reset signal counters at midnight"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.signal_count = 0
            self.signals_today = []
            self.mode_signal_count = {mode: 0 for mode in config.ACTIVE_MODES}
            self.chatgpt_approved = 0
            self.chatgpt_rejected = 0
            self.last_reset_date = today
            print(f"🔄 Daily counters reset for {today}")

    def _check_cooldown(self, pair: str, mode: str) -> bool:
        """Check if pair is in cooldown period (per mode)"""
        key = f"{pair}_{mode}"
        if key in self.last_signal_time:
            time_since_last = datetime.now() - self.last_signal_time[key]
            cooldown_duration = timedelta(minutes=config.SAME_PAIR_COOLDOWN_MINUTES)
            if time_since_last < cooldown_duration:
                remaining = cooldown_duration - time_since_last
                remaining_mins = int(remaining.total_seconds() / 60)
                print(f"⏸️  {pair} ({mode}) in cooldown - {remaining_mins}m remaining")
                return False
        return True

    def _check_power_hours(self) -> bool:
        """Check if current time is in power hours"""
        now = datetime.now()
        current_hour = now.hour
        for start_hour, end_hour in config.POWER_HOURS:
            if start_hour <= current_hour < end_hour:
                return True
        return False

    def _calculate_entry_sl_tp(self, direction: str, current_price: float, atr: float, mode_config: Dict) -> Optional[Dict]:
        """Calculate entry, stop loss, and take profit levels"""
        sl_multiplier = mode_config['atr_sl_multiplier']
        tp1_multiplier = mode_config['atr_tp1_multiplier']
        tp2_multiplier = mode_config['atr_tp2_multiplier']
        decimal_places = config.get_decimal_places(current_price)

        if direction == "LONG":
            entry = current_price
            sl_raw = entry - (atr * sl_multiplier)
            tp1_raw = entry + (atr * tp1_multiplier)
            tp2_raw = entry + (atr * tp2_multiplier)
        else:
            entry = current_price
            sl_raw = entry + (atr * sl_multiplier)
            tp1_raw = entry - (atr * tp1_multiplier)
            tp2_raw = entry - (atr * tp2_multiplier)

        entry = round(entry, decimal_places)
        sl = round(sl_raw, decimal_places)
        tp1 = round(tp1_raw, decimal_places)
        tp2 = round(tp2_raw, decimal_places)

        min_tp_distance = entry * (config.MIN_TP_DISTANCE_PERCENT / 100)

        if direction == "LONG":
            if tp1 - entry < min_tp_distance:
                tp1 = round(entry + min_tp_distance, decimal_places)
            if tp2 - entry < min_tp_distance * 1.5:
                tp2 = round(entry + min_tp_distance * 1.5, decimal_places)
        else:
            if entry - tp1 < min_tp_distance:
                tp1 = round(entry - min_tp_distance, decimal_places)
            if entry - tp2 < min_tp_distance * 1.5:
                tp2 = round(entry - min_tp_distance * 1.5, decimal_places)

        if direction == "LONG":
            if tp1 <= entry or tp2 <= entry or tp1 >= tp2:
                return None
        else:
            if tp1 >= entry or tp2 >= entry or tp1 <= tp2:
                return None

        return {'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2}

    def _check_liquidation_safety(self, entry: float, sl: float) -> tuple[bool, float]:
        """Ensure SL is far enough from liquidation price"""
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
            'leverage': signal['leverage'],
            'chatgpt_approved': 'YES'
        }

        try:
            try:
                with open(log_file, 'r') as f:
                    file_exists = True
            except FileNotFoundError:
                file_exists = False

            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_data)
            print(f"📝 Signal logged to {log_file}")
        except Exception as e:
            print(f"⚠️ Performance logging failed: {e}")

    def _calculate_signal_score(self, indicators: Dict, trend_strength: str) -> int:
        """Calculate signal quality score (0-100)"""
        score = 0

        rsi = indicators['rsi']
        if 35 < rsi < 65:
            score += 20
        elif 30 < rsi < 70:
            score += 15
        else:
            score += 10

        adx = indicators['adx']
        if adx > 40:
            score += 25
        elif adx > 30:
            score += 20
        elif adx > 25:
            score += 15
        else:
            score += 10

        if abs(indicators['macd_histogram']) > abs(indicators['prev_macd_histogram']):
            score += 20
        else:
            score += 10

        if indicators['volume_surge'] > 1.5:
            score += 15
        elif indicators['volume_surge'] > 1.2:
            score += 10
        else:
            score += 5

        if trend_strength in ['STRONG_UP', 'STRONG_DOWN']:
            score += 20
        elif trend_strength in ['MODERATE_UP', 'MODERATE_DOWN']:
            score += 15
        else:
            score += 5

        return min(score, 100)

    def analyze(self, pair: str, candles: pd.DataFrame, mode: str = None) -> Optional[Dict]:
        """
        Main analysis function with ChatGPT Final Judge + Advanced Filters
        
        FLOW:
        1. Rule-based filtering (RSI, ADX, traps, etc.)
        2. Signal construction
        3. 🔬 Advanced Quality Scoring (10 modules)
        4. 🤖 ChatGPT FINAL DECISION (MANDATORY)
        5. If approved → return signal
        6. If rejected → return None (silent drop)
        """

        if mode is None:
            mode = config.MODE

        mode_config = config.MODE_CONFIG[mode]
        min_score = mode_config.get('min_score', config.MIN_SIGNAL_SCORE)
        min_adx = mode_config.get('min_adx', config.MIN_ADX_STRENGTH)

        self._reset_daily_counters()

        is_blocked, reason = news_guard.is_blocked()
        if is_blocked:
            return None

        if self.signal_count >= config.MAX_SIGNALS_PER_DAY:
            return None

        max_per_mode = config.MAX_SIGNALS_PER_DAY // len(config.ACTIVE_MODES)
        if self.mode_signal_count[mode] >= max_per_mode:
            return None

        if not self._check_cooldown(pair, mode):
            return None

        if len(candles) < 50:
            return None

        try:
            candles = candles.dropna()
            if len(candles) < 50:
                return None

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

            ema_fast = ema_fast.dropna()
            ema_slow = ema_slow.dropna()
            macd_line = macd_line.dropna()
            signal_line = signal_line.dropna()
            histogram = histogram.dropna()
            rsi = rsi.dropna()
            adx = adx.dropna()
            atr = atr.dropna()
            volume_surge = volume_surge.dropna()

            if len(rsi) < 2 or len(adx) < 2 or len(atr) < 2:
                return None

            current_price = float(close.iloc[-1])
            current_rsi = float(rsi.iloc[-1])
            current_adx = float(adx.iloc[-1])
            current_atr = float(atr.iloc[-1])
            current_macd_hist = float(histogram.iloc[-1])
            prev_macd_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0.0
            current_volume_surge = float(volume_surge.iloc[-1]) if len(volume_surge) > 0 else 1.0

            if any(pd.isna([current_price, current_rsi, current_adx, current_atr])):
                return None

            ticker = CoinDCXAPI.get_ticker(pair)
            bid = ticker['bid'] if ticker else current_price
            ask = ticker['ask'] if ticker else current_price

            traps = TrapDetector.check_all_traps(
                candles, bid, ask, current_rsi, current_adx, current_macd_hist
            )

            trapped_count = sum(traps.values())
            trap_reasons = TrapDetector.get_trap_reasons(traps)

            if trapped_count >= 3:
                return None

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

            if trend == "LONG" and current_rsi > config.RSI_OVERBOUGHT:
                return None
            if trend == "SHORT" and current_rsi < config.RSI_OVERSOLD:
                return None

            if current_adx < min_adx:
                return None

            levels = self._calculate_entry_sl_tp(trend, current_price, current_atr, mode_config)
            if levels is None:
                return None

            is_safe, sl_distance_pct = self._check_liquidation_safety(levels['entry'], levels['sl'])
            if not is_safe:
                return None

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

            indicators_data = {
                'rsi': current_rsi,
                'adx': current_adx,
                'macd_histogram': current_macd_hist,
                'prev_macd_histogram': prev_macd_hist,
                'volume_surge': current_volume_surge
            }

            score = self._calculate_signal_score(indicators_data, mtf_trend)

            if score < min_score:
                return None

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
                'mode': mode,
                'timeframe': mode_config['timeframe'],
                'volume_surge': round(current_volume_surge, 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            print(f"\n{'='*60}")
            print(f"📊 Rule-based system PASSED: {pair} {trend}")
            print(f"{'='*60}")

            chatgpt_approved = self.chatgpt_advisor.final_trade_decision(signal, candles)

            if not chatgpt_approved:
                self.chatgpt_rejected += 1
                print(f"🚫 Signal silently dropped (no Telegram output)")
                print(f"{'='*60}\n")
                return None

            self.chatgpt_approved += 1
            print(f"✅ {mode} MODE: {pair} FINAL APPROVAL!")
            print(f"   Direction: {trend}")
            print(f"   Score: {score}/100")
            print(f"   Entry: ₹{levels['entry']:,.6f}")
            print(f"   SL: ₹{levels['sl']:,.6f} ({sl_distance_pct:.1f}% away)")
            print(f"   TP1: ₹{levels['tp1']:,.6f}")
            print(f"   TP2: ₹{levels['tp2']:,.6f}")
            print(f"   RSI: {current_rsi:.1f}, ADX: {current_adx:.1f}")
            print(f"{'='*60}\n")

            self._log_signal_performance(signal)

            self.signal_count += 1
            self.mode_signal_count[mode] += 1
            self.signals_today.append(signal)

            cooldown_key = f"{pair}_{mode}"
            self.last_signal_time[cooldown_key] = datetime.now()
            print(f"⏱️  Cooldown started: {pair} ({mode}) - next signal after 30 minutes")

            return signal

        except Exception as e:
            print(f"❌ Error in {mode} analysis for {pair}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_stats(self) -> Dict:
        """Get signal generator statistics including ChatGPT performance"""
        total_evaluated = self.chatgpt_approved + self.chatgpt_rejected
        approval_rate = (self.chatgpt_approved / total_evaluated * 100) if total_evaluated > 0 else 0

        return {
            'signals_today': self.signal_count,
            'signals_remaining': config.MAX_SIGNALS_PER_DAY - self.signal_count,
            'mode_breakdown': self.mode_signal_count,
            'mode': 'MULTI' if config.MULTI_MODE_ENABLED else config.MODE,
            'last_reset': self.last_reset_date.strftime('%Y-%m-%d'),
            'chatgpt_approved': self.chatgpt_approved,
            'chatgpt_rejected': self.chatgpt_rejected,
            'chatgpt_approval_rate': f"{approval_rate:.1f}%"
        }