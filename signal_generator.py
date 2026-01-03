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
from signal_explainer import SignalExplainer
from telegram_notifier import TelegramNotifier
from entry_filters import EntryFilters


class SignalGenerator:
    """
    Enhanced Signal Generator with Entry Filters
    """
    
    def __init__(self):
        self.signal_count = 0
        self.signals_today = []
        self.last_signal_time: Dict[str, datetime] = {}
        self.last_signal_price: Dict[str, float] = {}
        self.last_signal_direction: Dict[str, str] = {}
        self.last_reset_date = datetime.now().date()
        self.mode_signal_count = {mode: 0 for mode in config.ACTIVE_MODES}
        self.coin_signal_count: Dict[str, int] = {}
        self.coin_mode_signal_count: Dict[str, int] = {}
        self.active_trends: Dict[str, Dict] = {}
        self.chatgpt_advisor = ChatGPTAdvisor()
        self.chatgpt_approved = 0
        self.chatgpt_rejected = 0
        self.active_positions: Dict[str, Dict] = {}
        self.coin_total_signals: Dict[str, int] = {}
        self.coin_active_mode: Dict[str, str] = {}
        self.score_buckets = {'high': 0, 'medium': 0, 'low': 0}

    def get_stats(self) -> dict:
        return {
            "signals_today": self.signal_count,
            "mode_breakdown": dict(self.mode_signal_count),
            "coin_breakdown": dict(self.coin_signal_count),
            "chatgpt_approved": self.chatgpt_approved,
            "chatgpt_rejected": self.chatgpt_rejected,
            "signals_today_list": self.signals_today.copy(),
            "score_buckets": dict(self.score_buckets)
        }

    def _reset_daily_counters(self):
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.signal_count = 0
            self.signals_today = []
            self.mode_signal_count = {mode: 0 for mode in config.ACTIVE_MODES}
            self.coin_signal_count = {}
            self.coin_mode_signal_count = {}
            self.coin_total_signals = {}
            self.coin_active_mode = {}
            self.last_signal_price = {}
            self.last_signal_direction = {}
            self.active_trends = {}
            self.chatgpt_approved = 0
            self.chatgpt_rejected = 0
            self.score_buckets = {'high': 0, 'medium': 0, 'low': 0}
            self.last_reset_date = today
            print(f"🔄 Daily counters reset for {today}")

    def _check_cooldown(self, pair: str, mode: str) -> bool:
        key = f"{pair}_{mode}"
        if key in self.last_signal_time:
            time_since_last = datetime.now() - self.last_signal_time[key]
            if mode == 'TREND':
                cooldown_duration = timedelta(hours=2)
            elif mode == 'MID':
                cooldown_duration = timedelta(minutes=45)
            else:
                cooldown_duration = timedelta(minutes=config.SAME_PAIR_COOLDOWN_MINUTES)
            if time_since_last < cooldown_duration:
                remaining = cooldown_duration - time_since_last
                remaining_mins = int(remaining.total_seconds() / 60)
                print(f"⏸️  {pair} ({mode}) in cooldown - {remaining_mins}m remaining")
                return False
        return True

    def _check_coin_daily_limit(self, pair: str, mode: str) -> bool:
        LIMITS = {'TREND': 2, 'MID': 3, 'QUICK': 4, 'SCALP': 4}
        max_signals = LIMITS.get(mode, 3)
        coin = pair.split('USDT')[0]
        coin_mode_key = f"{coin}_{mode}"
        if coin_mode_key not in self.coin_mode_signal_count:
            self.coin_mode_signal_count[coin_mode_key] = 0
        if self.coin_mode_signal_count[coin_mode_key] >= max_signals:
            print(f"❌ BLOCKED: {pair} | {mode} coin limit ({max_signals})")
            return False
        return True

    def _check_global_coin_exposure(self, pair: str) -> bool:
        coin = pair.split('USDT')[0]
        if coin not in self.coin_total_signals:
            self.coin_total_signals[coin] = 0
        MAX_COIN_EXPOSURE = 2
        if self.coin_total_signals[coin] >= MAX_COIN_EXPOSURE:
            print(f"❌ BLOCKED: {pair} | Global coin exposure limit ({MAX_COIN_EXPOSURE})")
            return False
        return True

    def _check_one_coin_one_mode(self, pair: str, mode: str) -> bool:
        coin = pair.split('USDT')[0]
        if coin in self.coin_active_mode:
            existing_mode = self.coin_active_mode[coin]
            last_signal_key = f"{coin}_last_signal"
            if last_signal_key in self.last_signal_time:
                time_since = (datetime.now() - self.last_signal_time[last_signal_key]).total_seconds() / 3600
                if time_since < 4:
                    print(f"❌ BLOCKED: {pair} | Already has {existing_mode} signal ({time_since:.1f}h ago)")
                    return False
        return True

    def _check_direction_flip_cooldown(self, pair: str, direction: str) -> bool:
        coin = pair.split('USDT')[0]
        if coin in self.last_signal_direction:
            if self.last_signal_direction[coin] != direction:
                key = f"{coin}_last_signal"
                if key in self.last_signal_time:
                    time_since = datetime.now() - self.last_signal_time[key]
                    if time_since < timedelta(hours=4):
                        remaining = timedelta(hours=4) - time_since
                        remaining_mins = int(remaining.total_seconds() / 60)
                        print(f"❌ BLOCKED: {pair} | Direction flip cooldown ({remaining_mins}m remaining)")
                        return False
        return True

    def _check_score_bucket_limit(self, score: int) -> bool:
        if score >= 85:
            bucket = 'high'
            limit = 3
        elif score >= 75:
            bucket = 'medium'
            limit = 5
        else:
            bucket = 'low'
            limit = 4
        if self.score_buckets[bucket] >= limit:
            print(f"❌ BLOCKED: Score bucket '{bucket}' limit reached ({limit})")
            return False
        return True

    def _check_active_trend(self, pair: str, trend: str, mode: str, mtf_trend: str, 
                           current_price: float, ema_fast: float, ema_slow: float, 
                           macd_hist: float) -> bool:
        trend_key = f"{pair}_{mode}_{trend}"
        if trend_key in self.active_trends:
            active = self.active_trends[trend_key]
            stored_mtf = active.get('mtf_trend', 'UNKNOWN')
            if stored_mtf in ['STRONG_UP', 'STRONG_DOWN']:
                ttl_hours = 6
            elif stored_mtf in ['MODERATE_UP', 'MODERATE_DOWN']:
                ttl_hours = 4
            else:
                ttl_hours = 3
            time_elapsed = (datetime.now() - active['started']).total_seconds() / 3600
            if time_elapsed > ttl_hours:
                del self.active_trends[trend_key]
                print(f"🔓 TREND expired: {pair} {mode} {trend} ({ttl_hours}h TTL)")
                return True
            price_change = (current_price - active['entry_price']) / active['entry_price'] * 100
            structural_break = False
            if trend == 'LONG':
                if price_change < -2.5:
                    structural_break = True
                if ema_fast < ema_slow:
                    structural_break = True
            else:
                if price_change > 2.5:
                    structural_break = True
                if ema_fast > ema_slow:
                    structural_break = True
            if structural_break:
                del self.active_trends[trend_key]
                print(f"🔓 TREND invalidated: {pair} {mode} {trend} (structure broken)")
                return True
            else:
                print(f"❌ BLOCKED: {pair} | {mode} TREND active since {active['started'].strftime('%H:%M')} ({time_elapsed:.1f}h)")
                return False
        return True

    def _check_btc_context(self) -> tuple[bool, str]:
        try:
            btc_candles = CoinDCXAPI.get_candles('BTCUSDT', '15m', 20)
            if btc_candles is None or btc_candles.empty or len(btc_candles) < 10:
                return True, "BTC check skipped"
            btc_close = btc_candles['close']
            btc_high = btc_candles['high']
            btc_low = btc_candles['low']
            btc_atr = Indicators.atr(btc_high, btc_low, btc_close)
            if btc_atr is None or len(btc_atr) < 2:
                return True, "BTC check skipped"
            current_btc_atr = float(btc_atr.iloc[-1])
            avg_btc_atr = float(btc_atr.tail(10).mean())
            if current_btc_atr > avg_btc_atr * 2.5:
                return False, "BTC extreme volatility"
            return True, "BTC stable"
        except Exception as e:
            print(f"⚠️ BTC check error: {e}")
            return True, "BTC check skipped"

    def _check_trading_hours(self) -> tuple[bool, str]:
        return True, "24/7 Trading"

    def _calculate_entry_sl_tp(self, direction: str, current_price: float, 
                               atr: float, mode_config: Dict) -> Optional[Dict]:
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
        
        min_sl_distance_pct = entry * 0.008
        min_tp_distance = entry * (config.MIN_TP_DISTANCE_PERCENT / 100)
        
        if current_price < 1.0:
            ABSOLUTE_MIN_SL = current_price * 0.020
        elif current_price < 10.0:
            ABSOLUTE_MIN_SL = current_price * 0.015
        else:
            ABSOLUTE_MIN_SL = current_price * 0.012
        
        min_sl_distance = max(min_sl_distance_pct, ABSOLUTE_MIN_SL)
        
        if direction == "LONG":
            if entry - sl < min_sl_distance:
                sl = round(entry - min_sl_distance, decimal_places)
            if tp1 - entry < min_tp_distance:
                tp1 = round(entry + min_tp_distance, decimal_places)
            if tp2 - tp1 < min_tp_distance * 0.5:
                tp2 = round(tp1 + min_tp_distance * 0.5, decimal_places)
        else:
            if sl - entry < min_sl_distance:
                sl = round(entry + min_sl_distance, decimal_places)
            if entry - tp1 < min_tp_distance:
                tp1 = round(entry - min_tp_distance, decimal_places)
            if tp1 - tp2 < min_tp_distance * 0.5:
                tp2 = round(tp1 - min_tp_distance * 0.5, decimal_places)
        
        if direction == "LONG":
            if tp1 <= entry or tp2 <= entry or tp1 >= tp2 or sl >= entry:
                return None
        else:
            if tp1 >= entry or tp2 >= entry or tp1 <= tp2 or sl <= entry:
                return None
        
        actual_sl_distance_pct = abs(entry - sl) / entry * 100
        MIN_SL_PCT = 2.0 if current_price < 1 else (1.5 if current_price < 10 else 1.2)
        if actual_sl_distance_pct < MIN_SL_PCT:
            print(f"❌ SL too close: {actual_sl_distance_pct:.2f}% (need {MIN_SL_PCT}%)")
            return None
        
        return {'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2}

    def _check_liquidation_safety(self, entry: float, sl: float) -> tuple[bool, float]:
        distance_pct = abs(entry - sl) / entry * 100
        MIN_DISTANCE = config.LIQUIDATION_BUFFER * 100
        is_safe = distance_pct >= MIN_DISTANCE
        return is_safe, distance_pct

    def _check_minimum_rr(self, entry: float, sl: float, tp1: float, mode: str) -> tuple[bool, float]:
        sl_distance = abs(entry - sl) / entry * 100
        tp1_distance = abs(tp1 - entry) / entry * 100
        rr = tp1_distance / sl_distance if sl_distance > 0 else 0
        MIN_RR = {'TREND': 1.8, 'MID': 1.6, 'QUICK': 1.4, 'SCALP': 1.4}
        required_rr = MIN_RR.get(mode, 1.4)
        if rr < required_rr:
            return False, rr
        return True, rr

    def _log_signal_performance(self, signal: Dict):
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
            file_exists = False
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
        except Exception as e:
            print(f"⚠️ Performance logging failed: {e}")

    def _calculate_base_score(self, rsi: float, adx: float, macd_hist: float, 
                             prev_macd_hist: float) -> int:
        score = 0
        if 35 < rsi < 65:
            score += 18
        elif 30 < rsi < 70:
            score += 14
        else:
            score += 8
        if adx > 40:
            score += 22
        elif adx > 30:
            score += 18
        elif adx > 25:
            score += 14
        else:
            score += 8
        if abs(macd_hist) > abs(prev_macd_hist):
            score += 16
        else:
            score += 8
        return score

    def _apply_volume_scoring(self, score: int, volume_surge: float, mode: str) -> tuple[int, str]:
        penalty = 0
        reason = ""
        if mode == 'TREND':
            if volume_surge < 0.8:
                return 0, f"TREND volume too low ({volume_surge:.2f}x < 0.8x)"
            elif volume_surge >= 1.2:
                score += 14
            elif volume_surge >= 1.0:
                score += 12
            else:
                score += 8
                penalty = 5
reason = f"Low volume ({volume_surge:.2f}x)"
        elif mode == 'MID':
            if volume_surge < 0.7:
                return 0, f"MID volume too low ({volume_surge:.2f}x < 0.7x)"
            elif volume_surge >= 1.0:
                score += 14
            elif volume_surge >= 0.8:
                score += 10
            else:
                score += 6
                penalty = 4
                reason = f"Low volume ({volume_surge:.2f}x)"
        elif mode == 'QUICK':
            if volume_surge < 1.0:
                return 0, f"QUICK volume too low ({volume_surge:.2f}x < 1.0x)"
            elif volume_surge >= 1.5:
                score += 14
            elif volume_surge >= 1.2:
                score += 12
            else:
                score += 8
        else:
            if volume_surge > 1.5:
                score += 14
            elif volume_surge > 1.0:
                score += 10
            else:
                score += 5
        return score - penalty, reason

    def _apply_mtf_scoring(self, score: int, mtf_trend: str, mode: str) -> tuple[int, str]:
        penalty = 0
        reason = ""
        if mtf_trend in ['STRONG_UP', 'STRONG_DOWN']:
            score += 12
        elif mtf_trend in ['MODERATE_UP', 'MODERATE_DOWN']:
            score += 8
            if mode == 'TREND':
                penalty = 8
                reason = f"TREND prefers STRONG MTF (got {mtf_trend})"
        elif mtf_trend in ['WEAK_UP', 'WEAK_DOWN']:
            score += 3
            penalty = 12
            reason = f"Weak MTF trend ({mtf_trend})"
        elif mtf_trend == 'CHOPPY':
            score += 0
            penalty = 15
            reason = "Choppy market"
        elif mtf_trend == 'MIXED':
            score += 2
            penalty = 10
            reason = "MIXED MTF"
        else:
            score += 2
            penalty = 5
            reason = "MTF unknown"
        return score - penalty, reason

    def _apply_regime_penalty(self, score: int, regime: str, mode: str) -> tuple[int, str]:
        penalty = 0
        reason = ""
        if regime == "RANGE" and mode == 'TREND':
            penalty = 10
            reason = "RANGE market (TREND mode)"
        elif regime == "TRENDING" and mode == 'QUICK':
            penalty = 8
            reason = "TRENDING market (QUICK mode)"
        return score - penalty, reason

    def _apply_smart_signals_bonus(self, score: int, sweep: bool, ob: bool, 
                                   fvg: bool, key_level: bool) -> int:
        if sweep:
            score += 8
        if ob:
            score += 8
        if fvg:
            score += 6
        if key_level:
            score += 6
        return score

    def analyze(self, pair: str, candles: pd.DataFrame, mode: str = None) -> Optional[Dict]:
        if mode is None:
            mode = config.MODE
        mode_config = config.MODE_CONFIG[mode]
        MIN_SCORE = {'TREND': 72, 'MID': 68, 'QUICK': 65, 'SCALP': 60}
        min_score = MIN_SCORE.get(mode, 65)
        
        self._reset_daily_counters()
        
        trading_allowed, time_reason = self._check_trading_hours()
        if not trading_allowed:
            print(f"❌ BLOCKED: {pair} | {mode} | {time_reason}")
            return None
        
        is_blocked, reason = news_guard.is_blocked()
        if is_blocked:
            print(f"❌ BLOCKED: {pair} | {mode} | News: {reason}")
            return None
        
        btc_stable, btc_reason = self._check_btc_context()
        if not btc_stable:
            print(f"❌ BLOCKED: {pair} | {mode} | {btc_reason}")
            return None
        
        if self.signal_count >= config.MAX_SIGNALS_PER_DAY:
            print(f"❌ BLOCKED: {pair} | {mode} | Daily limit")
            return None
        
        max_per_mode = config.MAX_SIGNALS_PER_DAY // len(config.ACTIVE_MODES)
        if self.mode_signal_count[mode] >= max_per_mode:
            print(f"❌ BLOCKED: {pair} | {mode} | Mode limit")
            return None
        
        if not self._check_cooldown(pair, mode):
            return None
        
        if not self._check_coin_daily_limit(pair, mode):
            return None
        
        if not self._check_global_coin_exposure(pair):
            return None
        
        if not self._check_one_coin_one_mode(pair, mode):
            return None
        
        if len(candles) < 50:
            print(f"❌ BLOCKED: {pair} | {mode} | Insufficient data")
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
            current_ema_fast = float(ema_fast.iloc[-1])
            current_ema_slow = float(ema_slow.iloc[-1])
            
            if any(pd.isna([current_price, current_rsi, current_adx, current_atr])):
                return None
MIN_ADX = {'TREND': 28, 'MID': 24, 'QUICK': 20, 'SCALP': 18}
            if current_adx < MIN_ADX.get(mode, 20):
                print(f"❌ BLOCKED: {pair} | {mode} | ADX too weak ({current_adx:.1f} < {MIN_ADX[mode]})")
                return None
            
            if mode == 'MID' and current_adx > 48:
                print(f"❌ BLOCKED: {pair} | MID | ADX too high ({current_adx:.1f} > 48)")
                return None
            
            if mode == 'TREND' and current_adx > 50:
                print(f"❌ BLOCKED: {pair} | TREND | ADX too high ({current_adx:.1f} > 50)")
                return None
            
            if mode == 'QUICK' and current_adx > 40:
                print(f"❌ BLOCKED: {pair} | QUICK | ADX too high ({current_adx:.1f} > 40)")
                return None
            
            ticker = CoinDCXAPI.get_ticker(pair)
            bid = ticker['bid'] if ticker else current_price
            ask = ticker['ask'] if ticker else current_price
            traps = TrapDetector.check_all_traps(candles, bid, ask, current_rsi, current_adx, current_macd_hist)
            trapped_count = sum(traps.values())
            if trapped_count >= 5:
                print(f"❌ BLOCKED: {pair} | {mode} | Too many traps ({trapped_count})")
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
                print(f"❌ BLOCKED: {pair} | {mode} | No clear trend")
                return None
            
            if mode == 'MID':
                if trend == "LONG":
                    if current_rsi > 68:
                        print(f"❌ BLOCKED: {pair} | MID LONG | RSI too high ({current_rsi:.1f} > 68)")
                        return None
                    if current_rsi < 30:
                        print(f"❌ BLOCKED: {pair} | MID LONG | RSI too low ({current_rsi:.1f} < 30)")
                        return None
                if trend == "SHORT":
                    if current_rsi < 32:
                        print(f"❌ BLOCKED: {pair} | MID SHORT | RSI too low ({current_rsi:.1f} < 32)")
                        return None
                    if current_rsi > 68:
                        print(f"❌ BLOCKED: {pair} | MID SHORT | RSI too high ({current_rsi:.1f} > 68)")
                        return None
            
            if mode == 'TREND':
                if trend == "LONG":
                    if current_rsi > 70:
                        print(f"❌ BLOCKED: {pair} | TREND LONG | RSI too high ({current_rsi:.1f} > 70)")
                        return None
                    if current_rsi < 35:
                        print(f"❌ BLOCKED: {pair} | TREND LONG | RSI too low ({current_rsi:.1f} < 35)")
                        return None
                if trend == "SHORT":
                    if current_rsi < 30:
                        print(f"❌ BLOCKED: {pair} | TREND SHORT | RSI too low ({current_rsi:.1f} < 30)")
                        return None
                    if current_rsi > 65:
                        print(f"❌ BLOCKED: {pair} | TREND SHORT | RSI too high ({current_rsi:.1f} > 65)")
                        return None
            
            if mode == 'QUICK':
                if trend == "LONG" and current_rsi > 65:
                    print(f"❌ BLOCKED: {pair} | QUICK LONG | RSI too high ({current_rsi:.1f} > 65)")
                    return None
                if trend == "SHORT" and current_rsi < 35:
                    print(f"❌ BLOCKED: {pair} | QUICK SHORT | RSI too low ({current_rsi:.1f} < 35)")
                    return None
            
            if not self._check_direction_flip_cooldown(pair, trend):
                return None
            
            # ✅ NEW: Pullback Entry Check
            timing_ok, timing_reason = EntryFilters.check_pullback_entry(candles, trend, mode)
            if not timing_ok:
                print(f"❌ BLOCKED: {pair} | {mode} | {timing_reason}")
                return None
            else:
                print(f"✅ Entry Timing: {timing_reason}")
            
            # ✅ NEW: Multi-Confirmation Check
            is_confirmed, conf_score, conf_list = EntryFilters.multi_confirmation_check(
                pair, candles, trend, mode
            )
            if not is_confirmed:
                print(f"❌ BLOCKED: {pair} | {mode} | Only {conf_score}/7 confirmations")
                for conf in conf_list:
                    print(f"   {conf}")
                return None
            
            print(f"✅ Confirmations: {conf_score}/7")
            for conf in conf_list:
                print(f"   {conf}")
            
            # ✅ NEW: Market Context Check
            context_ok, context_reason, market_context = EntryFilters.check_market_context(pair)
            if not context_ok:
                print(f"❌ BLOCKED: {pair} | {mode} | {context_reason}")
                return None
            
            print(f"✅ Market Context: {context_reason}")
            
            try:
                candles_1h = CoinDCXAPI.get_candles(pair, '1h', 50)
                if candles_1h.empty:
                    print(f"❌ BLOCKED: {pair} | {mode} | 1H fetch fail")
                    return None
            except Exception as e:
                print(f"❌ BLOCKED: {pair} | {mode} | HTF error: {e}")
                return None
            
            try:
                candles_5m = CoinDCXAPI.get_candles(pair, '5m', 50)
                candles_15m = CoinDCXAPI.get_candles(pair, '15m', 50)
                if not candles_5m.empty and not candles_15m.empty and not candles_1h.empty:
                    mtf_trend = Indicators.mtf_trend(candles_5m['close'], candles_15m['close'], candles_1h['close'])
                else:
                    mtf_trend = "UNKNOWN"
            except Exception as e:
                mtf_trend = "UNKNOWN"
            
            if mode == 'TREND':
                if mtf_trend in ['MIXED', 'CHOPPY', 'UNKNOWN']:
                    print(f"❌ BLOCKED: {pair} | TREND | MTF not aligned ({mtf_trend})")
                    return None
                if trend == 'LONG' and mtf_trend in ['STRONG_DOWN', 'MODERATE_DOWN']:
                    print(f"❌ BLOCKED: {pair} | TREND LONG vs {mtf_trend}")
                    return None
                if trend == 'SHORT' and mtf_trend in ['STRONG_UP', 'MODERATE_UP']:
                    print(f"❌ BLOCKED: {pair} | TREND SHORT vs {mtf_trend}")
                    return None
            
            regime = "NORMAL"
            try:
                candles_daily = CoinDCXAPI.get_candles(pair, '1d', 30)
                if not candles_daily.empty:
                    regime = Indicators.detect_market_regime(candles_daily)
            except Exception as e:
                pass
            
            has_fvg = False
            fvg_info = {}
            try:
                has_fvg, fvg_info = Indicators.detect_fvg(candles, trend)
                if has_fvg:
                    in_fvg = Indicators.is_price_in_fvg(current_price, fvg_info)
                    if not in_fvg:
                        has_fvg = False
            except Exception as e:
                pass
            
            near_key_level = False
            key_level_info = ""
            try:
                candles_daily = CoinDCXAPI.get_candles(pair, '1d', 30)
                if not candles_daily.empty:
                    key_levels = Indicators.get_daily_key_levels(candles_daily)
                    near_key_level, key_level_info = Indicators.check_key_level_proximity(current_price, key_levels, trend)
            except Exception as e:
                pass
            
            sweep_detected = False
            sweep_info = {}
            try:
                sweep_detected, sweep_info = Indicators.detect_liquidity_sweep(candles, trend)
            except Exception as e:
                pass
            
            near_ob = False
            ob_info = {}
            try:
                candles_4h_ob = CoinDCXAPI.get_candles(pair, '4h', 100)
                if not candles_4h_ob.empty:
                    order_blocks = Indicators.find_order_blocks(candles_4h_ob, trend)
                    near_ob, ob_info = Indicators.is_near_order_block(current_price, order_blocks)
            except Exception as e:
                pass
            
            base_score = self._calculate_base_score(current_rsi, current_adx, current_macd_hist, prev_macd_hist)
            score = base_score
            penalties = []
            
            print(f"\n🔍 {pair} {mode} {trend} - Base Score: {base_score}/100")
            
            score, vol_reason = self._apply_volume_scoring(score, current_volume_surge, mode)
            if score == 0:
                print(f"❌ BLOCKED: {pair} | {mode} | {vol_reason}")
                return None
            if vol_reason:
                penalties.append(vol_reason)
            
            score, mtf_reason = self._apply_mtf_scoring(score, mtf_trend, mode)
            if mtf_reason:
                penalties.append(mtf_reason)
            
            score, regime_reason = self._apply_regime_penalty(score, regime, mode)
            if regime_reason:
                penalties.append(regime_reason)
            
            if trapped_count >= 3:
                trap_penalty = trapped_count * 5
                score -= trap_penalty
                penalties.append(f"{trapped_count} traps detected")
            
            score = self._apply_smart_signals_bonus(score, sweep_detected, near_ob, has_fvg, near_key_level)
            
            score = max(0, min(100, score))
            
            print(f"   ✅ Final Score: {score}/100 (min: {min_score})")
            
            if score < min_score:
                print(f"❌ BLOCKED: {pair} | {mode} | Score: {score}/{min_score}")
                return None
            
            if not self._check_score_bucket_limit(score):
                return None
            
            levels = self._calculate_entry_sl_tp(trend, current_price, current_atr, mode_config)
            if levels is None:
                print(f"❌ BLOCKED: {pair} | {mode} | Invalid SL/TP structure")
                return None
            
            if mode == 'MID':
                tp1_distance_pct = abs(levels['tp1'] - levels['entry']) / levels['entry'] * 100
                if tp1_distance_pct < 0.3:
                    print(f"❌ BLOCKED: {pair} | MID | TP1 too close ({tp1_distance_pct:.2f}%)")
                    return None
            
            is_safe, sl_distance_pct = self._check_liquidation_safety(levels['entry'], levels['sl'])
            if not is_safe:
                print(f"❌ BLOCKED: {pair} | {mode} | Liquidation risk")
                return None
            
            rr_valid, rr_value = self._check_minimum_rr(levels['entry'], levels['sl'], levels['tp1'], mode)
            if not rr_valid:
                print(f"❌ BLOCKED: {pair} | {mode} | RR too low ({rr_value:.2f})")
                return None
            
            if mode == 'TREND':
                if not self._check_active_trend(pair, trend, mode, mtf_trend, current_price, 
                                               current_ema_fast, current_ema_slow, current_macd_hist):
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
                'base_score': base_score,
                'rsi': round(current_rsi, 1),
                'adx': round(current_adx, 1),
                'mtf_trend': mtf_trend,
                'mode': mode,
                'timeframe': mode_config['timeframe'],
                'volume_surge': round(current_volume_surge, 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_regime': regime,
                'liquidity_sweep': sweep_detected,
                'near_order_block': near_ob,
                'fvg_fill': has_fvg,
                'near_key_level': near_key_level,
                'sweep_info': sweep_info if sweep_detected else {},
                'ob_info': ob_info if near_ob else {},
                'fvg_info': fvg_info if has_fvg else {},
                'key_level_info': key_level_info if near_key_level else "",
                'ema_fast_period': mode_config['ema_fast'],
                'ema_slow_period': mode_config['ema_slow'],
                'penalties': penalties,
                'trapped_count': trapped_count,
                'market_context': market_context,  # ✅ NEW
                'confirmation_score': conf_score   # ✅ NEW
            }
            
            print(f"\n{'='*60}")
            print(f"📊 Rule-based PASSED: {pair} {trend}")
            print(f"{'='*60}")
            
            chatgpt_approved = self.chatgpt_advisor.final_trade_decision(signal, candles)
            if not chatgpt_approved:
                self.chatgpt_rejected += 1
                print(f"❌ ChatGPT Safety check failed")
                return None
            
            self.chatgpt_approved += 1
            
            print(f"✅ {mode}: {pair} APPROVED!")
            print(f"   Score: {score}/100 (Base: {base_score}) | RR: {rr_value:.2f}R | Vol: {current_volume_surge:.2f}x")
            print(f"   RSI: {current_rsi:.1f} | ADX: {current_adx:.1f} | MTF: {mtf_trend}")
            print(f"   Confirmations: {conf_score}/7")
            print(f"   Entry: ₹{levels['entry']:,.6f}")
            print(f"   SL: ₹{levels['sl']:,.6f} ({sl_distance_pct:.2f}%)")
            print(f"   TP1: ₹{levels['tp1']:,.6f}")
            print(f"   TP2: ₹{levels['tp2']:,.6f}")
            
            if sweep_detected:
                print(f"   💎 Liquidity Swept")
            if near_ob:
                print(f"   🎯 Near OB")
            if has_fvg:
                print(f"   💎 FVG Fill")
            if near_key_level:
                print(f"   🎯 {key_level_info}")
            if penalties:
                print(f"   ⚠️  Penalties: {', '.join(penalties)}")
            
            print(f"{'='*60}\n")
            
            try:
                explainer_result = SignalExplainer.explain_signal(signal, candles)
                if explainer_result['chart_path']:
                    TelegramNotifier.send_chart(explainer_result['chart_path'])
                if explainer_result['explanation']:
                    TelegramNotifier.send_explanation(explainer_result['explanation'])
            except Exception as e:
                print(f"⚠️ Explainer failed (non-critical): {e}")
            
            self._log_signal_performance(signal)
            
            self.signal_count += 1
            self.mode_signal_count[mode] += 1
            coin = pair.split('USDT')[0]
            if coin not in self.coin_signal_count:
                self.coin_signal_count[coin] = 0
            self.coin_signal_count[coin] += 1
            
            coin_mode_key = f"{coin}_{mode}"
            if coin_mode_key not in self.coin_mode_signal_count:
                self.coin_mode_signal_count[coin_mode_key] = 0
            self.coin_mode_signal_count[coin_mode_key] += 1
            
            if coin not in self.coin_total_signals:
                self.coin_total_signals[coin] = 0
            self.coin_total_signals[coin] += 1
            self.coin_active_mode[coin] = mode
            
            if score >= 85:
                self.score_buckets['high'] += 1
            elif score >= 75:
                self.score_buckets['medium'] += 1
            else:
                self.score_buckets['low'] += 1
            
            self.last_signal_time[f"{pair}_{mode}"] = datetime.now()
            self.last_signal_time[f"{coin}_last_signal"] = datetime.now()
            
            price_key = f"{pair}_{mode}"
            self.last_signal_price[price_key] = current_price
            self.last_signal_direction[coin] = trend
            
            if mode == 'TREND':
                trend_key = f"{pair}_{mode}_{trend}"
                self.active_trends[trend_key] = {
                    'started': datetime.now(),
                    'entry_price': current_price,
                    'direction': trend,
                    'mtf_trend': mtf_trend
                }
            
            self.signals_today.append(signal)
            
            return signal
        
        except Exception as e:
            print(f"❌ ERROR analyzing {pair}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None