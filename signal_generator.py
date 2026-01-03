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
    """Enhanced Signal Generator with Entry Filters"""
    
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
            print(f"❌ BLOCKED: {pair} | Global coin exposure limit")
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
                    print(f"❌ BLOCKED: {pair} | Already has {existing_mode}")
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
                        print(f"❌ BLOCKED: {pair} | Direction flip cooldown")
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
            print(f"❌ BLOCKED: Score bucket limit")
            return False
        return True

    def _check_btc_context(self) -> tuple[bool, str]:
        try:
            btc_candles = CoinDCXAPI.get_candles('BTCUSDT', '15m', 20)
            if btc_candles is None or btc_candles.empty:
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
        except:
            return True, "BTC check skipped"

    def _calculate_entry_sl_tp(self, direction: str, current_price: float, 
                               atr: float, mode_config: Dict) -> Optional[Dict]:
        sl_multiplier = mode_config['atr_sl_multiplier']
        tp1_multiplier = mode_config['atr_tp1_multiplier']
        tp2_multiplier = mode_config['atr_tp2_multiplier']
        decimal_places = config.get_decimal_places(current_price)
        
        if direction == "LONG":
            entry = current_price
            sl = entry - (atr * sl_multiplier)
            tp1 = entry + (atr * tp1_multiplier)
            tp2 = entry + (atr * tp2_multiplier)
        else:
            entry = current_price
            sl = entry + (atr * sl_multiplier)
            tp1 = entry - (atr * tp1_multiplier)
            tp2 = entry - (atr * tp2_multiplier)
        
        entry = round(entry, decimal_places)
        sl = round(sl, decimal_places)
        tp1 = round(tp1, decimal_places)
        tp2 = round(tp2, decimal_places)
        
        if direction == "LONG":
            if tp1 <= entry or tp2 <= entry or sl >= entry:
                return None
        else:
            if tp1 >= entry or tp2 >= entry or sl <= entry:
                return None
        
        return {'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2}

    def _calculate_base_score(self, rsi: float, adx: float, macd_hist: float, prev_macd_hist: float) -> int:
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
                return 0, "TREND volume too low"
            elif volume_surge >= 1.2:
                score += 14
            else:
                score += 8
        elif mode == 'MID':
            if volume_surge < 0.7:
                return 0, "MID volume too low"
            elif volume_surge >= 1.0:
                score += 14
            else:
                score += 8
        elif mode == 'QUICK':
            if volume_surge < 1.0:
                return 0, "QUICK volume too low"
            elif volume_surge >= 1.5:
                score += 14
            else:
                score += 8
        
        return score - penalty, reason

    def _apply_mtf_scoring(self, score: int, mtf_trend: str, mode: str) -> tuple[int, str]:
        if mtf_trend in ['STRONG_UP', 'STRONG_DOWN']:
            score += 12
        elif mtf_trend in ['MODERATE_UP', 'MODERATE_DOWN']:
            score += 8
        else:
            score += 3
        
        return score, ""

    def analyze(self, pair: str, candles: pd.DataFrame, mode: str = None) -> Optional[Dict]:
        if mode is None:
            mode = config.MODE
        
        mode_config = config.MODE_CONFIG[mode]
        MIN_SCORE = {'TREND': 72, 'MID': 68, 'QUICK': 65, 'SCALP': 60}
        min_score = MIN_SCORE.get(mode, 65)
        
        self._reset_daily_counters()
        
        is_blocked, reason = news_guard.is_blocked()
        if is_blocked:
            print(f"❌ BLOCKED: {pair} | News: {reason}")
            return None
        
        btc_stable, btc_reason = self._check_btc_context()
        if not btc_stable:
            print(f"❌ BLOCKED: {pair} | {btc_reason}")
            return None
        
        if self.signal_count >= config.MAX_SIGNALS_PER_DAY:
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
            
            if len(rsi) < 2 or len(adx) < 2 or len(atr) < 2:
                return None
            
            current_price = float(close.iloc[-1])
            current_rsi = float(rsi.iloc[-1])
            current_adx = float(adx.iloc[-1])
            current_atr = float(atr.iloc[-1])
            current_macd_hist = float(histogram.iloc[-1])
            prev_macd_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0.0
            current_volume_surge = float(volume_surge.iloc[-1]) if len(volume_surge) > 0 else 1.0
            
            MIN_ADX = {'TREND': 28, 'MID': 24, 'QUICK': 20}
            if current_adx < MIN_ADX.get(mode, 20):
                return None
            
            ticker = CoinDCXAPI.get_ticker(pair)
            bid = ticker['bid'] if ticker else current_price
            ask = ticker['ask'] if ticker else current_price
            traps = TrapDetector.check_all_traps(candles, bid, ask, current_rsi, current_adx, current_macd_hist)
            trapped_count = sum(traps.values())
            if trapped_count >= 5:
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
            
            # Entry filters
            try:
                timing_ok, timing_reason = EntryFilters.check_pullback_entry(candles, trend, mode)
                if not timing_ok:
                    print(f"❌ Entry timing: {timing_reason}")
                    return None
                
                is_confirmed, conf_score, conf_list = EntryFilters.multi_confirmation_check(pair, candles, trend, mode)
                if not is_confirmed:
                    print(f"❌ Confirmations: {conf_score}/7")
                    return None
                
                context_ok, context_reason, market_context = EntryFilters.check_market_context(pair)
                if not context_ok:
                    print(f"❌ Context: {context_reason}")
                    return None
            except Exception as e:
                print(f"⚠️ Entry filter error: {e}")
                market_context = {}
                conf_score = 0
            
            try:
                candles_5m = CoinDCXAPI.get_candles(pair, '5m', 50)
                candles_15m = CoinDCXAPI.get_candles(pair, '15m', 50)
                candles_1h = CoinDCXAPI.get_candles(pair, '1h', 50)
                if not candles_5m.empty and not candles_15m.empty and not candles_1h.empty:
                    mtf_trend = Indicators.mtf_trend(candles_5m['close'], candles_15m['close'], candles_1h['close'])
                else:
                    mtf_trend = "UNKNOWN"
            except:
                mtf_trend = "UNKNOWN"
            
            base_score = self._calculate_base_score(current_rsi, current_adx, current_macd_hist, prev_macd_hist)
            score = base_score
            
            score, vol_reason = self._apply_volume_scoring(score, current_volume_surge, mode)
            if score == 0:
                return None
            
            score, mtf_reason = self._apply_mtf_scoring(score, mtf_trend, mode)
            score = max(0, min(100, score))
            
            if score < min_score:
                return None
            
            levels = self._calculate_entry_sl_tp(trend, current_price, current_atr, mode_config)
            if levels is None:
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
                'ema_fast_period': mode_config['ema_fast'],
                'ema_slow_period': mode_config['ema_slow'],
                'trapped_count': trapped_count,
                'market_context': market_context,
                'confirmation_score': conf_score
            }
            
            chatgpt_approved = self.chatgpt_advisor.final_trade_decision(signal, candles)
            if not chatgpt_approved:
                self.chatgpt_rejected += 1
                return None
            
            self.chatgpt_approved += 1
            
            print(f"✅ {mode}: {pair} {trend} APPROVED!")
            print(f"   Score: {score}/100 | Confirmations: {conf_score}/7")
            
            self.signal_count += 1
            self.mode_signal_count[mode] += 1
            coin = pair.split('USDT')[0]
            if coin not in self.coin_signal_count:
                self.coin_signal_count[coin] = 0
            self.coin_signal_count[coin] += 1
            
            self.last_signal_time[f"{pair}_{mode}"] = datetime.now()
            self.signals_today.append(signal)
            
            return signal
        
        except Exception as e:
            print(f"❌ ERROR analyzing {pair}: {str(e)}")
            return None