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


class SignalGenerator:
    """SMART Rule-Based Signal Generator – Balanced Quality Filtering"""

    def __init__(self):
        self.signal_count = 0
        self.signals_today = []
        self.last_signal_time: Dict[str, datetime] = {}
        self.last_signal_price: Dict[str, float] = {}
        self.last_reset_date = datetime.now().date()
        self.mode_signal_count = {mode: 0 for mode in config.ACTIVE_MODES}
        self.coin_signal_count: Dict[str, int] = {}
        self.coin_mode_signal_count: Dict[str, int] = {}
        self.active_trends: Dict[str, Dict] = {}
        self.chatgpt_advisor = ChatGPTAdvisor()
        self.chatgpt_approved = 0
        self.chatgpt_rejected = 0
        self.active_positions: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        return {
            "signals_today": self.signal_count,
            "mode_breakdown": dict(self.mode_signal_count),
            "coin_breakdown": dict(self.coin_signal_count),
            "chatgpt_approved": self.chatgpt_approved,
            "chatgpt_rejected": self.chatgpt_rejected,
            "signals_today_list": self.signals_today.copy(),
        }

    def _reset_daily_counters(self):
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.signal_count = 0
            self.signals_today.clear()
            self.mode_signal_count = {mode: 0 for mode in config.ACTIVE_MODES}
            self.coin_signal_count.clear()
            self.coin_mode_signal_count.clear()
            self.last_signal_price.clear()
            self.active_trends.clear()
            self.chatgpt_approved = 0
            self.chatgpt_rejected = 0
            self.last_reset_date = today
            print(f"🔄 Daily counters reset for {today}")

    def _check_cooldown(self, pair: str, mode: str) -> bool:
        key = f"{pair}_{mode}"
        if key in self.last_signal_time:
            elapsed = datetime.now() - self.last_signal_time[key]
            cooldown_map = {
                "TREND": timedelta(hours=2),
                "MID": timedelta(hours=1),
            }
            cooldown = cooldown_map.get(mode, timedelta(minutes=config.SAME_PAIR_COOLDOWN_MINUTES))
            if elapsed < cooldown:
                mins_left = int((cooldown - elapsed).total_seconds() / 60)
                print(f"⏸️  {pair} ({mode}) in cooldown – {mins_left}m remaining")
                return False
        return True

    def _check_price_movement(self, pair: str, current_price: float, mode: str) -> bool:
        key = f"{pair}_{mode}"
        if key in self.last_signal_price:
            last = self.last_signal_price[key]
            pct = abs(current_price - last) / last * 100
            min_move = {"TREND": 1.5, "MID": 1.0, "QUICK": 0.6, "SCALP": 0.6}.get(mode, 1.0)
            if pct < min_move:
                print(f"❌ BLOCKED: {pair} | {mode} needs {min_move}% move (got {pct:.2f}%)")
                return False
        return True

    def _check_coin_daily_limit(self, pair: str, mode: str) -> bool:
        coin = pair.split("USDT")[0]
        key = f"{coin}_{mode}"
        self.coin_mode_signal_count.setdefault(key, 0)
        limit = {"TREND": 2, "MID": 3, "QUICK": 4, "SCALP": 4}.get(mode, 3)
        if self.coin_mode_signal_count[key] >= limit:
            print(f"❌ BLOCKED: {pair} | {mode} coin limit ({limit})")
            return False
        return True

    def _check_active_trend(self, pair: str, trend: str, price: float, ema_f: float, ema_s: float, macd: float) -> bool:
        key = f"{pair}_{trend}"
        if key not in self.active_trends:
            return True
        active = self.active_trends[key]
        hrs = (datetime.now() - active["started"]).total_seconds() / 3600
        if hrs > 8:
            del self.active_trends[key]
            print(f"🔓 TREND expired: {pair} {trend} (8h TTL)")
            return True
        pct = (price - active["entry_price"]) / active["entry_price"] * 100
        broken = (
            (trend == "LONG" and (pct < -2.5 or ema_f < ema_s))
            or (trend == "SHORT" and (pct > 2.5 or ema_f > ema_s))
        )
        if broken:
            del self.active_trends[key]
            print(f"🔓 TREND invalidated: {pair} {trend} (structure broken)")
            return True
        print(f"❌ BLOCKED: {pair} | TREND active since {active['started'].strftime('%H:%M')} ({hrs:.1f}h)")
        return False

    def _check_btc_context(self) -> tuple[bool, str]:
        try:
            btc = CoinDCXAPI.get_candles("BTCUSDT", "15m", 20)
            if btc is None or btc.empty or len(btc) < 10:
                return True, "BTC check skipped"
            atr = Indicators.atr(btc["high"], btc["low"], btc["close"])
            if atr is None or len(atr) < 2:
                return True, "BTC check skipped"
            curr, avg = float(atr.iloc[-1]), float(atr.tail(10).mean())
            if curr > avg * 2.0:
                return False, "BTC high volatility"
            return True, "BTC stable"
        except Exception as e:
            print(f"⚠️ BTC check error: {e}")
            return True, "BTC check skipped"

    def _check_trading_hours(self) -> tuple[bool, str]:
        return True, "24/7 Trading"

    # ------------------------------------------------------------------
    # SL / TP
    # ------------------------------------------------------------------
    def _calculate_entry_sl_tp(self, direction: str, price: float, atr: float, cfg: Dict) -> Optional[Dict]:
        sl_mult = cfg["atr_sl_multiplier"]
        tp1_mult = cfg["atr_tp1_multiplier"]
        tp2_mult = cfg["atr_tp2_multiplier"]
        dec = config.get_decimal_places(price)

        if direction == "LONG":
            entry = round(price, dec)
            sl = round(entry - atr * sl_mult, dec)
            tp1 = round(entry + atr * tp1_mult, dec)
            tp2 = round(entry + atr * tp2_mult, dec)
        else:
            entry = round(price, dec)
            sl = round(entry + atr * sl_mult, dec)
            tp1 = round(entry - atr * tp1_mult, dec)
            tp2 = round(entry - atr * tp2_mult, dec)

        min_sl_dist = entry * 0.008
        min_tp_dist = entry * (config.MIN_TP_DISTANCE_PERCENT / 100)

        if direction == "LONG":
            if entry - sl < min_sl_dist:
                sl = round(entry - min_sl_dist, dec)
            if tp1 - entry < min_tp_dist:
                tp1 = round(entry + min_tp_dist, dec)
            if tp2 - tp1 < min_tp_dist * 0.5:
                tp2 = round(tp1 + min_tp_dist * 0.5, dec)
            if tp1 <= entry or tp2 <= entry or tp1 >= tp2 or sl >= entry:
                return None
        else:
            if sl - entry < min_sl_dist:
                sl = round(entry + min_sl_dist, dec)
            if entry - tp1 < min_tp_dist:
                tp1 = round(entry - min_tp_dist, dec)
            if tp1 - tp2 < min_tp_dist * 0.5:
                tp2 = round(tp1 - min_tp_dist * 0.5, dec)
            if tp1 >= entry or tp2 >= entry or tp1 <= tp2 or sl <= entry:
                return None

        return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2}

    def _check_liquidation_safety(self, entry: float, sl: float) -> tuple[bool, float]:
        dist = abs(entry - sl) / entry * 100
        return dist >= config.LIQUIDATION_BUFFER * 100, dist

    def _check_minimum_rr(self, entry: float, sl: float, tp1: float, mode: str) -> tuple[bool, float]:
        sl_dist = abs(entry - sl) / entry * 100
        tp_dist = abs(tp1 - entry) / entry * 100
        rr = tp_dist / sl_dist if sl_dist else 0
        required = {"TREND": 2.0, "MID": 1.8, "QUICK": 1.5, "SCALP": 1.5}.get(mode, 1.5)
        return rr >= required, rr

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_signal_performance(self, sig: Dict):
        if not getattr(config, "TRACK_PERFORMANCE", False):
            return
        log_file = getattr(config, "PERFORMANCE_LOG_FILE", "signal_performance.csv")
        row = {
            "timestamp": sig.get("timestamp", datetime.now().isoformat()),
            "pair": sig["pair"],
            "direction": sig["direction"],
            "mode": sig.get("mode", "UNKNOWN"),
            "timeframe": sig.get("timeframe", "UNKNOWN"),
            "entry": sig["entry"],
            "sl": sig["sl"],
            "tp1": sig["tp1"],
            "tp2": sig["tp2"],
            "score": sig["score"],
            "rsi": sig["rsi"],
            "adx": sig["adx"],
            "leverage": sig["leverage"],
        }
        try:
            import os
            exists = os.path.isfile(log_file)
            with open(log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"⚠️ Performance logging failed: {e}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _calculate_signal_score(self, ind: Dict, mtf: str, sweep: bool, ob: bool, fvg: bool, kl: bool, mode: str) -> int:
        score = 0
        rsi, adx, vol = ind["rsi"], ind["adx"], ind["volume_surge"]

        # RSI
        if 35 < rsi < 65:
            score += 18
        elif 30 < rsi < 70:
            score += 14
        else:
            score += 8

        # ADX
        if adx > 40:
            score += 22
        elif adx > 30:
            score += 18
        elif adx > 25:
            score += 14
        else:
            score += 8

        # MACD
        if abs(ind["macd_histogram"]) > abs(ind["prev_macd_histogram"]):
            score += 16
        else:
            score += 8

        # Volume
        if mode == "TREND":
            if vol >= 1.2:
                score += 14
            elif vol >= 1.0:
                score += 10
            elif vol >= 0.8:
                score += 6
            else:
                score += 0
        else:
            if vol > 1.5:
                score += 14
            elif vol > 1.0:
                score += 10
            else:
                score += 5

        # MTF
        if mtf in ("STRONG_UP", "STRONG_DOWN"):
            score += 16
        elif mtf in ("MODERATE_UP", "MODERATE_DOWN"):
            score += 12
        else:
            score += 4

        # Extras
        if sweep:
            score += 8
        if ob:
            score += 8
        if fvg:
            score += 6
        if kl:
            score += 6

        return min(score, 100)

    # ------------------------------------------------------------------
    # Main analyser
    # ------------------------------------------------------------------
    def analyze(self, pair: str, candles: pd.DataFrame, mode: str = None) -> Optional[Dict]:
        if mode is None:
            mode = config.MODE
        cfg = config.MODE_CONFIG[mode]
        min_score = cfg.get("min_score", config.MIN_SIGNAL_SCORE)
        min_adx = cfg.get("min_adx", config.MIN_ADX_STRENGTH)
        self._reset_daily_counters()

        penalties: list[tuple[str, int]] = []

        # ---------------- Global gatekeepers ----------------
        ok, reason = self._check_trading_hours()
        if not ok:
            print(f"❌ BLOCKED: {pair} | {mode} | {reason}")
            return None

        if news_guard.is_blocked()[0]:
            print(f"❌ BLOCKED: {pair} | {mode} | News: {news_guard.is_blocked()[1]}")
            return None

        btc_ok, btc_reason = self._check_btc_context()
        if not btc_ok:
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

        if len(candles) < 50:
            print(f"❌ BLOCKED: {pair} | {mode} | Insufficient data")
            return None

        # ---------------- Technical prep ----------------
        try:
            candles = candles.dropna()
            if len(candles) < 50:
                return None

            c = candles["close"]
            h = candles["high"]
            l = candles["low"]
            v = candles["volume"]

            ema_f = Indicators.ema(c, cfg["ema_fast"]).dropna()
            ema_s = Indicators.ema(c, cfg["ema_slow"]).dropna()
            macd_l, sig_l, hist = Indicators.macd(c)
            rsi = Indicators.rsi(c).dropna()
            adx, pdi, mdi = Indicators.adx(h, l, c)
            atr = Indicators.atr(h, l, c).dropna()
            vol_surge = Indicators.volume_surge(v).dropna()

            if any(
                len(x) < 2
                for x in (rsi, adx, atr, vol_surge, ema_f, ema_s, hist, pdi, mdi)
            ):
                return None

            price = float(c.iloc[-1])
            cur_rsi = float(rsi.iloc[-1])
            cur_adx = float(adx.iloc[-1])
            cur_atr = float(atr.iloc[-1])
            cur_hist = float(hist.iloc[-1])
            prev_hist = float(hist.iloc[-2]) if len(hist) > 1 else 0.0
            cur_vol = float(vol_surge.iloc[-1]) if len(vol_surge) else 1.0
            cur_ema_f = float(ema_f.iloc[-1])
            cur_ema_s = float(ema_s.iloc[-1])

            if any(pd.isna([price, cur_rsi, cur_adx, cur_atr])):
                return None

            # Volume filters
            if mode == "TREND" and cur_vol < 0.8:
                print(f"❌ BLOCKED: {pair} | TREND volume < 0.8x (got {cur_vol:.2f}x)")
                return None
            if mode == "MID" and cur_vol < 0.5:
                print(f"❌ BLOCKED: {pair} | MID volume < 0.5x (got {cur_vol:.2f}x)")
                return None
            if mode == "QUICK" and cur_vol < 1.0:
                print(f"❌ BLOCKED: {pair} | QUICK volume < 1.0x (got {cur_vol:.2f}x)")
                return None

            if not self._check_price_movement(pair, price, mode):
                return None

            # ---------------- Trend direction ----------------
            if ema_f.iloc[-1] > ema_s.iloc[-1] and macd_l.iloc[-1] > sig_l.iloc[-1] and pdi.iloc[-1] > mdi.iloc[-1]:
                trend = "LONG"
            elif ema_f.iloc[-1] < ema_s.iloc[-1] and macd_l.iloc[-1] < sig_l.iloc[-1] and mdi.iloc[-1] > pdi.iloc[-1]:
                trend = "SHORT"
            else:
                print(f"❌ BLOCKED: {pair} | {mode} | No clear trend")
                return None

            # ---------------- Higher-time-frame checks ----------------
            try:
                candles_1h = CoinDCXAPI.get_candles(pair, "1h", 50)
                candles_4h = None
                if mode == "TREND":
                    candles_4h = CoinDCXAPI.get_candles(pair, "4h", 50)
                if candles_1h.empty:
                    print(f"❌ BLOCKED: {pair} | {mode} | 1H fetch fail")
                    return None
                htf_ok, htf_reason = Indicators.check_htf_alignment(
                    trend,
                    mode,
                    candles_1h["close"],
                    candles_4h["close"] if candles_4h is not None and not candles_4h.empty else None,
                )
                if not htf_ok:
                    penalties.append(("HTF_AGAINST", 12))
                    print(f"⚠️ WARNING: {pair} | {mode} | {htf_reason} (score -12)")
                else:
                    print(f"✅ HTF: {htf_reason}")
            except Exception as e:
                print(f"❌ BLOCKED: {pair} | {mode} | HTF error: {e}")
                return None

            # ---------------- Regime ----------------
            regime = "NORMAL"
            try:
                daily = CoinDCXAPI.get_candles(pair, "1d", 30)
                if not daily.empty:
                    regime = Indicators.detect_market_regime(daily)
            except Exception:
                pass

            # ---------------- FVG ----------------
            has_fvg, fvg_info = False, {}
            try:
                has_fvg, fvg_info = Indicators.detect_fvg(candles, trend)
                if has_fvg:
                    if not Indicators.is_price_in_fvg(price, fvg_info):
                        has_fvg = False
            except Exception:
                pass

            # ---------------- Key level ----------------
            near_kl, kl_info = False, ""
            try:
                daily = CoinDCXAPI.get_candles(pair, "1d", 30)
                if not daily.empty:
                    levels = Indicators.get_daily_key_levels(daily)
                    near_kl, kl_info = Indicators.check_key_level_proximity(price, levels, trend)
            except Exception:
                pass

            # ---------------- Sweep ----------------
            sweep, sweep_info = False, {}
            try:
                sweep, sweep_info = Indicators.detect_liquidity_sweep(candles, trend)
            except Exception:
                pass

            # ---------------- Order block ----------------
            near_ob, ob_info = False, {}
            try:
                ob_candles = CoinDCXAPI.get_candles(pair, "4h", 100)
                if not ob_candles.empty:
                    obs = Indicators.find_order_blocks(ob_candles, trend)
                    near_ob, ob_info = Indicators.is_near_order_block(price, obs)
            except Exception:
                pass

            # ---------------- Mode-specific RSI filters ----------------
            if mode == "TREND":
                if trend == "LONG" and (cur_rsi > 65 or cur_rsi < 38):
                    print(f"❌ BLOCKED: {pair} | TREND LONG | RSI out of range (38-65) | Got {cur_rsi:.1f}")
                    return None
                if trend == "SHORT" and (cur_rsi < 35 or cur_rsi > 62):
                    print(f"❌ BLOCKED: {pair} | TREND SHORT | RSI out of range (35-62) | Got {cur_rsi:.1f}")
                    return None
            elif mode == "MID":
                if trend == "LONG":
                    if cur_rsi > config.RSI_OVERBOUGHT:
                        print(f"❌ BLOCKED: {pair} | MID LONG | RSI overbought")
                        return None
                    if cur_rsi < 32:
                        print(f"❌ BLOCKED: {pair} | MID LONG | RSI too low (<32)")
                        return None
                if trend == "SHORT":
                    if cur_adx > 50 and cur_rsi < 32:
                        print(f"❌ BLOCKED: {pair} | MID SHORT | Exhaustion zone (ADX {cur_adx:.1f}, RSI {cur_rsi:.1f})")
                        return None
                    if cur_rsi < 35:
                        print(f"❌ BLOCKED: {pair} | MID SHORT | RSI too low (<35)")
                        return None
                    if cur_rsi > 65:
                        print(f"❌ BLOCKED: {pair} | MID SHORT | RSI too high (>65)")
                        return None
            elif mode == "QUICK":
                if trend == "LONG" and cur_rsi > 62:
                    print(f"❌ BLOCKED: {pair} | QUICK LONG | RSI too high (>62)")
                    return None
                if trend == "SHORT" and cur_rsi < 38:
                    print(f"❌ BLOCKED: {pair} | QUICK SHORT | RSI too low (<38)")
                    return None
                if cur_adx > 45:
                    print(f"❌ BLOCKED: {pair} | QUICK | ADX too strong (>45)")
                    return None
            else:  # SCALP
                if trend == "LONG" and cur_rsi > config.RSI_OVERBOUGHT:
                    print(f"❌ BLOCKED: {pair} | SCALP LONG | RSI overbought")
                    return None
                if trend == "SHORT" and cur_rsi < config.RSI_OVERSOLD:
                    print(f"❌ BLOCKED: {pair} | SCALP SHORT | RSI oversold")
                    return None

            if cur_adx < min_adx:
                print(f"❌ BLOCKED: {pair} | {mode} | ADX weak")
                return None

            # ---------------- SL / TP ----------------
            levels = self._calculate_entry_sl_tp(trend, price, cur_atr, cfg)
            if levels is None:
                print(f"❌ BLOCKED: {pair} | {mode} | Invalid SL/TP")
                return None

            if mode == "MID":
                tp1_dist = abs(levels["tp1"] - levels["entry"]) / levels["entry"] * 100
                if tp1_dist < 0.4:
                    print(f"❌ BLOCKED: {pair} | MID | TP1 too close to entry ({tp1_dist:.2f}%)")
                    return None

            safe, liq_dist = self._check_liquidation_safety(levels["entry"], levels["sl"])
            if not safe:
                print(f"❌ BLOCKED: {pair} | {mode} | Liquidation risk")
                return None

            rr_ok, rr_val = self._check_minimum_rr(levels["entry"], levels["sl"], levels["tp1"], mode)
            if not rr_ok:
                print(f"❌ BLOCKED: {pair} | {mode} | RR too low ({rr_val:.2f})")
                return None

            # ---------------- MTF trend ----------------
            try:
                c5 = CoinDCXAPI.get_candles(pair, "5m", 50)
                c15 = CoinDCXAPI.get_candles(pair, "15m", 50)
                c1h = CoinDCXAPI.get_candles(pair, "1h", 50)
                if not c5.empty and not c15.empty and not c1h.empty:
                    mtf = Indicators.mtf_trend(c5["close"], c15["close"], c1h["close"])
                else:
                    mtf = "UNKNOWN"
            except Exception:
                mtf = "UNKNOWN"

            if mode == "TREND":
                if mtf not in ("STRONG_UP", "STRONG_DOWN", "MODERATE_UP", "MODERATE_DOWN"):
                    print(f"❌ BLOCKED: {pair} | TREND requires trending MTF (got {mtf})")
                    return None
                if mtf == "MODERATE_UP" or mtf == "MODERATE_DOWN":
                    penalties.append(("MTF_MODERATE", 5))
                    print(f"⚠️ WARNING: {pair} | TREND | MODERATE MTF (score -5)")
                if mtf == "STRONG_DOWN" and trend == "LONG":
                    if cur_rsi <= 50:
                        print(f"❌ BLOCKED: {pair} | TREND LONG vs STRONG_DOWN | RSI not bullish")
                        return None
                    if len(candles) >= 2 and candles["low"].iloc[-1] <= candles["low"].iloc[-2]:
                        print(f"❌ BLOCKED: {pair} | TREND LONG vs STRONG_DOWN | No higher low")
                        return None
                if mtf == "STRONG_UP" and trend == "SHORT":
                    if cur_rsi >= 50:
                        print(f"❌ BLOCKED: {pair} | TREND SHORT vs STRONG_UP | RSI not bearish")
                        return None
                    if len(candles) >= 2 and candles["high"].iloc[-1] >= candles["high"].iloc[-2]:
                        print(f"❌ BLOCKED: {pair} | TREND SHORT vs STRONG_UP | No lower high")
                        return None
            elif mode in ("MID", "QUICK"):
                if (mtf == "STRONG_DOWN" and trend == "LONG") or (mtf == "STRONG_UP" and trend == "SHORT"):
                    penalties.append(("COUNTER_TREND", 8))
                    print(f"⚠️ WARNING: {pair} | {mode} | Counter-trend setup (score -8)")

            # ---------------- Score ----------------
            indicators_data = {
                "rsi": cur_rsi,
                "adx": cur_adx,
                "macd_histogram": cur_hist,
                "prev_macd_histogram": prev_hist,
                "volume_surge": cur_vol,
            }
            score = self._calculate_signal_score(indicators_data, mtf, sweep, near_ob, has_fvg, near_kl, mode)
            for reason, val in penalties:
                score -= val
            if score < min_score:
                print(f"❌ BLOCKED: {pair} | {mode} | Score: {score}/{min_score} (after penalties)")
                return None

            if mode == "TREND":
                if not self._check_active_trend(pair, trend, price, cur_ema_f, cur_ema_s, cur_hist):
                    return None

            # ---------------- Build signal ----------------
            signal = {
                "pair": pair,
                "direction": trend,
                "entry": levels["entry"],
                "sl": levels["sl"],
                "tp1": levels["tp1"],
                "tp2": levels["tp2"],
                "leverage": cfg["leverage"],
                "score": score,
                "rsi": round(cur_rsi, 1),
                "adx": round(cur_adx, 1),
                "mtf_trend": mtf,
                "mode": mode,
                "timeframe": cfg["timeframe"],
                "volume_surge": round(cur_vol, 2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market_regime": regime,
                "liquidity_sweep": sweep,
                "near_order_block": near_ob,
                "fvg_fill": has_fvg,
                "near_key_level": near_kl,
                "sweep_info": sweep_info if sweep else {},
                "ob_info": ob_info if near_ob else {},
                "fvg_info": fvg_info if has_fvg else {},
                "key_level_info": kl_info if near_kl else "",
                "ema_fast_period": cfg["ema_fast"],
                "ema_slow_period": cfg["ema_slow"],
            }

            print("\n" + "=" * 60)
            print(f"📊 Rule-based PASSED: {pair} {trend}")
            print("=" * 60)

            # ---------------- AI safety ----------------
            if not self.chatgpt_advisor.final_trade_decision(signal, candles):
                self.chatgpt_rejected += 1
                print("❌ Safety check failed")
                return None
            self.chatgpt_approved += 1
            print(f"✅ {mode}: {pair} APPROVED!")
            print(f"   Score: {score}/100 | RR: {rr_val:.2f}R | Vol: {cur_vol:.2f}x")
            print(f"   Entry: ₹{levels['entry']:,.6f}")
            print(f"   SL:    ₹{levels['sl']:,.6f}")
            print(f"   TP1:   ₹{levels['tp1']:,.6f}")
            print(f"   TP2:   ₹{levels['tp2']:,.6f}")
            if sweep:
                print("   💎 Liquidity Swept")
            if near_ob:
                print("   🎯 Near OB")
            if has_fvg:
                print("   💎 FVG Fill")
            if near_kl:
                print(f"   🎯 {kl_info}")
            print("=" * 60 + "\n")

            # ---------------- Notify ----------------
            try:
                exp = SignalExplainer.explain_signal(signal, candles)
                if exp.get("chart_path"):
                    TelegramNotifier.send_chart(exp["chart_path"])
                if exp.get("explanation"):
                    TelegramNotifier.send_explanation(exp["explanation"])
            except Exception as e:
                print(f"⚠️ Explainer failed (non-critical): {e}")

            # ---------------- Book-keeping ----------------
            self._log_signal_performance(signal)
            self.signal_count += 1
            self.mode_signal_count[mode] += 1
            coin = pair.split("USDT")[0]
            self.coin_signal_count[coin] = self.coin_signal_count.get(coin, 0) + 1
            coin_mode_key = f"{coin}_{mode}"
            self.coin_mode_signal_count[coin_mode_key] = self.coin_mode_signal_count.get(coin_mode_key, 0) + 1
            self.last_signal_time[f"{pair}_{mode}"] = datetime.now()
            self.last_signal_price[f"{pair}_{mode}"] = price
            if mode == "TREND":
                self.active_trends[f"{pair}_{trend}"] = {
                    "started": datetime.now(),
                    "entry_price": price,
                    "direction": trend,
                }
            self.signals_today.append(signal)
            return signal

        except Exception as e:
            print(f"❌ ERROR analyzing {pair}: {e}")
            import traceback

            traceback.print_exc()
            return None
