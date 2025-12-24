from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from config import config
import time
import json


class ChatGPTAdvisor:
    """
    CHATGPT SNIPER FILTER - FINAL DECISION LAYER
    ---------------------------------------------
    PHILOSOSPHY: Discretionary Human Trader
    - Probability > Indicator Perfection
    - HARD blocks are RARE and FINAL
    - Soft imperfections do NOT block trades
    - Target: 10-15 tradable setups/day

    HARD REJECT CRITERIA (Non-negotiable):
    1. RR < 1.5
    2. SL < 0.8%
    3. TRUE exhaustion: (RSI > 78 OR < 22) AND ADX > 50
    4. Invalid price data
    5. BTC strong impulse AGAINST signal direction

    CONTEXT ONLY (NOT blockers):
    - Low/average volume
    - Mixed MTF
    - 1-2 trap warnings
    - Imperfect RSI
    - Late entry
    - BTC ranging/sideways

    TREND MODE: More lenient
    - ADX ≥ 30 + RR ≥ 2.0 = VALID continuation

    OUTPUT: {"approved": true/false}
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 8
        self.max_retries = 2

    # ------------------------------------------------------------------
    # low-level helpers
    # ------------------------------------------------------------------
    def _call_chatgpt_with_timeout(self, messages: List[Dict]) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                return (
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1200,  # enough for 10-pro
                        temperature=0.2,
                        timeout=self.timeout,
                    )
                    .choices[0]
                    .message.content.strip()
                )
            except Exception as e:
                print(f"⚠️ ChatGPT attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        return None

    def _parse_decision(self, response: str) -> bool:
        if not response:
            return False
        try:
            response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)
            return bool(data["approved"])
        except Exception as e:
            print(f"⚠️ ChatGPT parse error: {e}")
            return False

    # ------------------------------------------------------------------
    # 10-PRO MODULES (SINGLE API CALL)
    # ------------------------------------------------------------------
    def _build_10pro_prompt(self, signal: Dict, btc_context: Dict) -> str:
        direction = signal.get("direction", "UNKNOWN")
        entry = float(signal.get("entry", 0))
        sl = float(signal.get("sl", 0))
        tp1 = float(signal.get("tp1", 0))
        rsi = float(signal.get("rsi", 50))
        adx = float(signal.get("adx", 20))
        vol = float(signal.get("volume_surge", 1.0))
        pair = signal.get("pair", "UNKNOWN")
        mode = signal.get("mode", "UNKNOWN")

        sl_dist = abs(entry - sl) / entry * 100 if entry else 0
        tp_dist = abs(tp1 - entry) / entry * 100 if entry else 0
        rr = tp_dist / sl_dist if sl_dist else 0
        btc_txt = f"{btc_context.get('direction','')} ({btc_context.get('strength','')})" if btc_context.get('valid_context') else "NO_DATA"

        return f"""You are a 10-year crypto SNIPER. Fill ALL 10 keys in 1 JSON (no text outside).

SIGNAL: {pair} {direction} entry={entry} sl={sl} tp={tp1} rr={rr:.2f} rsi={rsi} adx={adx} vol={vol}x mode={mode} btc={btc_txt}

1) multi_tf_score: {{"m5": 0-100, "m15": 0-100, "m1h": 0-100, "weighted": 0-100}}  (weight: m5=40%, m15=35%, m1h=25%)
2) liquidity_path: {{"liquidity_above": price, "liquidity_below": price, "entry_distance_pips": int, "hunt_risk": "LOW|MED|HIGH"}}
3) smart_money: {{"smart_bias": "LONG|SHORT|NEUTRAL", "strength": 0-10}}
4) fake_out_bonus: {{"fake_sweep": bool, "bonus": 0|20}}
5) stop_hunt_distance: {{"atr5_pips": int, "sl_distance_pips": int, "sl_safe": bool}}
6) rr_optimiser: {{"tp2_best": price, "rr_new": float}}
7) news_guard: {{"news_2h": bool, "recommend": "WAIT_15M|REDUCE_50%|NORMAL"}}
8) continuation_check: {{"type": "continuation|mean_reversion", "pass": bool}}
9) partial_tp_map: {{"tp25": price, "tp50": price, "tp75": price}}
10) trailing_plan: {{"trail_on": "EMA-21|structure|ATR", "move_sl_to": price}}

RETURN ONLY JSON:
{{"multi_tf_score": {{...}}, "liquidity_path": {{...}}, ..., "trailing_plan": {{...}}, "approved": true/false}}"""

    def _parse_10pro(self, txt: str) -> Dict:
        if not txt:
            return {"approved": False}
        try:
            txt = txt.replace("```json", "").replace("```", "").strip()
            return json.loads(txt)
        except Exception as e:
            print("⚠️ 10-pro parse error:", e)
            return {"approved": False}

    # ------------------------------------------------------------------
    # BTC context (unchanged)
    # ------------------------------------------------------------------
    def _get_btc_context(self) -> Dict:
        try:
            from coindcx_api import CoinDCXAPI
            from indicators import Indicators

            btc_5m = CoinDCXAPI.get_candles('BTCUSDT', '5m', 20)
            btc_15m = CoinDCXAPI.get_candles('BTCUSDT', '15m', 20)
            btc_1h = CoinDCXAPI.get_candles('BTCUSDT', '1h', 20)
            if btc_5m.empty or btc_15m.empty or btc_1h.empty:
                return {'valid_context': False, 'reason': 'BTC data unavailable'}

            latest = btc_5m.iloc[-1]
            open_p, close_p, high_p, low_p = float(latest['open']), float(latest['close']), float(latest['high']), float(latest['low'])
            body_pct = abs(close_p - open_p) / (high_p - low_p) * 100 if high_p != low_p else 0
            is_green, is_red = close_p > open_p, close_p < open_p
            wick_dominated = (high_p - max(open_p, close_p)) > 0.6 * (high_p - low_p) or (min(open_p, close_p) - low_p) > 0.6 * (high_p - low_p)
            volume_surge = float(latest['volume']) / btc_5m['volume'].rolling(10).mean().iloc[-1] if btc_5m['volume'].rolling(10).mean().iloc[-1] > 0 else 1.0
            impulse_detected = (body_pct > 60) and (volume_surge > 1.5)

            ema_fast_5m = Indicators.ema(btc_5m['close'], 8).iloc[-1]
            ema_slow_5m = Indicators.ema(btc_5m['close'], 21).iloc[-1]
            ema_fast_15m = Indicators.ema(btc_15m['close'], 12).iloc[-1]
            ema_slow_15m = Indicators.ema(btc_15m['close'], 26).iloc[-1]
            ema_fast_1h = Indicators.ema(btc_1h['close'], 20).iloc[-1]
            ema_slow_1h = Indicators.ema(btc_1h['close'], 50).iloc[-1]

            mtf_bullish = ema_fast_5m > ema_slow_5m and ema_fast_15m > ema_slow_15m and ema_fast_1h > ema_slow_1h
            mtf_bearish = ema_fast_5m < ema_slow_5m and ema_fast_15m < ema_slow_15m and ema_fast_1h < ema_slow_1h

            if wick_dominated:
                direction, strength = "RANGING", "WEAK"
            elif is_green and impulse_detected and mtf_bullish:
                direction, strength = "BULLISH", "STRONG_IMPULSE"
            elif is_red and impulse_detected and mtf_bearish:
                direction, strength = "BEARISH", "STRONG_IMPULSE"
            elif is_green:
                direction, strength = ("BULLISH", "MODERATE") if mtf_bullish else ("BULLISH", "MILD")
            elif is_red:
                direction, strength = ("BEARISH", "MODERATE") if mtf_bearish else ("BEARISH", "MILD")
            else:
                direction, strength = "RANGING", "NEUTRAL"

            return {
                'valid_context': True, 'direction': direction, 'strength': strength,
                'impulse': impulse_detected, 'wick_dominated': wick_dominated,
                'volume_surge': volume_surge, 'mtf_aligned': mtf_bullish or mtf_bearish,
                'candle_color': 'GREEN' if is_green else 'RED' if is_red else 'DOJI'
            }
        except Exception as e:
            print(f"⚠️ BTC context fetch failed: {e}")
            return {'valid_context': False, 'reason': str(e)}

    def _check_btc_alignment(self, signal_direction: str, btc_context: Dict) -> Tuple[bool, str]:
        if not btc_context.get('valid_context'):
            return True, "BTC_DATA_UNAVAILABLE (allowing signal)"
        btc_dir = btc_context.get('direction', 'RANGING')
        btc_strength = btc_context.get('strength', 'WEAK')
        impulse = btc_context.get('impulse', False)
        wick_dominated = btc_context.get('wick_dominated', False)
        candle_color = btc_context.get('candle_color', 'DOJI')

        if wick_dominated or btc_dir == "RANGING":
            return True, f"BTC_RANGING (ignoring, {candle_color} wick-dominated)"
        if btc_strength == "STRONG_IMPULSE":
            if signal_direction == "LONG" and btc_dir == "BEARISH":
                return False, f"BTC_STRONG_BEARISH_IMPULSE (blocks LONG)"
            if signal_direction == "SHORT" and btc_dir == "BULLISH":
                return False, f"BTC_STRONG_BULLISH_IMPULSE (blocks SHORT)"
        if not wick_dominated and impulse:
            if signal_direction == "SHORT" and candle_color == "GREEN":
                return False, f"BTC_GREEN_IMPULSE (blocks SHORT)"
            if signal_direction == "LONG" and candle_color == "RED":
                return False, f"BTC_RED_IMPULSE (blocks LONG)"
        if btc_strength in ["MILD", "MODERATE", "WEAK"]:
            return True, f"BTC_{btc_dir}_{btc_strength} (context only, allowing)"
        return True, f"BTC_{btc_dir} (no conflict)"

    def _check_hard_rejections(self, signal: Dict, btc_context: Dict) -> Tuple[bool, List[Tuple[str, str]]]:
        hard_reasons = []
        direction = signal.get("direction", "UNKNOWN")
        rsi = float(signal.get("rsi", 50))
        adx = float(signal.get("adx", 20))
        entry = float(signal.get("entry", 0))
        sl = float(signal.get("sl", 0))
        tp1 = float(signal.get("tp1", 0))

        sl_distance = abs(entry - sl) / entry * 100 if entry > 0 else 0
        tp1_distance = abs(tp1 - entry) / entry * 100 if entry > 0 else 0
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0

        if entry <= 0 or sl <= 0 or tp1 <= 0:
            hard_reasons.append(("HARD_INVALID_DATA", "Missing or invalid price levels"))
        if rr_ratio < 1.5:
            hard_reasons.append(("HARD_POOR_RR", f"RR={rr_ratio:.2f} < 1.5"))
        if sl_distance < 0.8:
            hard_reasons.append(("HARD_SL_TOO_TIGHT", f"SL={sl_distance:.2f}% < 0.8%"))
        if adx > 50:
            if rsi > 78:
                hard_reasons.append(("HARD_EXHAUSTION", f"RSI={rsi:.1f} > 78 with ADX={adx:.1f}"))
            elif rsi < 22:
                hard_reasons.append(("HARD_EXHAUSTION", f"RSI={rsi:.1f} < 22 with ADX={adx:.1f}"))
        btc_allowed, btc_reason = self._check_btc_alignment(direction, btc_context)
        if not btc_allowed:
            hard_reasons.append(("HARD_BTC_CONFLICT", btc_reason))
        return len(hard_reasons) > 0, hard_reasons

    # ------------------------------------------------------------------
    # NEW: _log_decision (exact old copy)
    # ------------------------------------------------------------------
    def _log_decision(self, signal: Dict, approved: bool, reason: str, hard_reasons: List, btc_context: Dict):
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        score = signal.get("score", 0)
        mode = signal.get("mode", "UNKNOWN")
        status_symbol = "✅" if approved else "🚫"
        status_text = "APPROVED" if approved else "REJECTED"
        print(f"\n{'='*75}")
        print(f"{status_symbol} {status_text}: {pair} {direction} ({mode}, Score: {score})")
        print(f"{'='*75}")
        print(f"🎯 DECISION: {reason}")
        if btc_context.get('valid_context'):
            btc_dir = btc_context.get('direction', 'UNKNOWN')
            btc_str = btc_context.get('strength', 'UNKNOWN')
            print(f"📊 BTC Context: {btc_dir} ({btc_str})")
        if hard_reasons:
            print(f"\n🔴 HARD BLOCKS ({len(hard_reasons)}):")
            for code, msg in hard_reasons:
                print(f"   • [{code}] {msg}")
        print(f"{'='*75}\n")

    # ------------------------------------------------------------------
    # MAIN DECISION FLOW (10-PRO enabled)
    # ------------------------------------------------------------------
    def final_trade_decision(self, signal: Dict, candles_data: Dict = None) -> bool:
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        mode = signal.get("mode", "UNKNOWN")
        score = int(signal.get("score", 0))
        print(f"\n🎯 10-PRO SNIPER: {pair} {direction} ({mode}, Score: {score})")

        btc_context = self._get_btc_context()
        has_hard_reject, hard_reasons = self._check_hard_rejections(signal, btc_context)
        if has_hard_reject:
            self._log_decision(signal, False, "HARD_REJECT (non-negotiable)", hard_reasons, btc_context)
            return False

        # 10-PRO single call
        prompt = self._build_10pro_prompt(signal, btc_context)
        messages = [
            {"role": "system", "content": "You are a 10-year crypto sniper. Return ONLY the requested JSON."},
            {"role": "user", "content": prompt}
        ]
        print(f"🤖 Consulting ChatGPT 10-PRO sniper...")
        response = self._call_chatgpt_with_timeout(messages)
        pro_data = self._parse_10pro(response)

        approved = pro_data.get("approved", False)
        reason = "10PRO_APPROVED" if approved else "10PRO_REJECTED"
        if approved:
            print("✅ 10-PRO keys:", {k: v for k, v in pro_data.items() if k != "approved"})
        else:
            print("⚠️ 10-PRO rejected")
        self._log_decision(signal, approved, reason, [], btc_context)
        return approved

    # ------------------------------------------------------------------
    # legacy
    # ------------------------------------------------------------------
    def validate_signal_with_traps(self, signal: Dict) -> Dict:
        approved = self.final_trade_decision(signal)
        return {"approved": approved, "reason": "10Pro sniper decision", "confidence": 100 if approved else 0}
