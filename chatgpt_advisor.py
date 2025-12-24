from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from config import config
import time


class ChatGPTAdvisor:
    """
    CHATGPT SNIPER FILTER - FINAL DECISION LAYER
    ---------------------------------------------
    
    PHILOSOPHY: Discretionary Human Trader
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


    def _call_chatgpt_with_timeout(self, messages: List[Dict]) -> Optional[str]:
        """Call ChatGPT with timeout protection."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.2,
                    timeout=self.timeout
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ ChatGPT attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                continue
        return None


    def _parse_decision(self, response: str) -> bool:
        """Parse ChatGPT response - STRICT JSON only."""
        if not response:
            return False
        
        try:
            import json
            response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)
            
            if "approved" not in data:
                print("⚠️ ChatGPT response missing 'approved' field")
                return False
            
            return bool(data["approved"])
        except Exception as e:
            print(f"⚠️ ChatGPT parse error: {e}")
            return False


    def _get_btc_context(self) -> Dict:
        """
        Get BTC market context for signal validation.
        Returns: {direction, strength, impulse, valid_context}
        """
        try:
            from coindcx_api import CoinDCXAPI
            
            # Get BTC 5m, 15m, 1h candles
            btc_5m = CoinDCXAPI.get_candles('BTCUSDT', '5m', 20)
            btc_15m = CoinDCXAPI.get_candles('BTCUSDT', '15m', 20)
            btc_1h = CoinDCXAPI.get_candles('BTCUSDT', '1h', 20)
            
            if btc_5m.empty or btc_15m.empty or btc_1h.empty:
                return {'valid_context': False, 'reason': 'BTC data unavailable'}
            
            # Current candle analysis (5m)
            latest = btc_5m.iloc[-1]
            open_price = float(latest['open'])
            close_price = float(latest['close'])
            high_price = float(latest['high'])
            low_price = float(latest['low'])
            volume = float(latest['volume'])
            
            # Calculate candle metrics
            body = abs(close_price - open_price)
            total_range = high_price - low_price
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            
            body_pct = (body / total_range * 100) if total_range > 0 else 0
            upper_wick_pct = (upper_wick / total_range * 100) if total_range > 0 else 0
            lower_wick_pct = (lower_wick / total_range * 100) if total_range > 0 else 0
            
            # Determine candle direction
            is_green = close_price > open_price
            is_red = close_price < open_price
            
            # Check if wick-dominated (ignore color)
            wick_dominated = (upper_wick_pct > 60) or (lower_wick_pct > 60)
            
            # Volume expansion check
            avg_volume = btc_5m['volume'].rolling(10).mean().iloc[-1]
            volume_surge = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Impulse detection (strong move)
            impulse_detected = (body_pct > 60) and (volume_surge > 1.5)
            
            # Multi-timeframe alignment
            from indicators import Indicators
            
            ema_fast_5m = Indicators.ema(btc_5m['close'], 8).iloc[-1]
            ema_slow_5m = Indicators.ema(btc_5m['close'], 21).iloc[-1]
            
            ema_fast_15m = Indicators.ema(btc_15m['close'], 12).iloc[-1]
            ema_slow_15m = Indicators.ema(btc_15m['close'], 26).iloc[-1]
            
            ema_fast_1h = Indicators.ema(btc_1h['close'], 20).iloc[-1]
            ema_slow_1h = Indicators.ema(btc_1h['close'], 50).iloc[-1]
            
            mtf_bullish = (ema_fast_5m > ema_slow_5m and 
                          ema_fast_15m > ema_slow_15m and 
                          ema_fast_1h > ema_slow_1h)
            
            mtf_bearish = (ema_fast_5m < ema_slow_5m and 
                          ema_fast_15m < ema_slow_15m and 
                          ema_fast_1h < ema_slow_1h)
            
            # Determine BTC context
            if wick_dominated:
                direction = "RANGING"
                strength = "WEAK"
            elif is_green and impulse_detected and mtf_bullish:
                direction = "BULLISH"
                strength = "STRONG_IMPULSE"
            elif is_red and impulse_detected and mtf_bearish:
                direction = "BEARISH"
                strength = "STRONG_IMPULSE"
            elif is_green:
                direction = "BULLISH"
                strength = "MILD" if not mtf_bullish else "MODERATE"
            elif is_red:
                direction = "BEARISH"
                strength = "MILD" if not mtf_bearish else "MODERATE"
            else:
                direction = "RANGING"
                strength = "NEUTRAL"
            
            return {
                'valid_context': True,
                'direction': direction,
                'strength': strength,
                'impulse': impulse_detected,
                'wick_dominated': wick_dominated,
                'volume_surge': volume_surge,
                'mtf_aligned': mtf_bullish or mtf_bearish,
                'candle_color': 'GREEN' if is_green else 'RED' if is_red else 'DOJI'
            }
            
        except Exception as e:
            print(f"⚠️ BTC context fetch failed: {e}")
            return {'valid_context': False, 'reason': str(e)}


    def _check_btc_alignment(self, signal_direction: str, btc_context: Dict) -> Tuple[bool, str]:
        """
        Check if BTC context allows signal.
        
        Returns: (is_allowed, reason)
        """
        if not btc_context.get('valid_context'):
            # BTC data unavailable - allow signal (don't block on missing data)
            return True, "BTC_DATA_UNAVAILABLE (allowing signal)"
        
        btc_dir = btc_context.get('direction', 'RANGING')
        btc_strength = btc_context.get('strength', 'WEAK')
        impulse = btc_context.get('impulse', False)
        wick_dominated = btc_context.get('wick_dominated', False)
        candle_color = btc_context.get('candle_color', 'DOJI')
        
        # IGNORE BTC if wick-dominated or ranging
        if wick_dominated or btc_dir == "RANGING":
            return True, f"BTC_RANGING (ignoring, {candle_color} wick-dominated)"
        
        # HARD BLOCK: BTC strong impulse AGAINST signal
        if btc_strength == "STRONG_IMPULSE":
            if signal_direction == "LONG" and btc_dir == "BEARISH":
                return False, f"BTC_STRONG_BEARISH_IMPULSE (blocks LONG)"
            elif signal_direction == "SHORT" and btc_dir == "BULLISH":
                return False, f"BTC_STRONG_BULLISH_IMPULSE (blocks SHORT)"
        
        # CANDLE COLOR RULE: Only block if IMPULSE detected
        # (Don't block MILD/MODERATE moves based on color alone)
        if not wick_dominated and impulse:
            if signal_direction == "SHORT" and candle_color == "GREEN":
                return False, f"BTC_GREEN_IMPULSE (blocks SHORT)"
            elif signal_direction == "LONG" and candle_color == "RED":
                return False, f"BTC_RED_IMPULSE (blocks LONG)"
        
        # MILD/MODERATE moves - allow (context only)
        if btc_strength in ["MILD", "MODERATE", "WEAK"]:
            return True, f"BTC_{btc_dir}_{btc_strength} (context only, allowing)"
        
        # Default: allow
        return True, f"BTC_{btc_dir} (no conflict)"


    def _check_hard_rejections(self, signal: Dict, btc_context: Dict) -> Tuple[bool, List[Tuple[str, str]]]:
        """
        Check HARD rejection criteria ONLY.
        These are NON-NEGOTIABLE blocks.
        
        Returns: (has_hard_reject, hard_reasons)
        """
        hard_reasons = []
        
        # Extract data
        direction = signal.get("direction", "UNKNOWN")
        rsi = float(signal.get("rsi", 50))
        adx = float(signal.get("adx", 20))
        entry = float(signal.get("entry", 0))
        sl = float(signal.get("sl", 0))
        tp1 = float(signal.get("tp1", 0))
        mode = signal.get("mode", "UNKNOWN")
        
        # Calculate metrics
        sl_distance = abs(entry - sl) / entry * 100 if entry > 0 else 0
        tp1_distance = abs(tp1 - entry) / entry * 100 if entry > 0 else 0
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0
        
        # HARD 1: Invalid price data
        if entry <= 0 or sl <= 0 or tp1 <= 0:
            hard_reasons.append(("HARD_INVALID_DATA", "Missing or invalid price levels"))
        
        # HARD 2: Poor RR
        if rr_ratio < 1.5:
            hard_reasons.append(("HARD_POOR_RR", f"RR={rr_ratio:.2f} < 1.5"))
        
        # HARD 3: SL too tight
        if sl_distance < 0.8:
            hard_reasons.append(("HARD_SL_TOO_TIGHT", f"SL={sl_distance:.2f}% < 0.8%"))
        
        # HARD 4: TRUE exhaustion (RARE - very extreme)
        if adx > 50:
            if rsi > 78:
                hard_reasons.append(("HARD_EXHAUSTION", f"RSI={rsi:.1f} > 78 with ADX={adx:.1f}"))
            elif rsi < 22:
                hard_reasons.append(("HARD_EXHAUSTION", f"RSI={rsi:.1f} < 22 with ADX={adx:.1f}"))
        
        # HARD 5: BTC strong impulse AGAINST signal
        btc_allowed, btc_reason = self._check_btc_alignment(direction, btc_context)
        if not btc_allowed:
            hard_reasons.append(("HARD_BTC_CONFLICT", btc_reason))
        
        return len(hard_reasons) > 0, hard_reasons


    def _build_sniper_prompt(self, signal: Dict, btc_context: Dict) -> str:
        """Build ChatGPT sniper prompt with full context."""
        
        direction = signal.get("direction", "UNKNOWN")
        entry = float(signal.get("entry", 0))
        sl = float(signal.get("sl", 0))
        tp1 = float(signal.get("tp1", 0))
        rsi = float(signal.get("rsi", 50))
        adx = float(signal.get("adx", 20))
        volume_surge = float(signal.get("volume_surge", 1.0))
        score = int(signal.get("score", 0))
        mode = signal.get("mode", "UNKNOWN")
        
        sl_distance = abs(entry - sl) / entry * 100
        tp1_distance = abs(tp1 - entry) / entry * 100
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0
        
        # BTC context summary
        btc_summary = "BTC_UNAVAILABLE"
        if btc_context.get('valid_context'):
            btc_dir = btc_context.get('direction', 'UNKNOWN')
            btc_str = btc_context.get('strength', 'UNKNOWN')
            btc_summary = f"{btc_dir} ({btc_str})"
        
        return f"""You are a 10-year crypto futures SNIPER - discretionary human trader.

SIGNAL:
Pair: {signal.get('pair', 'UNKNOWN')}
Direction: {direction}
Mode: {mode}
Entry: {entry}
SL: {sl} ({sl_distance:.2f}% away)
TP1: {tp1} (R/R: {rr_ratio:.2f})

INDICATORS:
RSI: {rsi}
ADX: {adx}
Volume: {volume_surge:.2f}x
Score: {score}/100

BTC CONTEXT: {btc_summary}

YOUR MINDSET (CRITICAL):
- You are a SNIPER, not a machine
- Probability > Indicator Perfection
- Minor imperfections are NORMAL in real trading
- Context matters more than isolated indicators
- TREND continuation (ADX ≥ 30, RR ≥ 2.0) is VALID even if late
- Volume 1.2x+ is acceptable (not every trade has 2x volume)
- RSI 25-75 is tradable range (extremes already filtered)
- Mixed signals are normal - look for EDGE, not perfection

APPROVE IF:
- Edge exists (trend + structure + adequate RR)
- Setup is tradable (not perfect, but probability favors it)
- Risk is managed (RR ≥ 1.5, SL ≥ 0.8%)
- No severe conflicts (BTC strongly against, etc.)

REJECT ONLY IF:
- No clear edge or probability advantage
- Severe quality issues (not minor imperfections)
- Strong counter-evidence

Think like a human trader making 10-15 trades/day.
NOT a perfectionist machine making 0 trades.

RESPOND WITH STRICT JSON ONLY:
{{"approved": true}}  OR  {{"approved": false}}"""


    def _log_decision(self, signal: Dict, approved: bool, reason: str, hard_reasons: List, btc_context: Dict):
        """Log decision with context."""
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


    def final_trade_decision(self, signal: Dict, candles_data: Dict = None) -> bool:
        """
        🎯 SNIPER FILTER - Final Decision
        
        Flow:
        1. Check HARD rejections (if any → BLOCK immediately)
        2. Get BTC context
        3. Consult ChatGPT sniper
        4. Return decision
        
        Returns: True = APPROVED, False = REJECTED
        """
        
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        mode = signal.get("mode", "UNKNOWN")
        score = int(signal.get("score", 0))
        
        print(f"\n🎯 SNIPER FILTER: {pair} {direction} ({mode}, Score: {score})")
        
        # STEP 1: Get BTC context
        btc_context = self._get_btc_context()
        
        # STEP 2: Check HARD rejections
        has_hard_reject, hard_reasons = self._check_hard_rejections(signal, btc_context)
        
        if has_hard_reject:
            self._log_decision(signal, False, "HARD_REJECT (non-negotiable)", hard_reasons, btc_context)
            return False
        
        # STEP 3: Build sniper prompt
        prompt = self._build_sniper_prompt(signal, btc_context)
        
        messages = [
            {
                "role": "system",
                "content": "You are a SNIPER trader. Respond ONLY with JSON: {\"approved\": true/false}. Think probability, not perfection. Goal: 10-15 trades/day."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # STEP 4: Consult ChatGPT
        print(f"🤖 Consulting ChatGPT sniper...")
        response = self._call_chatgpt_with_timeout(messages)
        
        if response is None:
            # ChatGPT timeout - REJECT (safety first)
            self._log_decision(signal, False, "CHATGPT_TIMEOUT", [("TIMEOUT", "API unavailable")], btc_context)
            return False
        
        # Parse decision
        approved = self._parse_decision(response)
        
        reason = "SNIPER_APPROVED" if approved else "SNIPER_REJECTED"
        self._log_decision(signal, approved, reason, [], btc_context)
        
        return approved


    def validate_signal_with_traps(self, signal: Dict) -> Dict:
        """Legacy compatibility."""
        approved = self.final_trade_decision(signal)
        return {
            "approved": approved,
            "reason": "Sniper decision",
            "confidence": 100 if approved else 0
        }