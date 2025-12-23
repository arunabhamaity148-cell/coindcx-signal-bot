from opena import OpenAI
from typing import Dict, List, Optional, Tuple
from config import config
import time


class ChatGPTAdvisor:
    """
    CHATGPT AS QUALITY ADVISOR (NOT FINAL JUDGE)
    ---------------------------------------------
    ChatGPT provides quality assessment, but uses intelligent filtering:
    
    APPROVAL RULES:
    1. Score >= 75: Auto-approve (unless HARD reject)
    2. Score >= 65 + <= 2 soft reasons: Approve
    3. HARD rejection reasons always block
    
    HARD REJECTION REASONS (Non-negotiable):
    - RR < 1.5
    - SL < 0.8%
    - Extreme RSI (>75 or <25) + High ADX (>45)
    - ChatGPT timeout/error
    
    SOFT REJECTION REASONS (Advisory only):
    - Moderate RSI levels
    - Low volume
    - Late entry warnings
    - Weak trend structure
    
    GOAL: 8-12 quality signals/day (profitable, not perfect)
    
    OUTPUT: Strict JSON only
    {"approved": true/false}
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 8
        self.max_retries = 2


    def _call_chatgpt_with_timeout(self, messages: List[Dict]) -> Optional[str]:
        """
        Call ChatGPT with timeout protection.
        Returns None on failure (timeout/error).
        """
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
        """
        Parse ChatGPT response - STRICT JSON only.
        Returns False on any parsing error (safety first).
        """
        if not response:
            return False
        
        try:
            import json
            
            # Remove markdown code blocks if present
            response = response.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate structure
            if "approved" not in data:
                print("⚠️ ChatGPT response missing 'approved' field")
                return False
            
            return bool(data["approved"])
        
        except json.JSONDecodeError as e:
            print(f"⚠️ ChatGPT JSON parse error: {e}")
            print(f"   Response: {response[:200]}")
            return False
        
        except Exception as e:
            print(f"⚠️ ChatGPT response validation error: {e}")
            return False


    def _classify_rejection_reasons(self, signal: Dict) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Classify rejection reasons into HARD and SOFT.
        
        Returns:
            (hard_reasons, soft_reasons)
        """
        hard_reasons = []
        soft_reasons = []
        
        # Extract signal data
        direction = signal.get("direction", "UNKNOWN")
        rsi = float(signal.get("rsi", 50))
        adx = float(signal.get("adx", 20))
        volume_surge = float(signal.get("volume_surge", 1.0))
        entry = float(signal.get("entry", 0))
        sl = float(signal.get("sl", 0))
        tp1 = float(signal.get("tp1", 0))
        
        # Calculate metrics
        sl_distance = abs(entry - sl) / entry * 100 if entry > 0 else 0
        tp1_distance = abs(tp1 - entry) / entry * 100 if entry > 0 else 0
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0
        
        # ================================
        # HARD REJECTION REASONS (Non-negotiable)
        # ================================
        
        # HARD 1: Poor Risk/Reward
        if rr_ratio < 1.5:
            hard_reasons.append((
                "HARD_POOR_RR",
                f"Risk/Reward={rr_ratio:.2f} < 1.5 (unacceptable risk profile)"
            ))
        
        # HARD 2: SL Too Tight
        if sl_distance < 0.8:
            hard_reasons.append((
                "HARD_SL_TOO_TIGHT",
                f"SL distance={sl_distance:.2f}% < 0.8% (extreme whipsaw risk)"
            ))
        
        # HARD 3: Extreme RSI + High ADX (true exhaustion)
        if adx >= 45:
            if rsi >= 75:
                hard_reasons.append((
                    "HARD_EXHAUSTED_TREND",
                    f"RSI={rsi:.1f} >= 75 with ADX={adx:.1f} (severe overbought exhaustion)"
                ))
            elif rsi <= 25:
                hard_reasons.append((
                    "HARD_EXHAUSTED_TREND",
                    f"RSI={rsi:.1f} <= 25 with ADX={adx:.1f} (severe oversold exhaustion)"
                ))
        
        # ================================
        # SOFT REJECTION REASONS (Advisory)
        # ================================
        
        # SOFT 1: Moderate RSI Extremes
        if 70 <= rsi < 75:
            soft_reasons.append((
                "SOFT_RSI_OVERBOUGHT",
                f"RSI={rsi:.1f} in 70-75 range (moderate overbought)"
            ))
        elif 25 < rsi <= 30:
            soft_reasons.append((
                "SOFT_RSI_OVERSOLD",
                f"RSI={rsi:.1f} in 25-30 range (moderate oversold)"
            ))
        
        # SOFT 2: Low Volume
        if volume_surge < 1.2:
            soft_reasons.append((
                "SOFT_LOW_VOLUME",
                f"Volume surge={volume_surge:.2f}x < 1.2x (weak confirmation)"
            ))
        elif volume_surge < 1.5 and ((direction == "LONG" and rsi >= 60) or (direction == "SHORT" and rsi <= 40)):
            soft_reasons.append((
                "SOFT_LOW_VOLUME",
                f"Volume surge={volume_surge:.2f}x < 1.5x with unfavorable RSI={rsi:.1f}"
            ))
        
        # SOFT 3: Late Entry (moderate ADX)
        if 40 <= adx < 45:
            if direction == "LONG" and rsi >= 65:
                soft_reasons.append((
                    "SOFT_LATE_ENTRY",
                    f"LONG at RSI={rsi:.1f} with ADX={adx:.1f} (late in uptrend)"
                ))
            elif direction == "SHORT" and rsi <= 35:
                soft_reasons.append((
                    "SOFT_LATE_ENTRY",
                    f"SHORT at RSI={rsi:.1f} with ADX={adx:.1f} (late in downtrend)"
                ))
        
        # SOFT 4: Weak Trend Structure
        if adx < 25:
            if rsi >= 65 or rsi <= 35:
                soft_reasons.append((
                    "SOFT_WEAK_TREND",
                    f"ADX={adx:.1f} < 25 with RSI={rsi:.1f} (weak trend structure)"
                ))
        
        # SOFT 5: Marginal RR with weak setup
        if 1.5 <= rr_ratio < 2.0:
            if adx < 28 or volume_surge < 1.3:
                soft_reasons.append((
                    "SOFT_MARGINAL_RR",
                    f"R/R={rr_ratio:.2f} marginal with weak setup (ADX={adx:.1f}, Vol={volume_surge:.2f}x)"
                ))
        
        # SOFT 6: SL Tight (but acceptable)
        if 0.8 <= sl_distance < 1.0:
            soft_reasons.append((
                "SOFT_SL_TIGHT",
                f"SL distance={sl_distance:.2f}% tight but acceptable"
            ))
        
        # SOFT 7: Chasing Price
        if direction == "LONG" and rsi >= 62 and adx >= 32 and volume_surge < 1.5:
            soft_reasons.append((
                "SOFT_CHASING_PRICE",
                f"LONG at RSI={rsi:.1f} without strong pullback confirmation"
            ))
        elif direction == "SHORT" and rsi <= 38 and adx >= 32 and volume_surge < 1.5:
            soft_reasons.append((
                "SOFT_CHASING_PRICE",
                f"SHORT at RSI={rsi:.1f} without strong pullback confirmation"
            ))
        
        # SOFT 8: Overextended Momentum
        if adx >= 52:
            soft_reasons.append((
                "SOFT_OVEREXTENDED",
                f"ADX={adx:.1f} >= 52 (trend may be near exhaustion)"
            ))
        
        return hard_reasons, soft_reasons


    def _apply_intelligent_filtering(self, signal: Dict, hard_reasons: List, soft_reasons: List, chatgpt_approved: bool) -> Tuple[bool, str]:
        """
        Apply intelligent filtering rules.
        
        Returns:
            (final_approved, approval_reason)
        """
        score = int(signal.get("score", 0))
        
        # RULE 1: HARD rejections always block
        if hard_reasons:
            return False, f"HARD_REJECT ({len(hard_reasons)} critical issues)"
        
        # RULE 2: Score >= 75 - Auto-approve (high quality)
        if score >= 75:
            return True, f"AUTO_APPROVED (Score={score} >= 75, high quality)"
        
        # RULE 3: Score >= 65 with <= 2 soft reasons - Approve
        if score >= 65 and len(soft_reasons) <= 2:
            return True, f"APPROVED (Score={score} >= 65, {len(soft_reasons)} minor issues acceptable)"
        
        # RULE 4: ChatGPT approved + Score >= 60 + <= 3 soft reasons
        if chatgpt_approved and score >= 60 and len(soft_reasons) <= 3:
            return True, f"CHATGPT_APPROVED (Score={score}, {len(soft_reasons)} minor issues)"
        
        # RULE 5: Excellent fundamentals (low soft reasons) despite lower score
        if score >= 55 and len(soft_reasons) <= 1:
            return True, f"APPROVED (Score={score}, minimal issues: {len(soft_reasons)})"
        
        # Otherwise, reject
        rejection_summary = f"Score={score}, {len(soft_reasons)} soft issues"
        if not chatgpt_approved:
            rejection_summary += ", ChatGPT rejected"
        
        return False, f"SOFT_REJECT ({rejection_summary})"


    def _log_decision(self, signal: Dict, approved: bool, reason: str, hard_reasons: List, soft_reasons: List):
        """
        Print decision log with all reasons.
        """
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        score = signal.get("score", 0)
        rsi = signal.get("rsi", 0)
        adx = signal.get("adx", 0)
        volume_surge = signal.get("volume_surge", 0)
        entry = signal.get("entry", 0)
        sl = signal.get("sl", 0)
        tp1 = signal.get("tp1", 0)
        mode = signal.get("mode", "UNKNOWN")
        
        sl_distance = abs(entry - sl) / entry * 100 if entry > 0 else 0
        tp1_distance = abs(tp1 - entry) / entry * 100 if entry > 0 else 0
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0
        
        status_symbol = "✅" if approved else "🚫"
        status_text = "APPROVED" if approved else "REJECTED"
        
        print(f"\n{'='*75}")
        print(f"{status_symbol} SIGNAL {status_text}: {pair} {direction}")
        print(f"{'='*75}")
        print(f"Mode:      {mode}")
        print(f"Score:     {score}/100")
        print(f"Entry:     {entry:,.6f}")
        print(f"SL:        {sl:,.6f} ({sl_distance:.2f}% distance)")
        print(f"TP1:       {tp1:,.6f} (R/R: {rr_ratio:.2f})")
        print(f"\n📊 Indicators:")
        print(f"   RSI:          {rsi:.1f}")
        print(f"   ADX:          {adx:.1f}")
        print(f"   Volume Surge: {volume_surge:.2f}x")
        print(f"\n🎯 DECISION: {reason}")
        
        if hard_reasons:
            print(f"\n🔴 HARD REJECTION REASONS ({len(hard_reasons)}):")
            for i, (reason_code, detail_message) in enumerate(hard_reasons, 1):
                print(f"   {i}. [{reason_code}]")
                print(f"      {detail_message}")
        
        if soft_reasons:
            print(f"\n🟡 SOFT ADVISORY NOTES ({len(soft_reasons)}):")
            for i, (reason_code, detail_message) in enumerate(soft_reasons, 1):
                print(f"   {i}. [{reason_code}]")
                print(f"      {detail_message}")
        
        if not hard_reasons and not soft_reasons and approved:
            print(f"\n✨ Clean signal - no quality concerns")
        
        print(f"{'='*75}\n")


    def final_trade_decision(self, signal: Dict, candles_data: Dict = None) -> bool:
        """
        🎯 QUALITY ADVISOR DECISION (Not Final Judge)
        
        Uses intelligent filtering:
        - Score >= 75: Auto-approve
        - Score >= 65 + <= 2 soft reasons: Approve
        - HARD reasons always block
        
        Args:
            signal: Signal dictionary from rule-based system
            candles_data: Optional recent price action data
        
        Returns:
            True = APPROVED (send signal)
            False = REJECTED (silently drop, no Telegram message)
        """
        
        # Extract signal data
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        entry = float(signal.get("entry", 0))
        sl = float(signal.get("sl", 0))
        tp1 = float(signal.get("tp1", 0))
        tp2 = float(signal.get("tp2", 0))
        rsi = float(signal.get("rsi", 50))
        adx = float(signal.get("adx", 20))
        volume_surge = float(signal.get("volume_surge", 1.0))
        score = int(signal.get("score", 0))
        mode = signal.get("mode", "UNKNOWN")
        
        # Calculate metrics
        sl_distance = abs(entry - sl) / entry * 100
        tp1_distance = abs(tp1 - entry) / entry * 100
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0
        
        # ================================
        # STEP 1: Classify rejection reasons
        # ================================
        hard_reasons, soft_reasons = self._classify_rejection_reasons(signal)
        
        # ================================
        # STEP 2: Check HARD rejections first
        # ================================
        if hard_reasons:
            self._log_decision(signal, False, f"HARD_REJECT ({len(hard_reasons)} critical issues)", hard_reasons, soft_reasons)
            return False
        
        # ================================
        # STEP 3: Get ChatGPT opinion
        # ================================
        prompt = f"""You are an experienced cryptocurrency futures trader with 10+ years of experience.

SIGNAL TO EVALUATE:
Pair: {pair}
Direction: {direction}
Entry: {entry}
Stop Loss: {sl} ({sl_distance:.2f}% away)
Take Profit 1: {tp1} (R/R: {rr_ratio:.2f})
Mode: {mode}

INDICATORS:
RSI: {rsi}
ADX: {adx}
Volume Surge: {volume_surge}x
Rule-based Score: {score}/100

YOUR EXPERTISE - REJECT IF:
1. Late entry (RSI > 65 for LONG, RSI < 35 for SHORT + ADX > 40)
2. Exhausted momentum (RSI > 70 or < 30 with ADX > 45)
3. Low volume trend continuation (volume_surge < 1.2 + RSI > 60)
4. Poor risk/reward (R/R < 1.5 or SL distance < 1%)
5. Chasing price (no pullback confirmation)

PREFER:
- Fresh breakouts with volume confirmation
- Pullback entries in strong trends
- Clean support/resistance breaks
- RSI 40-60 range with rising ADX
- Volume surge > 1.5x on entry

RESPOND WITH STRICT JSON ONLY (no explanation):
{{"approved": true}}  OR  {{"approved": false}}"""

        messages = [
            {
                "role": "system",
                "content": "You are a strict, experienced trader. You ONLY respond with JSON format: {\"approved\": true/false}. No explanations. Reject late entries, exhausted momentum, and low-quality setups."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        print(f"🤖 Consulting ChatGPT for {pair} {direction} (Score: {score})...")
        response = self._call_chatgpt_with_timeout(messages)
        
        # If ChatGPT failed/timeout → Continue with local decision
        if response is None:
            print(f"⚠️ ChatGPT timeout - using local quality filters")
            chatgpt_approved = False
        else:
            chatgpt_approved = self._parse_decision(response)
            if chatgpt_approved:
                print(f"✅ ChatGPT: APPROVED")
            else:
                print(f"⚠️ ChatGPT: REJECTED (advisory)")
        
        # ================================
        # STEP 4: Apply intelligent filtering
        # ================================
        final_approved, approval_reason = self._apply_intelligent_filtering(
            signal, hard_reasons, soft_reasons, chatgpt_approved
        )
        
        # ================================
        # STEP 5: Log decision
        # ================================
        self._log_decision(signal, final_approved, approval_reason, hard_reasons, soft_reasons)
        
        return final_approved


    def validate_signal_with_traps(self, signal: Dict) -> Dict:
        """
        LEGACY FUNCTION - Now redirects to final_trade_decision()
        Kept for backward compatibility with existing trap detection flow.
        """
        approved = self.final_trade_decision(signal)
        
        return {
            "approved": approved,
            "reason": "Quality advisor decision",
            "confidence": 100 if approved else 0
        }