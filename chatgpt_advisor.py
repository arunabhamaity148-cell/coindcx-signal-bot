from openai import OpenAI
from typing import Dict, Optional
from config import config
import time
import json
from datetime import datetime, timedelta


class ChatGPTAdvisor:
    """
    PURE SAFETY VALIDATOR ONLY
    
    Philosophy: Validate hard rules, detect logical errors
    - NO timing judgement
    - NO momentum analysis
    - NO trader-like discretion
    - ONLY validates math and logic
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 5
        self.max_retries = 1
        self.recent_losses = []

    def _check_basic_safety(self, signal: Dict) -> tuple[bool, str]:
        """Hard mathematical safety checks"""
        try:
            entry = float(signal.get('entry', 0))
            sl = float(signal.get('sl', 0))
            tp1 = float(signal.get('tp1', 0))

            if entry <= 0 or sl <= 0 or tp1 <= 0:
                return False, "INVALID_PRICES"

            if abs(sl - entry) < 0.000001:
                return False, "SL_EQUALS_ENTRY"

            sl_distance = abs(entry - sl) / entry * 100
            tp1_distance = abs(tp1 - entry) / entry * 100
            rr = tp1_distance / sl_distance if sl_distance > 0 else 0

            if rr < 1.5:
                return False, f"LOW_RR: {rr:.2f}"

            if sl_distance < 0.8:
                return False, f"SL_TOO_TIGHT: {sl_distance:.2f}%"

            direction = signal.get('direction', 'UNKNOWN')
            if direction == "LONG":
                if sl >= entry or tp1 <= entry:
                    return False, "LONG_LOGIC_ERROR"
            elif direction == "SHORT":
                if sl <= entry or tp1 >= entry:
                    return False, "SHORT_LOGIC_ERROR"

            return True, "BASIC_SAFETY_OK"

        except Exception as e:
            return False, f"SAFETY_ERROR: {str(e)[:50]}"

    def _check_drawdown_protection(self) -> tuple[bool, str]:
        """Check recent losses"""
        try:
            two_hours_ago = datetime.now() - timedelta(hours=2)
            self.recent_losses = [loss for loss in self.recent_losses if loss > two_hours_ago]
            loss_count = len(self.recent_losses)

            if loss_count >= 3:
                return False, f"DRAWDOWN_PAUSE: {loss_count} losses in 2h"

            return True, "DRAWDOWN_OK"

        except:
            return True, "DRAWDOWN_CHECK_SKIPPED"

    def record_loss(self):
        """Record a losing trade"""
        try:
            self.recent_losses.append(datetime.now())
        except:
            pass

    def final_trade_decision(self, signal: Dict, candles=None) -> bool:
        """
        PURE SAFETY VALIDATOR
        
        Returns:
            True = Signal approved (passes safety checks)
            False = Signal rejected (safety violation)
        """
        try:
            pair = signal.get('pair', 'UNKNOWN')
            direction = signal.get('direction', 'UNKNOWN')
            mode = signal.get('mode', 'UNKNOWN')

            print(f"\n{'='*70}")
            print(f"🔒 SAFETY VALIDATOR: {pair} {direction} | {mode}")
            print(f"{'='*70}")

            # STEP 1: Basic safety
            safety_ok, safety_reason = self._check_basic_safety(signal)
            if not safety_ok:
                print(f"❌ BASIC SAFETY FAIL: {safety_reason}")
                print(f"{'='*70}\n")
                return False

            # STEP 2: Drawdown protection
            drawdown_ok, drawdown_reason = self._check_drawdown_protection()
            if not drawdown_ok:
                print(f"❌ {drawdown_reason}")
                print(f"{'='*70}\n")
                return False

            print(f"✅ SAFETY CHECKS PASSED")
            print(f"{'='*70}\n")

            return True

        except Exception as e:
            print(f"⚠️ CRITICAL ERROR: {e}")
            print(f"⚠️ FAIL-SAFE: Signal BLOCKED")
            print(f"{'='*70}\n")
            return False

    def validate_signal_with_traps(self, signal: Dict) -> Dict:
        """Legacy compatibility"""
        try:
            approved = self.final_trade_decision(signal, None)
            return {
                "approved": approved,
                "reason": "Safety validation",
                "confidence": 100 if approved else 0
            }
        except Exception as e:
            return {
                "approved": False,
                "reason": f"Error: {str(e)[:50]}",
                "confidence": 0
            }