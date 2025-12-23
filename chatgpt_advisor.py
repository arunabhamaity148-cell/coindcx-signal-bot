from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from config import config
import time
import json


class ChatGPTAdvisor:
    """
    ChatGPT-driven discretionary filter.
    HARD rejects are RARE and FINAL.
    ChatGPT approval overrides everything else.
    Goal: 10-15 high-probability trades/day.
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 8
        self.max_retries = 2

    # ------------------------------------------------------------------
    # low-level helpers
    # ------------------------------------------------------------------
    def _call_chatgpt(self, messages: List[Dict]) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                return (
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=60,
                        temperature=0.2,
                        timeout=self.timeout,
                    )
                    .choices[0]
                    .message.content.strip()
                )
            except Exception as e:
                print(f"⚠️  ChatGPT attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        return None

    def _parse_bool(self, txt: str) -> bool:
        if not txt:
            return False
        try:
            txt = txt.replace("```json", "").replace("```", "").strip()
            return bool(json.loads(txt)["approved"])
        except Exception as e:
            print(f"⚠️  JSON parse error: {e}  –  treating as reject")
            return False

    # ------------------------------------------------------------------
    # HARD rejection logic – STRICT & RARE
    # ------------------------------------------------------------------
    def _hard_reject_reasons(self, signal: Dict, candles_data: Optional[Dict]) -> List[str]:
        reasons: List[str] = []

        entry = float(signal.get("entry", 0))
        sl  = float(signal.get("sl", 0))
        tp1 = float(signal.get("tp1", 0))
        rsi = float(signal.get("rsi", 50))
        adx = float(signal.get("adx", 20))

        sl_dist = abs(entry - sl) / entry * 100 if entry else 0
        tp_dist = abs(tp1 - entry) / entry * 100 if entry else 0
        rr = tp_dist / sl_dist if sl_dist else 0

        # 1) RR check
        if rr < 1.5:
            reasons.append(f"RR={rr:.2f}<1.5")

        # 2) SL too tight
        if sl_dist < 0.8:
            reasons.append(f"SL-dist={sl_dist:.2f}%<0.8%")

        # 3) TRUE exhaustion
        if adx > 50 and (rsi > 78 or rsi < 22):
            reasons.append(f"Exhaustion RSI={rsi:.1f}+ADX={adx:.1f}")

        # 4) Data invalid
        if entry == 0 or tp1 == 0 or sl == 0:
            reasons.append("Invalid-price-data")

        # NOTE: late-entry is NOT a hard reject – handled in prompt
        return reasons

    # ------------------------------------------------------------------
    # Soft / context info for ChatGPT
    # ------------------------------------------------------------------
    def _late_entry_context(self, signal: Dict, candles_data: Optional[Dict]) -> str:
        """Return human-readable late-entry info for prompt."""
        entry = float(signal.get("entry", 0))
        tp1 = float(signal.get("tp1", 0))
        if not entry or not tp1:
            return "Late-entry: unknown (no price data)"

        # if we have live price, compute actual % toward TP1
        if candles_data and "close" in candles_data:
            current = float(candles_data["close"])
        elif candles_data and "current_price" in candles_data:
            current = float(candles_data["current_price"])
        else:
            return "Late-entry: check chart manually (no current price)"

        total_move = abs(tp1 - entry)
        done_move  = abs(current - entry)
        pct = (done_move / total_move) * 100 if total_move else 0
        return f"Price already moved {pct:.0f}% toward TP1" if pct > 0 else "Late-entry: OK"

    # ------------------------------------------------------------------
    # ChatGPT prompt – sniper trader mindset
    # ------------------------------------------------------------------
    def _build_prompt(self, signal: Dict, candles_data: Optional[Dict]) -> str:
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        mode  = signal.get("mode", "UNKNOWN")
        entry = float(signal.get("entry", 0))
        sl    = float(signal.get("sl", 0))
        tp1   = float(signal.get("tp1", 0))
        rsi   = float(signal.get("rsi", 50))
        adx   = float(signal.get("adx", 20))
        vol   = float(signal.get("volume_surge", 1.0))
        score = int(signal.get("score", 0))

        sl_dist = abs(entry - sl) / entry * 100 if entry else 0
        tp_dist = abs(tp1 - entry) / entry * 100 if entry else 0
        rr = tp_dist / sl_dist if sl_dist else 0

        late_info = self._late_entry_context(signal, candles_data)
        trend_lenient = mode.upper() == "TREND" and adx >= 30 and rr >= 2.0

        prompt = f"""You are a calm, discretionary crypto sniper with 10y exp.
Goal: 10-15 good trades/day, not perfection.

PAIR: {pair}   DIR: {direction}   MODE: {mode}
ENTRY: {entry}   SL: {sl}   TP1: {tp1}   RR: {rr:.2f}
RSI: {rsi}   ADX: {adx}   VOL: {vol}x   SCORE: {score}

HARD REJECT (block if ANY true):
- RR<1.5  –  SL-dist<0.8%  –  RSI>78/RSI<22 & ADX>50  –  data bad

SOFT NOTES (use judgement):
- {late_info}
- Low/average volume OK in TREND mode if ADX≥30 & RR≥2
- Mixed MTF, traps, imperfect RSI are warnings only

TREND MODE: if ADX≥30 & RR≥2.0 → continuation acceptable even with weak vol.

ASK: "Would I take this trade with controlled risk?"
Approve tradable setups, not textbook perfection.

JSON ONLY: {{"approved":true}} or {{"approved":false}}"""
        return prompt

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    def final_trade_decision(self, signal: Dict, candles_data: Optional[Dict] = None) -> bool:
        # 1) HARD reject check – rare & final
        hard = self._hard_reject_reasons(signal, candles_data)
        if hard:
            print(f"🚫 HARD_REJECT: {' | '.join(hard)}")
            return False

        # 2) ChatGPT decides – approval overrides everything
        messages = [
            {
                "role": "system",
                "content": "You are an experienced discretionary crypto trader. Answer ONLY with JSON: {\"approved\":true/false}",
            },
            {
                "role": "user",
                "content": self._build_prompt(signal, candles_data),
            },
        ]
        print(f"🤖  ChatGPT sniper check: {signal.get('pair')} {signal.get('direction')} (Score: {signal.get('score')})")
        resp = self._call_chatgpt(messages)
        approved = self._parse_bool(resp)
        print("✅ ChatGPT: APPROVED" if approved else "⚠️  ChatGPT: REJECTED")
        return approved

    # ------------------------------------------------------------------
    # legacy compatibility
    # ------------------------------------------------------------------
    def validate_signal_with_traps(self, signal: Dict) -> Dict:
        approved = self.final_trade_decision(signal)
        return {"approved": approved, "reason": "ChatGPT sniper decision", "confidence": 100 if approved else 0}
