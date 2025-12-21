from openai import OpenAI
from typing import Dict, List
from config import config


class ChatGPTAdvisor:
    """
    ADVANCED CHATGPT VALIDATOR (FULL VERSION)
    -----------------------------------------
    - SL distance awareness
    - Trap severity scoring
    - Indicator strength scoring
    - Strict decision parsing
    - Multi-layer security checks
    - No aggressive prompts
    - No risky overrides
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL


    # -----------------------------------
    # BASIC CHATGPT CALL HANDLER
    # -----------------------------------
    def _call_chatgpt(self, messages: List[Dict]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=350,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ ChatGPT API error: {e}")
            return "ERROR"


    # -----------------------------------
    # INTERNAL SCORING SYSTEM
    # -----------------------------------
    def _trap_severity_score(self, traps: List[str]) -> int:
        score = 0
        for t in traps:
            if "liquidity" in t:
                score += 3
            elif "market_maker" in t:
                score += 4
            elif "reversal" in t:
                score += 5
            else:
                score += 2
        return score


    def _indicator_strength(self, rsi, adx, sl_pct):
        score = 0

        # RSI
        if 45 <= rsi <= 55:
            score += 20
        elif 40 <= rsi <= 60:
            score += 10
        else:
            score += 0

        # ADX
        if adx >= 35:
            score += 20
        elif adx >= 25:
            score += 10

        # SL distance
        if sl_pct >= 1.5:
            score += 20
        elif sl_pct >= 1.0:
            score += 10

        return score


    # -----------------------------------
    # MAIN FUNCTION: ChatGPT Validation
    # -----------------------------------
    def validate_signal_with_traps(self, signal: Dict) -> Dict:

        entry = float(signal["entry"])
        sl = float(signal["sl"])
        rsi = float(signal["rsi"])
        adx = float(signal["adx"])
        traps = signal["trap_reasons"]
        trap_count = signal["trapped_count"]

        sl_pct = abs(entry - sl) / entry * 100

        # ---------------------------
        # HARD RULES (Unbreakable)
        # ---------------------------
        if sl_pct < 1.0:
            return {"approved": False, "reason": f"SL too close ({sl_pct:.2f}%)"}

        if rsi > 75 or rsi < 25:
            return {"approved": False, "reason": "RSI extreme zone"}

        if adx < 25:
            return {"approved": False, "reason": "ADX weak"}

        # ---------------------------
        # INTERNAL AUTO SCORING (Local check before AI)
        # ---------------------------
        trap_severity = self._trap_severity_score(traps)
        indicator_score = self._indicator_strength(rsi, adx, sl_pct)

        # If traps heavy + indicators weak → auto skip
        if trap_severity >= 8 and indicator_score < 25:
            return {"approved": False, "reason": "Trap severity high"}

        # ---------------------------
        # BUILD CHATGPT PROMPT (Safe)
        # ---------------------------
        prompt = f"""
You are a **risk-first cryptocurrency futures validator**.

### TRADE DETAILS
Direction: {signal['direction']}
Entry: {entry}
Stop Loss: {sl}
SL Distance: {sl_pct:.2f}%
RSI: {rsi}
ADX: {adx}

### TRAPS
Count: {trap_count}
List: {", ".join(traps)}

### INTERNAL PRE-SCORES
Trap Severity Score: {trap_severity}/15
Indicator Strength Score: {indicator_score}/60

### RULES (Important)
1. If SL distance < 1% → ALWAYS SKIP.
2. If RSI > 70 or < 30 → SKIP.
3. If ADX < 25 → SKIP.
4. 2 traps allowed ONLY IF:
       - RSI 45–55
       - ADX > 35
       - SL distance > 1.2%
5. NEVER approve a trade that increases liquidation risk.
6. Trend confirmation > trap override.

### REQUIRED OUTPUT (Exact format)
DECISION: TAKE or SKIP
CONFIDENCE: (0–100)%
REASON: One short sentence
"""

        messages = [
            {"role": "system", 
             "content": "You ALWAYS prioritize risk. You NEVER approve unsafe trades."},
            {"role": "user", "content": prompt}
        ]

        # ---------------------------
        # CALL CHATGPT
        # ---------------------------
        response = self._call_chatgpt(messages)
        print("🤖 ChatGPT Full Response:\n", response)

        if response == "ERROR":
            return {"approved": False, "reason": "ChatGPT unavailable"}


        # ---------------------------
        # STRICT PARSING
        # ---------------------------
        try:
            decision = response.split("DECISION:")[1].split("\n")[0].strip().upper()
            conf_raw = response.split("CONFIDENCE:")[1].split("%")[0].strip()
            confidence = int(''.join(filter(str.isdigit, conf_raw)))
        except:
            return {"approved": False, "reason": "Parsing error"}

        # ---------------------------
        # FINAL DECISION RULES
        # ---------------------------
        if decision != "TAKE":
            return {"approved": False, "reason": "AI decided SKIP", "confidence": confidence}

        if confidence < 70:
            return {"approved": False, "reason": f"Low confidence ({confidence}%)"}

        # Extra protection for 2 traps
        if trap_count == 2:
            if not (45 <= rsi <= 55 and adx > 35 and sl_pct > 1.2):
                return {"approved": False, "reason": "2 traps but weak indicators"}

        return {
            "approved": True,
            "reason": "AI approved safely",
            "confidence": confidence
        }