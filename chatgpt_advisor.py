from openai import OpenAI
from typing import Dict, List
from config import config

class ChatGPTAdvisor:
    """
    ChatGPT integration for decision-making (OpenAI 1.0+ compatible)
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
    
    def _call_chatgpt(self, messages: List[Dict]) -> str:
        """
        Internal method to call ChatGPT API (OpenAI 1.0+ format)
        
        Args:
            messages: List of message dicts with role and content
        
        Returns:
            ChatGPT response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ ChatGPT API error: {e}")
            return "Error: Unable to get AI advice"
    
    def validate_signal_with_traps(self, signal: Dict) -> Dict:
        """
        INTELLIGENT trap validation - Context matters!
        Even 1-2 traps can be acceptable if setup is strong
        
        Args:
            signal: Signal dict with trap info
        
        Returns:
            {'approved': bool, 'reason': str, 'confidence': int}
        """
        
        trapped_count = signal.get('trapped_count', 0)
        trap_reasons = signal.get('trap_reasons', [])
        
        prompt = f"""
You are an EXPERIENCED crypto futures trader with 10+ years experience.

**TRADE SETUP:**
Pair: {signal['pair']}
Direction: {signal['direction']}
Entry: ₹{signal['entry']:,.2f}
Stop Loss: ₹{signal['sl']:,.2f}

**TECHNICAL INDICATORS:**
RSI: {signal['rsi']} (30-70 = good, outside = extreme)
ADX: {signal['adx']} (25+ = trending, 40+ = strong trend)

**⚠️ TRAP WARNINGS ({trapped_count}):**
{', '.join(trap_reasons)}

**YOUR TASK:**
Decide: TAKE or SKIP this trade

**IMPORTANT CONTEXT:**
1. **Wick Manipulation**: Very common in low liquidity markets (CoinDCX). If other indicators are strong, wick traps can be ignored.

2. **Liquidity Grab**: Often happens before strong moves. If ADX > 30 and RSI is neutral, liquidity grab is actually a GOOD sign.

3. **1-2 traps are NORMAL** in real trading. Perfect setups are rare. Strong trends can override minor traps.

4. **What matters most:**
   - ADX > 30 = Real trend (most important!)
   - RSI 40-60 = Neutral zone (best entries)
   - Direction clear = High confidence

5. **When to SKIP:**
   - RSI extreme (>75 or <25)
   - ADX < 25 (weak/no trend)
   - Multiple traps + weak indicators

**EXAMPLES OF GOOD TRADES WITH TRAPS:**
- Wick manipulation + ADX 35 + RSI 52 = TAKE (strong trend overrides wick)
- Liquidity grab + ADX 40 + Neutral RSI = TAKE (liquidity grab before breakout)
- Both traps + ADX 45 + RSI 50 = TAKE (very strong trend, traps are noise)

**BE PRACTICAL, NOT OVERLY CAUTIOUS!**
Real profitable traders take calculated risks.

Reply in this EXACT format:
DECISION: TAKE or SKIP
CONFIDENCE: [number]%
REASON: [one sentence explaining why]
"""
        
        messages = [
            {
                "role": "system", 
                "content": "You are a profitable crypto trader. You understand that 1-2 traps with strong indicators are often GOOD trades. You are practical and experienced, not overly cautious. You know low liquidity markets have false trap signals."
            },
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._call_chatgpt(messages)
            
            # Parse response
            approved = 'TAKE' in response.upper() and 'SKIP' not in response.upper().split('DECISION:')[1].split('\n')[0]
            
            # Extract confidence
            confidence = 50
            if 'CONFIDENCE:' in response:
                try:
                    conf_part = response.split('CONFIDENCE:')[1].split('%')[0].strip()
                    confidence = int(''.join(filter(str.isdigit, conf_part)))
                except:
                    confidence = 50
            
            # Extract reason
            reason = "Trade analysis complete"
            if 'REASON:' in response:
                try:
                    reason = response.split('REASON:')[1].strip().split('\n')[0]
                except:
                    pass
            
            # Log full ChatGPT response
            print(f"🤖 ChatGPT Full Response:")
            print(f"   {response.replace(chr(10), chr(10) + '   ')}")
            
            # Decision: Approve if confidence >= 55% (lenient)
            final_approved = approved and confidence >= 55
            
            return {
                'approved': final_approved,
                'reason': reason,
                'confidence': confidence,
                'full_response': response
            }
            
        except Exception as e:
            print(f"⚠️ ChatGPT validation error: {e}")
            # On error, be LENIENT (not blocking)
            # If only 1 trap, approve by default
            if trapped_count <= 1:
                return {
                    'approved': True, 
                    'reason': 'Single trap acceptable (ChatGPT unavailable)', 
                    'confidence': 60
                }
            else:
                return {
                    'approved': False, 
                    'reason': 'Multiple traps, ChatGPT unavailable', 
                    'confidence': 30
                }

        """
        Ask ChatGPT to validate a trading signal
        USE SPARINGLY - only for final decision
        
        Args:
            signal: Signal dictionary
        
        Returns:
            Dict with advice and confidence
        """
        
        prompt = f"""
Analyze this futures trading signal briefly:

Pair: {signal['pair']}
Direction: {signal['direction']}
Entry: ₹{signal['entry']}
Stop Loss: ₹{signal['sl']}
Take Profit: ₹{signal['tp1']} / ₹{signal['tp2']}
Leverage: {signal['leverage']}x
Score: {signal['score']}/100

Indicators:
- RSI: {signal['rsi']}
- ADX: {signal['adx']}
- MTF Trend: {signal['mtf_trend']}

Reply in 3 lines:
1. Risk level (Low/Medium/High)
2. Confidence (0-100%)
3. One-line advice
"""
        
        messages = [
            {"role": "system", "content": "You are a trading risk analyst. Be concise."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_chatgpt(messages)
        
        return {
            'advice': response,
            'model': self.model
        }
    
    def suggest_mode(self, market_conditions: Dict) -> str:
        """
        Suggest best trading mode based on market conditions
        
        Args:
            market_conditions: Dict with volatility, trend info
        
        Returns:
            Suggested mode (QUICK/MID/TREND)
        """
        
        prompt = f"""
Current market conditions:
- BTC Volatility: {market_conditions.get('btc_volatility', 'Unknown')}
- Overall Trend: {market_conditions.get('trend', 'Unknown')}
- Volume: {market_conditions.get('volume', 'Unknown')}

Which trading mode is best?
QUICK (5m, fast scalping)
MID (15m, balanced)
TREND (1h, high accuracy)

Reply with ONE WORD only: QUICK, MID, or TREND
"""
        
        messages = [
            {"role": "system", "content": "You are a trading strategy advisor."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_chatgpt(messages)
        
        # Extract mode from response
        response_upper = response.upper()
        if 'QUICK' in response_upper:
            return 'QUICK'
        elif 'MID' in response_upper:
            return 'MID'
        elif 'TREND' in response_upper:
            return 'TREND'
        else:
            return config.MODE  # Keep current mode if unclear
    
    def explain_trap(self, trap_name: str) -> str:
        """
        Get brief explanation of a detected trap
        
        Args:
            trap_name: Name of the trap
        
        Returns:
            Brief explanation
        """
        
        prompt = f"""
Explain this trading trap in 2 sentences:
{trap_name.replace('_', ' ').title()}

Be concise and practical.
"""
        
        messages = [
            {"role": "system", "content": "You are a trading educator."},
            {"role": "user", "content": prompt}
        ]
        
        return self._call_chatgpt(messages)
    
    def rank_pairs(self, pairs_data: List[Dict]) -> List[str]:
        """
        Rank trading pairs by opportunity
        
        Args:
            pairs_data: List of dicts with pair stats
        
        Returns:
            List of pairs ranked by priority
        """
        
        # Format data for ChatGPT
        pairs_summary = "\n".join([
            f"{d['pair']}: RSI={d['rsi']}, ADX={d['adx']}, Trend={d['trend']}"
            for d in pairs_data
        ])
        
        prompt = f"""
Rank these crypto pairs by trading opportunity (best first):

{pairs_summary}

Reply with comma-separated pair names only.
Example: F-BTC_INR, F-ETH_INR, F-SOL_INR
"""
        
        messages = [
            {"role": "system", "content": "You are a crypto market analyst."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_chatgpt(messages)
        
        # Parse ranked pairs
        ranked = [p.strip() for p in response.split(',')]
        
        # Validate pairs
        valid_pairs = [p for p in ranked if p in config.PAIRS]
        
        return valid_pairs if valid_pairs else config.PAIRS
    
    def check_parameter_safety(self, entry: float, sl: float, tp: float, leverage: int) -> Dict:
        """
        Check if trading parameters are safe
        
        Args:
            entry: Entry price
            sl: Stop loss
            tp: Take profit
            leverage: Leverage multiplier
        
        Returns:
            Safety assessment
        """
        
        risk_reward = abs(tp - entry) / abs(entry - sl)
        
        prompt = f"""
Check these trade parameters:
Entry: ₹{entry}
Stop Loss: ₹{sl}
Take Profit: ₹{tp}
Leverage: {leverage}x
Risk/Reward Ratio: {risk_reward:.2f}

Is this SAFE or RISKY? Reply in one line.
"""
        
        messages = [
            {"role": "system", "content": "You are a risk management expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_chatgpt(messages)
        
        is_safe = 'SAFE' in response.upper()
        
        return {
            'safe': is_safe,
            'advice': response
        }