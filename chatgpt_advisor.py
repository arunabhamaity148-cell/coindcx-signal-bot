import openai
from typing import Dict, List
from config import config

class ChatGPTAdvisor:
    """
    ChatGPT integration for decision-making assistance
    Uses MINIMAL API calls to save costs
    """
    
    def __init__(self):
        openai.api_key = config.CHATGPT_API_KEY
        self.model = config.CHATGPT_MODEL
    
    def _call_chatgpt(self, messages: List[Dict]) -> str:
        """
        Internal method to call ChatGPT API
        
        Args:
            messages: List of message dicts with role and content
        
        Returns:
            ChatGPT response text
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=300,  # Keep it short
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ ChatGPT API error: {e}")
            return "Error: Unable to get AI advice"
    
    def validate_signal(self, signal: Dict) -> Dict:
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