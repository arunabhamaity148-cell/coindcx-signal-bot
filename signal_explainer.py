import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import Dict
import os
from datetime import datetime

class SignalExplainer:
    """
    Generate chart images and educational explanations
    RUNS AFTER signal approval - does NOT affect trading logic
    """

    @staticmethod
    def generate_chart(signal: Dict, candles: pd.DataFrame) -> str:
        """
        Generate chart image with entry, SL, TP marked
        Returns: filepath of saved image
        """
        try:
            # Prepare data
            df = candles.tail(100).copy()
            df.index = pd.to_datetime(df.index)
            
            # Add EMAs
            df['EMA_Fast'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA_Slow'] = df['close'].ewm(span=50, adjust=False).mean()

            # Create plot
            ap = [
                mpf.make_addplot(df['EMA_Fast'], color='blue', width=1),
                mpf.make_addplot(df['EMA_Slow'], color='orange', width=1)
            ]

            # Horizontal lines for entry, SL, TP
            hlines = {
                'hlines': [signal['entry'], signal['sl'], signal['tp1'], signal['tp2']],
                'colors': ['green', 'red', 'lightgreen', 'darkgreen'],
                'linestyle': '--',
                'linewidths': 1.5
            }

            # Style
            style = mpf.make_mpf_style(base_mpf_style='charles', 
                                         rc={'font.size': 8})

            # Save chart
            filename = f"chart_{signal['pair']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join('charts', filename)
            
            os.makedirs('charts', exist_ok=True)

            mpf.plot(df, type='candle', style=style, 
                     addplot=ap, hlines=hlines,
                     title=f"{signal['pair']} {signal['direction']} - {signal['timeframe']}",
                     savefig=filepath, figsize=(12, 6))

            print(f"📊 Chart saved: {filepath}")
            return filepath

        except Exception as e:
            print(f"⚠️ Chart generation failed: {e}")
            return None

    @staticmethod
    def generate_explanation(signal: Dict) -> str:
        """
        Generate educational explanation message
        Returns: formatted string for Telegram
        """
        try:
            # Build explanation
            explanation = f"""
📚 **TRADE BREAKDOWN - EDUCATIONAL**

━━━━━━━━━━━━━━━━━━━━
**WHY THIS TRADE?**

• **Trend Direction**: EMA{signal['mode']} + MACD aligned {signal['direction']}
• **HTF Check**: Passed (mode: {signal['mode']})
• **RSI**: {signal['rsi']} - Valid for {signal['direction']}
• **ADX**: {signal['adx']} - {"Strong" if signal['adx'] > 30 else "Moderate"} trend
• **Volume**: {signal['volume_surge']}x {"surge" if signal['volume_surge'] > 1.5 else "normal"}
"""

            # Add professional context
            if signal.get('liquidity_sweep'):
                sweep_info = signal.get('sweep_info', {})
                explanation += f"• **Liquidity Sweep**: {sweep_info.get('type', 'Detected')} - Stop hunt reversal\n"

            if signal.get('near_order_block'):
                ob_info = signal.get('ob_info', {})
                explanation += f"• **Order Block**: Near institutional zone ({ob_info.get('distance', 'N/A')}% away)\n"

            if signal.get('fvg_fill'):
                fvg_info = signal.get('fvg_info', {})
                explanation += f"• **FVG Fill**: {fvg_info.get('type', 'Detected')} - Gap: {fvg_info.get('gap_size_pct', 'N/A')}%\n"

            if signal.get('near_key_level'):
                explanation += f"• **Key Level**: {signal.get('key_level_info', 'Near support/resistance')}\n"

            # Entry logic
            explanation += f"""
━━━━━━━━━━━━━━━━━━━━
**🎯 ENTRY LOGIC**

Entry at ₹{signal['entry']:,.2f} because:
• Price aligned with {signal['direction']} trend on {signal['timeframe']}
• Confluence of technical indicators
• HTF timeframes support this direction
"""

            # SL logic
            sl_distance = abs(signal['entry'] - signal['sl']) / signal['entry'] * 100
            explanation += f"""
━━━━━━━━━━━━━━━━━━━━
**🛑 STOP LOSS LOGIC**

SL at ₹{signal['sl']:,.2f} ({sl_distance:.2f}% away) because:
• Based on ATR (volatility)
• Below/above key structure
• Invalidates the setup if hit
• Safe from liquidation
"""

            # TP logic
            tp1_distance = abs(signal['tp1'] - signal['entry']) / signal['entry'] * 100
            tp2_distance = abs(signal['tp2'] - signal['entry']) / signal['entry'] * 100
            rr1 = tp1_distance / sl_distance if sl_distance > 0 else 0
            rr2 = tp2_distance / sl_distance if sl_distance > 0 else 0

            explanation += f"""
━━━━━━━━━━━━━━━━━━━━
**✅ TAKE PROFIT LOGIC**

TP1 at ₹{signal['tp1']:,.2f} ({tp1_distance:.2f}% / {rr1:.1f}R)
• Conservative target - high probability
• Based on ATR and recent structure

TP2 at ₹{signal['tp2']:,.2f} ({tp2_distance:.2f}% / {rr2:.1f}R)
• Extended target - trend continuation
• Risk:Reward optimized
"""

            # Risk warnings
            explanation += f"""
━━━━━━━━━━━━━━━━━━━━
**⚠️ POSSIBLE RISKS**

• Market regime: {signal.get('market_regime', 'NORMAL')}
"""

            if signal.get('market_regime') == 'VOLATILE':
                explanation += "• High volatility - wider swings possible\n"
            elif signal.get('market_regime') == 'RANGING':
                explanation += "• Ranging market - may hit TP1 and reverse\n"

            explanation += f"""• Trade invalidated if price closes {"below" if signal['direction'] == "LONG" else "above"} SL
• News events can cause sudden reversals
• Always respect your risk management

━━━━━━━━━━━━━━━━━━━━
**💡 LEARNING POINTS**

• Score: {signal['score']}/100
• This is a {signal['mode']} mode setup
• MTF Trend: {signal.get('mtf_trend', 'N/A')}
• Multiple timeframes confirm direction

━━━━━━━━━━━━━━━━━━━━
_Generated for educational purposes_
_Not financial advice - DYOR_
"""

            return explanation.strip()

        except Exception as e:
            print(f"⚠️ Explanation generation failed: {e}")
            return "Educational breakdown unavailable"

    @staticmethod
    def explain_signal(signal: Dict, candles: pd.DataFrame) -> Dict:
        """
        Main method: Generate both chart and explanation
        Returns: {'chart_path': str, 'explanation': str}
        """
        try:
            chart_path = SignalExplainer.generate_chart(signal, candles)
            explanation = SignalExplainer.generate_explanation(signal)

            return {
                'chart_path': chart_path,
                'explanation': explanation
            }

        except Exception as e:
            print(f"⚠️ Signal explanation failed: {e}")
            return {
                'chart_path': None,
                'explanation': "Explanation unavailable"
            }