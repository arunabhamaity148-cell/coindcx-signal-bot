import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import Dict
import os
from datetime import datetime
from indicators import Indicators

class SignalExplainer:
    """
    Generate chart images and educational explanations
    RUNS AFTER signal approval - does NOT affect trading logic
    """

    @staticmethod
    def generate_chart(signal: Dict, candles: pd.DataFrame) -> str:
        """
        Generate chart image with entry, SL, TP marked
        Returns: filepath of saved image or None
        """
        try:
            os.makedirs('charts', exist_ok=True)
            df = candles.tail(100).copy()
            df.index = pd.to_datetime(df.index)

            ema_fast_period = signal.get('ema_fast_period', 20)
            ema_slow_period = signal.get('ema_slow_period', 50)
            df['EMA_Fast'] = Indicators.ema(df['close'], ema_fast_period)
            df['EMA_Slow'] = Indicators.ema(df['close'], ema_slow_period)

            ap = [
                mpf.make_addplot(df['EMA_Fast'], color='blue', width=1),
                mpf.make_addplot(df['EMA_Slow'], color='orange', width=1)
            ]

            hlines = {
                'hlines': [signal['entry'], signal['sl'], signal['tp1'], signal['tp2']],
                'colors': ['green', 'red', 'lightgreen', 'darkgreen'],
                'linestyle': '--',
                'linewidths': 1.5
            }

            style = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 8})
            filename = f"chart_{signal['pair']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join('charts', filename)

            mpf.plot(df, type='candle', style=style, addplot=ap, hlines=hlines,
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
        Generate FACTUAL explanation message (NO judgement, NO predictions)
        Returns: formatted string for Telegram
        """
        try:
            ema_fast = signal.get('ema_fast_period', 20)
            ema_slow = signal.get('ema_slow_period', 50)
            
            explanation = f"""
📚 TRADE BREAKDOWN

━━━━━━━━━━━━━━━━━━━━
SIGNAL DATA

• Direction: {signal['direction']}
• Mode: {signal['mode']}
• Timeframe: {signal['timeframe']}
• HTF Alignment: Passed
• EMA Fast ({ema_fast}) and EMA Slow ({ema_slow}) aligned
• RSI: {signal['rsi']}
• ADX: {signal['adx']}
• Volume: {signal['volume_surge']}x
"""

            if signal.get('liquidity_sweep'):
                sweep_info = signal.get('sweep_info', {})
                explanation += f"• Liquidity Sweep: {sweep_info.get('type', 'Detected')}\n"

            if signal.get('near_order_block'):
                ob_info = signal.get('ob_info', {})
                explanation += f"• Order Block: {ob_info.get('distance', 'N/A')}% away\n"

            if signal.get('fvg_fill'):
                fvg_info = signal.get('fvg_info', {})
                explanation += f"• FVG: {fvg_info.get('type', 'Detected')} - {fvg_info.get('gap_size_pct', 'N/A')}%\n"

            if signal.get('near_key_level'):
                explanation += f"• Key Level: {signal.get('key_level_info', 'Nearby')}\n"

            sl_distance = abs(signal['entry'] - signal['sl']) / signal['entry'] * 100
            tp1_distance = abs(signal['tp1'] - signal['entry']) / signal['entry'] * 100
            tp2_distance = abs(signal['tp2'] - signal['entry']) / signal['entry'] * 100
            rr1 = tp1_distance / sl_distance if sl_distance > 0 else 0
            rr2 = tp2_distance / sl_distance if sl_distance > 0 else 0

            explanation += f"""
━━━━━━━━━━━━━━━━━━━━
LEVELS

Entry: ₹{signal['entry']:,.2f}
Stop Loss: ₹{signal['sl']:,.2f} ({sl_distance:.2f}%)
TP1: ₹{signal['tp1']:,.2f} ({tp1_distance:.2f}% / {rr1:.1f}R)
TP2: ₹{signal['tp2']:,.2f} ({tp2_distance:.2f}% / {rr2:.1f}R)

━━━━━━━━━━━━━━━━━━━━
INVALIDATION

Trade invalidated if price closes {"below" if signal['direction'] == "LONG" else "above"} SL: ₹{signal['sl']:,.2f}

━━━━━━━━━━━━━━━━━━━━
CONTEXT

• Score: {signal['score']}/100
• MTF Trend: {signal.get('mtf_trend', 'N/A')}

━━━━━━━━━━━━━━━━━━━━
Educational only - Not financial advice
"""
            return explanation.strip()
        except Exception as e:
            print(f"⚠️ Explanation generation failed: {e}")
            return ""

    @staticmethod
    def explain_signal(signal: Dict, candles: pd.DataFrame) -> Dict:
        """
        Main method: Generate both chart and explanation
        Returns: {'chart_path': str or None, 'explanation': str}
        """
        try:
            chart_path = SignalExplainer.generate_chart(signal, candles)
            explanation = SignalExplainer.generate_explanation(signal)
            return {'chart_path': chart_path, 'explanation': explanation}
        except Exception as e:
            print(f"⚠️ Signal explanation failed: {e}")
            return {'chart_path': None, 'explanation': ""}