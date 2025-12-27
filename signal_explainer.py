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
        Generate PREMIUM chart image with entry, SL, TP marked
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
                mpf.make_addplot(df['EMA_Fast'], color='#00D9FF', width=2, alpha=0.9),
                mpf.make_addplot(df['EMA_Slow'], color='#FFB800', width=2, alpha=0.9)
            ]

            hlines = {
                'hlines': [signal['entry'], signal['sl'], signal['tp1'], signal['tp2']],
                'colors': ['#00FF41', '#FF3B30', '#7FFF00', '#32CD32'],
                'linestyle': '--',
                'linewidths': 2
            }

            mc = mpf.make_marketcolors(
                up='#26A69A', down='#EF5350',
                edge='inherit',
                wick={'up':'#26A69A', 'down':'#EF5350'},
                volume='in'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='',
                gridcolor='#1E1E1E',
                facecolor='#0D1117',
                figcolor='#0D1117',
                edgecolor='#30363D',
                rc={
                    'font.size': 10,
                    'axes.labelcolor': '#C9D1D9',
                    'axes.edgecolor': '#30363D',
                    'xtick.color': '#8B949E',
                    'ytick.color': '#8B949E',
                    'grid.alpha': 0.1
                }
            )

            filename = f"chart_{signal['pair']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join('charts', filename)

            fig, axes = mpf.plot(
                df, 
                type='candle', 
                style=s, 
                addplot=ap, 
                hlines=hlines,
                title=f"{signal['pair']} {signal['direction']} - {signal['timeframe']}",
                figsize=(14, 8),
                returnfig=True,
                volume=False,
                tight_layout=True
            )
            
            axes[0].set_facecolor('#0D1117')
            fig.savefig(filepath, facecolor='#0D1117', dpi=150)
            plt.close(fig)

            print(f"📊 Chart saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"⚠️ Chart generation failed: {e}")
            return None

    @staticmethod
    def generate_explanation(signal: Dict) -> str:
        """
        Generate BENGALI explanation with EMOJIS
        Returns: formatted string for Telegram
        """
        try:
            direction = signal['direction']
            ema_fast = signal.get('ema_fast_period', 20)
            ema_slow = signal.get('ema_slow_period', 50)
            entry = signal['entry']
            sl = signal['sl']
            tp1 = signal['tp1']
            tp2 = signal['tp2']
            timeframe = signal['timeframe']
            rsi = signal['rsi']
            adx = signal['adx']
            volume_surge = signal['volume_surge']
            
            sl_distance = abs(entry - sl) / entry * 100
            tp1_distance = abs(tp1 - entry) / entry * 100
            tp2_distance = abs(tp2 - entry) / entry * 100
            rr1 = tp1_distance / sl_distance if sl_distance > 0 else 0
            rr2 = tp2_distance / sl_distance if sl_distance > 0 else 0

            if direction == "LONG":
                explanation = f"""🟢 আমি বলছি দাম উপরে যাওয়ার চান্স আছে, তাই এটা LONG ট্রেড।

⏱️ {timeframe} চার্টে দেখি — মানে ছোট নড়াচড়া না, একটু স্থির ট্রেন্ড।

📈 EMA {ema_fast} আর EMA {ema_slow} উপরের দিকে, তাই ট্রেন্ড এখন UP।

📉 RSI শক্ত ({rsi}), কিন্তু এখনো ভাঙেনি।

📊 ADX দেখাচ্ছে ট্রেন্ড পরিষ্কার ({adx})।

💰 ₹{entry:,.2f} থেকে লং নিচ্ছো

🛑 ₹{sl:,.2f} এর নিচে ক্লোজ করলে বেরিয়ে যাবে — কারণ তখন আমি ভুল।

✅ ₹{tp1:,.2f} এ প্রথম লাভ ({rr1:.1f}R)

🚀 ₹{tp2:,.2f} এ বড় লাভ ({rr2:.1f}R)

⚖️ ঝুঁকি কম, লাভ বেশি — তাই ট্রেডটা লজিক্যাল।
"""
                if volume_surge < 1.2:
                    explanation += "\n📦 ভলিউম কম হলে ধীরে উঠতে পারে — ধৈর্য ধরো।"
                
                if signal.get('liquidity_sweep'):
                    explanation += "\n💎 লিকুইডিটি সুইপ হয়েছে — এটা ভালো সাইন।"
                
                if signal.get('near_order_block'):
                    explanation += "\n🎯 অর্ডার ব্লক কাছে — সাপোর্ট শক্ত।"

            else:  # SHORT
                explanation = f"""🔴 আমি বলছি দাম নিচে নামার চান্স আছে, তাই এটা SHORT ট্রেড।

⏱️ {timeframe} চার্টে দেখি — মানে ছোট নড়াচড়া না, পরিষ্কার ট্রেন্ড।

📉 EMA {ema_fast} আর EMA {ema_slow} নিচের দিকে, তাই ট্রেন্ড এখন DOWN।

📈 RSI উপরে ছিল ({rsi}), এখন দুর্বল হচ্ছে।

📊 ADX দেখাচ্ছে ট্রেন্ড শক্ত ({adx})।

💰 ₹{entry:,.2f} থেকে শর্ট নিচ্ছো

🛑 ₹{sl:,.2f} এর উপরে ক্লোজ করলে বেরিয়ে যাবে — কারণ তখন আমি ভুল।

✅ ₹{tp1:,.2f} এ প্রথম লাভ ({rr1:.1f}R)

🚀 ₹{tp2:,.2f} এ বড় লাভ ({rr2:.1f}R)

⚖️ ঝুঁকি কম, লাভ বেশি — তাই ট্রেডটা লজিক্যাল।
"""
                if volume_surge < 1.2:
                    explanation += "\n📦 ভলিউম কম হলে ধীরে নামতে পারে — ধৈর্য ধরো।"
                
                if signal.get('liquidity_sweep'):
                    explanation += "\n💎 লিকুইডিটি সুইপ হয়েছে — এটা ভালো সাইন।"
                
                if signal.get('near_order_block'):
                    explanation += "\n🎯 অর্ডার ব্লক কাছে — রেজিস্ট্যান্স শক্ত।"

            explanation += """

❗ নিয়ম ভাঙলে ট্রেড ফেল করবে।

🤖 আমি সিগন্যাল দিই, ডিসিপ্লিন তোমার দায়িত্ব।
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