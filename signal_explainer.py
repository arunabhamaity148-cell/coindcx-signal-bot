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
                mpf.make_addplot(df['EMA_Fast'], color='#4A90E2', width=1.8, alpha=0.9),
                mpf.make_addplot(df['EMA_Slow'], color='#F5A623', width=1.8, alpha=0.9)
            ]

            hlines = {
                'hlines': [signal['sl'], signal['tp1'], signal['tp2']],
                'colors': ['#E74C3C', '#85C1E9', '#27AE60'],
                'linestyle': '-',
                'linewidths': 2,
                'alpha': 0.95
            }

            mc = mpf.make_marketcolors(
                up='#00C853', down='#FF5252',
                edge='inherit',
                wick={'up':'#00C853', 'down':'#FF5252'},
                volume='in'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='',
                gridcolor='#2A2A2A',
                facecolor='#0E1117',
                figcolor='#0E1117',
                edgecolor='#1E1E1E',
                rc={
                    'font.size': 9,
                    'font.family': 'monospace',
                    'axes.labelcolor': '#D0D0D0',
                    'axes.edgecolor': '#2A2A2A',
                    'xtick.color': '#808080',
                    'ytick.color': '#808080',
                    'grid.alpha': 0.08,
                    'axes.linewidth': 0.5
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
                title=dict(title=f"{signal['pair']} | {signal['direction']} | {signal['timeframe']}", color='#E0E0E0', fontsize=12, weight='bold'),
                figsize=(15, 8),
                returnfig=True,
                volume=False,
                tight_layout=True,
                scale_padding={'left': 0.3, 'top': 0.3, 'right': 1.2, 'bottom': 0.3}
            )
            
            axes[0].set_facecolor('#0E1117')
            axes[0].tick_params(labelsize=8, colors='#808080')
            
            entry_price = signal['entry']
            last_index = df.index[-1]
            axes[0].plot(last_index, entry_price, marker='o', markersize=12, color='#00E676', markeredgecolor='#00C853', markeredgewidth=2.5, zorder=10)
            axes[0].text(last_index, entry_price, '  ENTRY', fontsize=9, color='#00E676', verticalalignment='center', weight='bold', alpha=0.95)
            
            axes[0].text(0.98, 0.02, 'CryptoBot Pro', transform=axes[0].transAxes, fontsize=8, color='#404040', alpha=0.4, ha='right', va='bottom', style='italic')
            
            fig.savefig(filepath, facecolor='#0E1117', dpi=150, bbox_inches='tight', pad_inches=0.1)
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
            rr1 = tp1_distance / sl_distance if sl_distance > 0 else 0

            if direction == "LONG":
                explanation = f"""🟢 আমি বলছি দাম উপরে যাওয়ার দিক আছে, তাই এটা LONG ট্রেড।

⏱️ {timeframe} চার্টে দেখি — মানে ছোট নড়াচড়া না।

📈 EMA {ema_fast} আর EMA {ema_slow} উপরের দিকে।

📉 RSI শক্ত ({rsi}), কিন্তু এখনো ভাঙেনি।

📊 ADX দেখাচ্ছে ট্রেন্ড পরিষ্কার ({adx})।

💰 ₹{entry:,.2f} এখানে ঢুকছো

🛑 ₹{sl:,.2f} এখানে ভুল প্রমাণ হলে বেরোবে

✅ ₹{tp1:,.2f} প্রথম লাভ ({rr1:.1f}R)

🚀 ₹{tp2:,.2f} বড় লাভ

⚖️ ঝুঁকি কম, লাভ বেশি
"""
                if volume_surge < 1.2:
                    explanation += "\n📦 ভলিউম কম — ধৈর্য ধরো।"
                
                if signal.get('liquidity_sweep'):
                    explanation += "\n💎 লিকুইডিটি সুইপ হয়েছে।"

            else:
                explanation = f"""🔴 আমি বলছি দাম নিচে নামার দিক আছে, তাই এটা SHORT ট্রেড।

⏱️ {timeframe} চার্টে দেখি — পরিষ্কার ট্রেন্ড।

📉 EMA {ema_fast} আর EMA {ema_slow} নিচের দিকে।

📈 RSI উপরে ছিল ({rsi}), এখন দুর্বল।

📊 ADX দেখাচ্ছে ট্রেন্ড শক্ত ({adx})।

💰 ₹{entry:,.2f} এখানে শর্ট নিচ্ছো

🛑 ₹{sl:,.2f} এখানে ভুল প্রমাণ হলে বেরোবে

✅ ₹{tp1:,.2f} প্রথম লাভ ({rr1:.1f}R)

🚀 ₹{tp2:,.2f} বড় লাভ

⚖️ ঝুঁকি কম, লাভ বেশি
"""
                if volume_surge < 1.2:
                    explanation += "\n📦 ভলিউম কম — ধৈর্য ধরো।"
                
                if signal.get('liquidity_sweep'):
                    explanation += "\n💎 লিকুইডিটি সুইপ হয়েছে।"

            explanation += "\n\n🤖 আমি সিগন্যাল দিই, নিয়ম মানা তোমার কাজ।"
            
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