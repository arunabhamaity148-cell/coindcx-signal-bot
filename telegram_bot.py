"""
📱 Telegram Bot - Premium Style Alerts with Charts
"""

import requests
import matplotlib.pyplot as plt
import io
from config import *

class TelegramBot:
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
    def send_message(self, text, parse_mode="Markdown"):
        """Send text message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.json()
        except Exception as e:
            print(f"❌ Telegram send failed: {e}")
            return None
    
    def generate_chart(self, df, signal):
        """Generate trading chart with indicators"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                                gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Price + EMAs
            ax1.plot(df.index[-100:], df['close'].iloc[-100:], label='Price', linewidth=2, color='white')
            ax1.plot(df.index[-100:], df['ema20'].iloc[-100:], label='EMA 20', alpha=0.7, color='cyan')
            ax1.plot(df.index[-100:], df['ema50'].iloc[-100:], label='EMA 50', alpha=0.7, color='orange')
            ax1.plot(df.index[-100:], df['ema200'].iloc[-100:], label='EMA 200', alpha=0.7, color='red')
            
            # Mark entry point
            entry_idx = df.index[-1]
            entry_price = signal['entry_price']
            ax1.scatter([entry_idx], [entry_price], color='lime' if signal['direction']=='LONG' else 'red', 
                       s=200, marker='*', zorder=5, label='Entry')
            
            ax1.set_title(f"{signal['symbol']} - {signal['direction']} Signal", fontsize=14, color='white')
            ax1.set_ylabel('Price', color='white')
            ax1.legend(loc='upper left')
            ax1.grid(alpha=0.3)
            ax1.set_facecolor('#1e1e1e')
            
            # RSI
            ax2.plot(df.index[-100:], df['rsi'].iloc[-100:], color='purple', linewidth=2)
            ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax2.fill_between(df.index[-100:], 30, 70, alpha=0.1, color='gray')
            ax2.set_ylabel('RSI', color='white')
            ax2.set_ylim(0, 100)
            ax2.grid(alpha=0.3)
            ax2.set_facecolor('#1e1e1e')
            
            # MACD
            ax3.plot(df.index[-100:], df['macd'].iloc[-100:], label='MACD', color='blue')
            ax3.plot(df.index[-100:], df['macd_signal'].iloc[-100:], label='Signal', color='red')
            ax3.bar(df.index[-100:], df['macd_hist'].iloc[-100:], label='Histogram', color='gray', alpha=0.5)
            ax3.legend(loc='upper left')
            ax3.set_ylabel('MACD', color='white')
            ax3.grid(alpha=0.3)
            ax3.set_facecolor('#1e1e1e')
            
            fig.patch.set_facecolor('#1e1e1e')
            for ax in [ax1, ax2, ax3]:
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
            
            plt.tight_layout()
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, facecolor='#1e1e1e')
            buf.seek(0)
            plt.close()
            
            return buf
        except Exception as e:
            print(f"❌ Chart generation failed: {e}")
            return None
    
    def send_photo(self, photo_buffer, caption=""):
        """Send photo to Telegram"""
        try:
            url = f"{self.base_url}/sendPhoto"
            files = {'photo': photo_buffer}
            data = {'chat_id': self.chat_id, 'caption': caption, 'parse_mode': 'Markdown'}
            response = requests.post(url, files=files, data=data, timeout=30)
            return response.json()
        except Exception as e:
            print(f"❌ Photo send failed: {e}")
            return None
    
    def send_signal_alert(self, signal, levels, chart_df):
        """Send complete signal with chart"""
        # Format message
        direction_emoji = EMOJI_CONFIG['long'] if signal['direction'] == 'LONG' else EMOJI_CONFIG['short']
        
        message = f"""
🚨 **NEW SIGNAL ALERT** 🚨

{direction_emoji} **{signal['direction']} {signal['symbol']}** ({signal['mode'].upper()} Mode)

━━━━━━━━━━━━━━━━━━
📊 **ENTRY DETAILS**
━━━━━━━━━━━━━━━━━━
💰 Entry Price: `₹{levels['entry']:.2f}`
📦 Position Size: `{levels['position_size']:.4f}` coins
💵 Margin: `₹{levels['margin']:,.0f}`
⚡ Leverage: `{levels['leverage']}x`

━━━━━━━━━━━━━━━━━━
🎯 **TAKE PROFIT**
━━━━━━━━━━━━━━━━━━
{EMOJI_CONFIG['tp1']} TP1: `₹{levels['tp1']:.2f}` | RR: `{levels['rr1']:.2f}`
{EMOJI_CONFIG['tp2']} TP2: `₹{levels['tp2']:.2f}` | RR: `{levels['rr2']:.2f}`

━━━━━━━━━━━━━━━━━━
{EMOJI_CONFIG['sl']} **STOP LOSS**
━━━━━━━━━━━━━━━━━━
🛑 SL: `₹{levels['sl']:.2f}`

━━━━━━━━━━━━━━━━━━
⚠️ **LIQUIDATION**
━━━━━━━━━━━━━━━━━━
💀 Liq Price: `₹{levels['liq_price']:.2f}`

━━━━━━━━━━━━━━━━━━
📈 **INDICATORS**
━━━━━━━━━━━━━━━━━━
📊 RSI: `{signal['indicators']['rsi']:.1f}`
🔄 Trend: `{signal['indicators']['ema_trend']}`
⭐ Signal Score: `{signal['score']}/15`

⏰ Time: {signal['timestamp'].strftime('%d %b %Y, %H:%M:%S')}

💯 **TRADE SMART | MANAGE RISK** 💯
"""
        
        # Generate and send chart
        if SEND_CHART_WITH_SIGNAL and chart_df is not None:
            chart_buffer = self.generate_chart(chart_df, signal)
            if chart_buffer:
                self.send_photo(chart_buffer, caption=message)
                return
        
        # Fallback: send text only
        self.send_message(message)
    
    def send_performance_report(self, stats):
        """Send daily performance summary"""
        win_emoji = EMOJI_CONFIG['win']
        loss_emoji = EMOJI_CONFIG['loss']
        
        report = f"""
📊 **DAILY PERFORMANCE REPORT** 📊

━━━━━━━━━━━━━━━━━━
📈 **SIGNALS SUMMARY**
━━━━━━━━━━━━━━━━━━
📡 Total Signals: `{stats['total_signals']}`
{win_emoji} Wins: `{stats['wins']}` ({stats['win_rate']:.1f}%)
{loss_emoji} Losses: `{stats['losses']}`

━━━━━━━━━━━━━━━━━━
💰 **PROFIT & LOSS**
━━━━━━━━━━━━━━━━━━
💵 Total PnL: `₹{stats['total_pnl']:,.2f}`
📊 Avg Win: `₹{stats['avg_win']:,.2f}`
📉 Avg Loss: `₹{stats['avg_loss']:,.2f}`
🎯 Best Trade: `₹{stats['best_trade']:,.2f}`

━━━━━━━━━━━━━━━━━━
🎯 **TARGET STATUS**
━━━━━━━━━━━━━━━━━━
Daily Target: `₹{TARGET_DAILY_PROFIT:,.0f}`
Progress: `{(stats['total_pnl']/TARGET_DAILY_PROFIT)*100:.1f}%`

━━━━━━━━━━━━━━━━━━
📋 **BREAKDOWN**
━━━━━━━━━━━━━━━━━━
{EMOJI_CONFIG['tp1']} TP1 Hit: `{stats['tp1_hits']}`
{EMOJI_CONFIG['tp2']} TP2 Hit: `{stats['tp2_hits']}`
{EMOJI_CONFIG['sl']} SL Hit: `{stats['sl_hits']}`

⏰ Report Time: {stats['report_time']}

{'🎉 TARGET ACHIEVED! 🎉' if stats['total_pnl'] >= TARGET_DAILY_PROFIT else '💪 Keep Trading Smart!'}
"""
        
        self.send_message(report)
    
    def send_startup_message(self):
        """Send bot startup notification"""
        message = f"""
🤖 **CoinDCX FUTURES BOT ACTIVATED** 🤖

━━━━━━━━━━━━━━━━━━
⚙️ **CONFIGURATION**
━━━━━━━━━━━━━━━━━━
💰 Margin/Trade: `₹{MARGIN_PER_TRADE:,.0f}`
⚡ Leverage: `{LEVERAGE}x`
🎯 Daily Target: `₹{TARGET_DAILY_PROFIT:,.0f}`
📊 Signal Range: `{MIN_DAILY_SIGNALS}-{MAX_DAILY_SIGNALS}/day`

━━━━━━━━━━━━━━━━━━
🔍 **WATCHLIST**
━━━━━━━━━━━━━━━━━━
📈 Monitoring: `{len(WATCHLIST)} pairs`

━━━━━━━━━━━━━━━━━━
🛡️ **SAFETY FEATURES**
━━━━━━━━━━━━━━━━━━
✅ 45-Logic Filter System
✅ Market Health Check
✅ Liquidation Protection
✅ Anti-Manipulation Filter
✅ Risk-Reward Validation

🚀 **BOT IS NOW SCANNING...** 🚀
"""
        self.send_message(message)