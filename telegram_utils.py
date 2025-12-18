import requests
from datetime import datetime

class TelegramUtils:
    
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, text):
        """Send plain text message"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Telegram error: {str(e)}")
            return False
    
    def format_signal_message(self, signal, leverage):
        """Format premium signal message"""
        emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
        
        msg = f"{emoji * 3} <b>{signal['direction']} SIGNAL</b> {emoji * 3}\n\n"
        msg += f"💹 <b>Market:</b> {signal['market']}\n"
        msg += f"⚡ <b>Leverage:</b> {leverage}x\n"
        msg += f"📊 <b>Score:</b> {signal['score']}/100 {signal['quality_emoji']}\n\n"
        
        msg += f"💰 <b>Entry:</b> ₹{signal['entry']:.2f}\n"
        msg += f"🛑 <b>Stop Loss:</b> ₹{signal['sl']:.2f}\n"
        msg += f"🎯 <b>TP1:</b> ₹{signal['tp1']:.2f}\n"
        msg += f"🎯 <b>TP2:</b> ₹{signal['tp2']:.2f}\n\n"
        
        msg += f"📈 <b>R:R Ratio:</b> 1:{signal['rr_ratio']:.2f}\n\n"
        
        msg += "📊 <b>Analysis:</b>\n"
        msg += f"• RSI: {signal['analysis']['rsi']:.1f}\n"
        msg += f"• ADX: {signal['analysis']['adx']:.1f}\n"
        msg += f"• MTF: {signal['mtf']['trend_15m']} / {signal['mtf']['bias_1h']}\n"
        msg += f"• Regime: {signal['analysis']['market_regime'].upper()}\n\n"
        
        msg += "<b>✅ Reasons:</b>\n"
        for i, reason in enumerate(signal['reasons'][:5], 1):
            msg += f"{i}. {reason}\n"
        
        msg += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}\n"
        msg += "⚠️ <b>Use proper risk management!</b>"
        
        return msg
    
    def send_signal(self, signal, leverage):
        """Send formatted signal"""
        message = self.format_signal_message(signal, leverage)
        return self.send_message(message)
    
    def send_startup_message(self, config):
        """Send bot startup notification"""
        msg = "🚀 <b>BOT DEPLOYED SUCCESSFULLY!</b>\n\n"
        msg += f"✅ <b>Status:</b> ACTIVE\n"
        msg += f"📊 <b>Markets:</b> {len(config.MARKETS)} pairs (INR Futures)\n"
        msg += f"⚡ <b>Leverage:</b> {config.LEVERAGE}x\n"
        msg += f"⏱️ <b>Signal TF:</b> {config.SIGNAL_TIMEFRAME}\n"
        msg += f"🔄 <b>Scan Every:</b> {config.CHECK_INTERVAL_MINUTES} min\n"
        msg += f"🎯 <b>Min Score:</b> {config.MIN_SIGNAL_SCORE}\n"
        msg += f"⏳ <b>Cooldown:</b> {config.COOLDOWN_MINUTES} min\n\n"
        msg += f"🕐 <b>Started:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        msg += "Bot is now scanning markets...\n"
        msg += "Signals will arrive when conditions are met!"
        
        return self.send_message(msg)
    
    def send_heartbeat(self, signals_today):
        """Send periodic heartbeat"""
        msg = f"💚 <b>Bot Heartbeat</b>\n"
        msg += f"🕐 {datetime.now().strftime('%H:%M:%S')}\n"
        msg += f"📊 <b>Signals Today:</b> {signals_today}\n"
        msg += f"✅ <b>Status:</b> Running"
        return self.send_message(msg)
    
    def send_btc_block_message(self, reason):
        """Send BTC stability block notification"""
        msg = f"⚠️ <b>BTC INSTABILITY DETECTED</b>\n\n"
        msg += f"Reason: {reason}\n"
        msg += f"All signals blocked until BTC stabilizes.\n"
        msg += f"Time: {datetime.now().strftime('%H:%M:%S')}"
        return self.send_message(msg)