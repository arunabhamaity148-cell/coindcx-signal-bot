"""
🛡️ Risk Manager - TP/SL/Liquidation Calculator
Logic 39: Risk-Reward Validation
"""

from config import *

class RiskManager:
    def __init__(self):
        self.active_positions = {}
        
    def calculate_position_size(self, entry_price, leverage=LEVERAGE):
        """Calculate position size based on margin"""
        position_value = MARGIN_PER_TRADE * leverage
        quantity = position_value / entry_price
        return quantity
    
    def calculate_liquidation_price(self, entry_price, direction, leverage=LEVERAGE):
        """Calculate liquidation price for CoinDCX"""
        if direction == "LONG":
            # Liq = Entry * (1 - 1/leverage - fee)
            liq_price = entry_price * (1 - (1 / leverage) - 0.001)
        else:  # SHORT
            # Liq = Entry * (1 + 1/leverage + fee)
            liq_price = entry_price * (1 + (1 / leverage) + 0.001)
        
        return liq_price
    
    def calculate_tp_sl(self, signal):
        """Calculate TP1, TP2, SL with liquidation buffer"""
        entry = signal['entry_price']
        direction = signal['direction']
        mode = signal['mode']
        
        mode_params = SIGNAL_MODES[mode]
        
        if direction == "LONG":
            # TP levels above entry
            tp1_price = entry * (1 + mode_params['tp1'] / 100)
            tp2_price = entry * (1 + mode_params['tp2'] / 100)
            sl_price = entry * (1 - mode_params['sl'] / 100)
            
            # Check liquidation distance
            liq_price = self.calculate_liquidation_price(entry, direction, LEVERAGE)
            liq_distance = ((entry - liq_price) / entry) * 100
            
            # Ensure SL is far from liquidation (Logic 39)
            if (entry - sl_price) / entry > liq_distance * (1 - LIQUIDATION_BUFFER):
                sl_price = entry * (1 - (liq_distance * (1 - LIQUIDATION_BUFFER)) / 100)
        
        else:  # SHORT
            tp1_price = entry * (1 - mode_params['tp1'] / 100)
            tp2_price = entry * (1 - mode_params['tp2'] / 100)
            sl_price = entry * (1 + mode_params['sl'] / 100)
            
            liq_price = self.calculate_liquidation_price(entry, direction, LEVERAGE)
            liq_distance = ((liq_price - entry) / entry) * 100
            
            if (sl_price - entry) / entry > liq_distance * (1 - LIQUIDATION_BUFFER):
                sl_price = entry * (1 + (liq_distance * (1 - LIQUIDATION_BUFFER)) / 100)
        
        # Validate Risk-Reward Ratio (Logic 39)
        risk = abs(entry - sl_price)
        reward_tp1 = abs(tp1_price - entry)
        reward_tp2 = abs(tp2_price - entry)
        
        rr1 = reward_tp1 / risk if risk > 0 else 0
        rr2 = reward_tp2 / risk if risk > 0 else 0
        
        if rr1 < MIN_RR_RATIO:
            print(f"⚠️ RR ratio too low: {rr1:.2f} < {MIN_RR_RATIO}")
            return None
        
        return {
            'entry': entry,
            'tp1': tp1_price,
            'tp2': tp2_price,
            'sl': sl_price,
            'liq_price': liq_price,
            'rr1': rr1,
            'rr2': rr2,
            'position_size': self.calculate_position_size(entry, LEVERAGE),
            'margin': MARGIN_PER_TRADE,
            'leverage': LEVERAGE
        }
    
    def calculate_pnl(self, entry, exit_price, direction, position_size):
        """Calculate PnL for a trade"""
        if direction == "LONG":
            pnl = (exit_price - entry) * position_size
        else:
            pnl = (entry - exit_price) * position_size
        
        pnl_percent = (pnl / MARGIN_PER_TRADE) * 100
        return pnl, pnl_percent
    
    def format_trade_summary(self, signal, levels):
        """Format complete trade info for Telegram"""
        direction_emoji = EMOJI_CONFIG['long'] if signal['direction'] == 'LONG' else EMOJI_CONFIG['short']
        
        summary = f"""
{direction_emoji} **{signal['direction']} {signal['symbol']}** - {signal['mode'].upper()} Mode

📊 **Entry Details**
Entry: ₹{levels['entry']:.2f}
Position: {levels['position_size']:.4f} coins
Margin: ₹{levels['margin']:,.0f}
Leverage: {levels['leverage']}x

🎯 **Take Profit Levels**
{EMOJI_CONFIG['tp1']} TP1: ₹{levels['tp1']:.2f} (RR: {levels['rr1']:.2f})
{EMOJI_CONFIG['tp2']} TP2: ₹{levels['tp2']:.2f} (RR: {levels['rr2']:.2f})

{EMOJI_CONFIG['sl']} **Stop Loss**
SL: ₹{levels['sl']:.2f}

⚠️ **Liquidation**
Liq Price: ₹{levels['liq_price']:.2f}

📈 **Indicators**
RSI: {signal['indicators']['rsi']:.1f}
Trend: {signal['indicators']['ema_trend']}
Score: {signal['score']}/15

⏰ Signal Time: {signal['timestamp'].strftime('%H:%M:%S')}
"""
        return summary