"""
🛡️ Risk Manager - TP/SL/Liquidation Calculator
Logic 39: Risk-Reward Validation
FIXED & STABLE VERSION ✅
"""

from config import *

class RiskManager:
    def __init__(self):
        self.active_positions = {}

    # ==================================================
    # Position size
    # ==================================================
    def calculate_position_size(self, entry_price, leverage=LEVERAGE):
        position_value = MARGIN_PER_TRADE * leverage
        quantity = position_value / entry_price
        return quantity

    # ==================================================
    # Liquidation price (safe approximation)
    # ==================================================
    def calculate_liquidation_price(self, entry_price, direction, leverage=LEVERAGE):
        fee_buffer = 0.001  # 0.1% safety fee buffer

        if direction == "LONG":
            liq_price = entry_price * (1 - (1 / leverage) - fee_buffer)
        else:  # SHORT
            liq_price = entry_price * (1 + (1 / leverage) + fee_buffer)

        return liq_price

    # ==================================================
    # TP / SL calculation with RR & liquidation safety
    # ==================================================
    def calculate_tp_sl(self, signal):
        entry = signal['entry_price']
        direction = signal['direction']
        mode = signal['mode']

        mode_params = SIGNAL_MODES[mode]

        # ---------- LONG ----------
        if direction == "LONG":
            tp1_price = entry * (1 + mode_params['tp1'] / 100)
            tp2_price = entry * (1 + mode_params['tp2'] / 100)
            sl_price  = entry * (1 - mode_params['sl'] / 100)

            liq_price = self.calculate_liquidation_price(entry, direction, LEVERAGE)
            liq_distance = ((entry - liq_price) / entry) * 100

            safe_distance = liq_distance * (1 - LIQUIDATION_BUFFER)
            max_safe_sl = entry * (1 - safe_distance / 100)

            if sl_price < max_safe_sl:
                sl_price = max_safe_sl

        # ---------- SHORT ----------
        else:
            tp1_price = entry * (1 - mode_params['tp1'] / 100)
            tp2_price = entry * (1 - mode_params['tp2'] / 100)
            sl_price  = entry * (1 + mode_params['sl'] / 100)

            liq_price = self.calculate_liquidation_price(entry, direction, LEVERAGE)
            liq_distance = ((liq_price - entry) / entry) * 100

            safe_distance = liq_distance * (1 - LIQUIDATION_BUFFER)
            max_safe_sl = entry * (1 + safe_distance / 100)

            if sl_price > max_safe_sl:
                sl_price = max_safe_sl

        # ==================================================
        # Risk Reward validation (Logic 39)
        # ==================================================
        risk = abs(entry - sl_price)
        reward_tp1 = abs(tp1_price - entry)
        reward_tp2 = abs(tp2_price - entry)

        if risk <= 0:
            return None

        rr1 = reward_tp1 / risk
        rr2 = reward_tp2 / risk

        # Allow trade if TP2 is strong even if TP1 is borderline
        if rr1 < MIN_RR_RATIO and rr2 < (MIN_RR_RATIO + 0.5):
            print(f"⚠️ RR too low | RR1={rr1:.2f}, RR2={rr2:.2f}")
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

    # ==================================================
    # PnL calculation
    # ==================================================
    def calculate_pnl(self, entry, exit_price, direction, position_size):
        if direction == "LONG":
            pnl = (exit_price - entry) * position_size
        else:
            pnl = (entry - exit_price) * position_size

        pnl_percent = (pnl / MARGIN_PER_TRADE) * 100
        return pnl, pnl_percent

    # ==================================================
    # Telegram trade summary
    # ==================================================
    def format_trade_summary(self, signal, levels):
        direction_emoji = EMOJI_CONFIG['long'] if signal['direction'] == 'LONG' else EMOJI_CONFIG['short']

        summary = f"""
{direction_emoji} **{signal['direction']} {signal['symbol']}**
📌 Mode: {signal['mode'].upper()}

💰 **Entry**
₹{levels['entry']:.2f}

🎯 **Targets**
{EMOJI_CONFIG['tp1']} TP1: ₹{levels['tp1']:.2f} | RR {levels['rr1']:.2f}
{EMOJI_CONFIG['tp2']} TP2: ₹{levels['tp2']:.2f} | RR {levels['rr2']:.2f}

{EMOJI_CONFIG['sl']} **Stop Loss**
₹{levels['sl']:.2f}

⚠️ **Liquidation**
₹{levels['liq_price']:.2f}

⚙️ **Trade Info**
Leverage: {levels['leverage']}x  
Margin: ₹{levels['margin']:,.0f}

📊 **Indicators**
RSI: {signal['indicators']['rsi']:.1f}
Trend: {signal['indicators']['ema_trend']}
Score: {signal['score']}/15

⏰ {signal['timestamp'].strftime('%H:%M:%S')}
"""
        return summary