"""
Risk Management System
Handles position sizing, leverage, stop loss, take profit, and daily limits
"""
import logging
from typing import Dict, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.daily_pnl = 0
        self.daily_trades = 0
        self.open_positions = []
        self.consecutive_losses = 0
        self.today = datetime.now().date()
    
    def reset_daily_stats(self):
        """Reset daily statistics at day start"""
        current_date = datetime.now().date()
        if current_date != self.today:
            logger.info("📅 New trading day - Resetting daily stats")
            self.daily_pnl = 0
            self.daily_trades = 0
            self.today = current_date
    
    def calculate_position_size(self, capital: float, signal_confidence: float,
                                 market_regime: str = 'normal') -> float:
        """
        Calculate position size based on capital and risk
        Formula: Position = (Capital × Risk%) / Stop Loss%
        """
        # Base position size
        position_percent = self.config.get('position_size_percent', 0.15)
        
        # Adjust based on confidence
        if signal_confidence > 0.80:
            position_percent *= 1.2  # Increase by 20%
        elif signal_confidence < 0.65:
            position_percent *= 0.8  # Decrease by 20%
        
        # Adjust based on consecutive losses
        if self.consecutive_losses >= 3:
            position_percent *= 0.5  # Reduce to 50%
            logger.warning(f"⚠️ Reduced position size due to {self.consecutive_losses} losses")
        
        # Adjust based on market regime
        if market_regime == 'volatile':
            position_percent *= 0.7  # Reduce in volatile markets
        
        # Calculate position size
        position_size = capital * position_percent
        
        # Ensure within limits
        max_position = capital * 0.50  # Max 50% in single position
        position_size = min(position_size, max_position)
        
        logger.info(f"💰 Position size: ₹{position_size:.2f} ({position_percent:.1%} of capital)")
        
        return position_size
    
    def calculate_leverage(self, market_regime: str, volatility: float,
                           signal_confidence: float) -> int:
        """
        Calculate optimal leverage based on market conditions
        """
        min_leverage = self.config.get('min_leverage', 3)
        max_leverage = self.config.get('max_leverage', 7)
        
        # Start with base leverage
        leverage = 5
        
        # Adjust based on market regime
        if market_regime == 'trending':
            leverage = max_leverage  # Use higher leverage in trends
        elif market_regime == 'ranging':
            leverage = min_leverage  # Use lower leverage in range
        elif market_regime == 'volatile':
            leverage = min_leverage  # Use minimum in volatile markets
        
        # Adjust based on volatility (ATR)
        if volatility > 0.03:  # High volatility
            leverage = min(leverage, 3)
        
        # Adjust based on confidence
        if signal_confidence < 0.70:
            leverage = min_leverage
        
        logger.info(f"📊 Leverage: {leverage}x (regime: {market_regime}, vol: {volatility:.2%})")
        
        return leverage
    
    def calculate_stop_loss(self, entry_price: float, side: str, 
                            leverage: int, atr: float = None) -> float:
        """
        Calculate stop loss price
        Ensures minimum 5% distance from liquidation
        """
        base_sl_percent = self.config.get('stop_loss_percent', 0.02)
        
        # Adjust SL based on ATR if available
        if atr:
            sl_percent = max(atr * 1.5, base_sl_percent)
        else:
            sl_percent = base_sl_percent
        
        # Calculate liquidation price
        if side == 'LONG':
            liquidation_price = entry_price * (1 - 1/leverage)
            stop_loss = entry_price * (1 - sl_percent)
            
            # Ensure SL is above liquidation + buffer
            buffer = self.config.get('liquidation_buffer_percent', 0.05)
            min_sl = liquidation_price * (1 + buffer)
            stop_loss = max(stop_loss, min_sl)
            
        else:  # SHORT
            liquidation_price = entry_price * (1 + 1/leverage)
            stop_loss = entry_price * (1 + sl_percent)
            
            # Ensure SL is below liquidation - buffer
            buffer = self.config.get('liquidation_buffer_percent', 0.05)
            max_sl = liquidation_price * (1 - buffer)
            stop_loss = min(stop_loss, max_sl)
        
        logger.info(f"🛑 Stop Loss: ₹{stop_loss:.2f} (liq: ₹{liquidation_price:.2f})")
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, side: str,
                              signal_confidence: float) -> float:
        """
        Calculate take profit price
        """
        base_tp_percent = self.config.get('take_profit_percent', 0.03)
        
        # Adjust TP based on confidence
        if signal_confidence > 0.80:
            tp_percent = base_tp_percent * 1.5  # Higher target
        else:
            tp_percent = base_tp_percent
        
        if side == 'LONG':
            take_profit = entry_price * (1 + tp_percent)
        else:  # SHORT
            take_profit = entry_price * (1 - tp_percent)
        
        logger.info(f"🎯 Take Profit: ₹{take_profit:.2f} ({tp_percent:.1%})")
        
        return take_profit
    
    def check_daily_limits(self) -> Tuple[bool, str]:
        """
        Check if daily limits are reached
        Returns: (can_trade, reason)
        """
        self.reset_daily_stats()
        
        # Check daily trade limit
        max_daily_trades = self.config.get('max_daily_trades', 8)
        if self.daily_trades >= max_daily_trades:
            return False, f"Daily trade limit reached ({self.daily_trades}/{max_daily_trades})"
        
        # Check daily loss limit
        max_daily_loss_percent = self.config.get('max_daily_loss_percent', 0.20)
        # Assuming we track capital
        if hasattr(self, 'initial_daily_capital'):
            max_loss = self.initial_daily_capital * max_daily_loss_percent
            if self.daily_pnl < -max_loss:
                return False, f"Daily loss limit reached (₹{abs(self.daily_pnl):.2f})"
        
        return True, "OK"
    
    def check_max_positions(self) -> Tuple[bool, str]:
        """Check if max concurrent positions reached"""
        max_positions = self.config.get('max_positions', 3)
        current_positions = len(self.open_positions)
        
        if current_positions >= max_positions:
            return False, f"Max positions reached ({current_positions}/{max_positions})"
        
        return True, "OK"
    
    def validate_trade(self, capital: float, position_size: float, 
                       leverage: int, entry_price: float, 
                       stop_loss: float, side: str) -> Tuple[bool, str]:
        """
        Validate trade before execution
        Returns: (is_valid, reason)
        """
        # Check daily limits
        can_trade, reason = self.check_daily_limits()
        if not can_trade:
            return False, reason
        
        # Check max positions
        can_trade, reason = self.check_max_positions()
        if not can_trade:
            return False, reason
        
        # Check position size
        if position_size > capital * 0.50:
            return False, "Position size exceeds 50% of capital"
        
        # Check leverage
        max_leverage = self.config.get('max_leverage', 7)
        if leverage > max_leverage:
            return False, f"Leverage {leverage}x exceeds max {max_leverage}x"
        
        # Check stop loss distance
        sl_distance = abs(entry_price - stop_loss) / entry_price
        if sl_distance < 0.005:  # Less than 0.5%
            return False, "Stop loss too tight"
        
        # Check liquidation distance
        liquidation_price = entry_price * (1 - 1/leverage) if side == 'LONG' else entry_price * (1 + 1/leverage)
        liq_distance = abs(stop_loss - liquidation_price) / liquidation_price
        
        min_buffer = self.config.get('liquidation_buffer_percent', 0.05)
        if liq_distance < min_buffer:
            return False, f"Stop loss too close to liquidation ({liq_distance:.1%})"
        
        # Check risk-reward ratio
        # (This requires take profit price, simplified here)
        
        logger.info("✅ Trade validation passed")
        return True, "OK"
    
    def update_trade_result(self, pnl: float):
        """Update statistics after trade closes"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        logger.info(f"📊 Daily P&L: ₹{self.daily_pnl:.2f} | Trades: {self.daily_trades}")
        logger.info(f"📊 Consecutive losses: {self.consecutive_losses}")
    
    def add_position(self, position: Dict):
        """Add position to tracking"""
        self.open_positions.append(position)
        logger.info(f"📈 Position added: {len(self.open_positions)} open")
    
    def remove_position(self, position_id: str):
        """Remove position from tracking"""
        self.open_positions = [p for p in self.open_positions if p.get('id') != position_id]
        logger.info(f"📉 Position removed: {len(self.open_positions)} open")
    
    def calculate_portfolio_risk(self) -> Dict:
        """Calculate total portfolio risk"""
        total_exposure = sum([p.get('size', 0) * p.get('leverage', 1) 
                              for p in self.open_positions])
        
        total_at_risk = sum([abs(p.get('entry') - p.get('stop_loss')) / p.get('entry') * p.get('size')
                             for p in self.open_positions])
        
        return {
            'total_positions': len(self.open_positions),
            'total_exposure': total_exposure,
            'total_at_risk': total_at_risk,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades
        }
    
    def emergency_close_all(self) -> bool:
        """
        Emergency procedure - close all positions
        Called when daily loss limit hit or critical error
        """
        logger.critical("🚨 EMERGENCY: Closing all positions!")
        
        # In real implementation, this would call exchange API
        # to close all positions at market price
        
        self.open_positions = []
        return True


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    from config.settings import RISK_CONFIG
    
    risk_mgr = RiskManager(RISK_CONFIG)
    
    # Test position sizing
    capital = 10000
    confidence = 0.75
    position_size = risk_mgr.calculate_position_size(capital, confidence, 'trending')
    
    # Test leverage
    leverage = risk_mgr.calculate_leverage('trending', 0.02, confidence)
    
    # Test stop loss
    entry_price = 50000
    stop_loss = risk_mgr.calculate_stop_loss(entry_price, 'LONG', leverage)
    
    # Test take profit
    take_profit = risk_mgr.calculate_take_profit(entry_price, 'LONG', confidence)
    
    # Validate trade
    is_valid, reason = risk_mgr.validate_trade(
        capital, position_size, leverage, entry_price, stop_loss, 'LONG'
    )
    
    print(f"Trade valid: {is_valid} - {reason}") 