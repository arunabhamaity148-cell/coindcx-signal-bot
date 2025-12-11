"""
Order Execution Engine
Handles order placement, position monitoring, TP/SL management
"""
import ccxt
import logging
from typing import Dict, Optional
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(self, exchanges: Dict):
        self.exchanges = {}
        self._initialize_exchanges(exchanges)
        self.open_orders = []
        self.open_positions = []
    
    def _initialize_exchanges(self, exchange_configs: Dict):
        """Initialize exchange connections"""
        for name, config in exchange_configs.items():
            try:
                if name == 'bybit':
                    exchange = ccxt.bybit({
                        'apiKey': config['api_key'],
                        'secret': config['secret'],
                        'enableRateLimit': True,
                        'options': {'defaultType': 'future'}
                    })
                elif name == 'okx':
                    exchange = ccxt.okx({
                        'apiKey': config['api_key'],
                        'secret': config['secret'],
                        'password': config['password'],
                        'enableRateLimit': True,
                        'options': {'defaultType': 'swap'}
                    })
                
                self.exchanges[name] = exchange
                logger.info(f"✅ {name.upper()} executor initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {name}: {e}")
    
    def place_order(self, symbol: str, side: str, order_type: str,
                    amount: float, price: Optional[float] = None,
                    exchange_name: str = 'bybit',
                    params: Dict = None) -> Dict:
        """
        Place order on exchange
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            amount: Order size in base currency
            price: Limit price (for limit orders)
            exchange_name: Which exchange to use
            params: Additional parameters (leverage, reduceOnly, etc.)
        
        Returns:
            Order response from exchange
        """
        try:
            exchange = self.exchanges[exchange_name]
            
            # Default params
            if params is None:
                params = {}
            
            # Place order
            if order_type == 'market':
                order = exchange.create_market_order(symbol, side, amount, params)
            elif order_type == 'limit':
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = exchange.create_limit_order(symbol, side, amount, price, params)
            else:
                raise ValueError(f"Unknown order type: {order_type}")
            
            logger.info(f"✅ Order placed: {side.upper()} {amount} {symbol} @ {price or 'MARKET'}")
            logger.info(f"   Order ID: {order['id']}")
            
            # Track order
            self.open_orders.append(order)
            
            return order
        
        except Exception as e:
            logger.error(f"❌ Order placement failed: {e}")
            return {'error': str(e)}
    
    def open_position(self, symbol: str, side: str, size: float,
                      leverage: int, stop_loss: float, take_profit: float,
                      exchange_name: str = 'bybit') -> Dict:
        """
        Open a new position with SL and TP
        
        Args:
            symbol: Trading pair
            side: 'LONG' or 'SHORT'
            size: Position size in USDT
            leverage: Leverage to use
            stop_loss: Stop loss price
            take_profit: Take profit price
            exchange_name: Which exchange
        
        Returns:
            Position info
        """
        try:
            exchange = self.exchanges[exchange_name]
            
            # Set leverage
            self._set_leverage(symbol, leverage, exchange_name)
            
            # Calculate amount based on size and price
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            amount = size / current_price
            
            # Place main order
            order_side = 'buy' if side == 'LONG' else 'sell'
            main_order = self.place_order(
                symbol=symbol,
                side=order_side,
                order_type='market',
                amount=amount,
                exchange_name=exchange_name
            )
            
            if 'error' in main_order:
                return main_order
            
            # Wait for order to fill
            time.sleep(2)
            
            # Get actual entry price
            filled_order = exchange.fetch_order(main_order['id'], symbol)
            entry_price = filled_order.get('average', current_price)
            
            # Place stop loss order
            sl_side = 'sell' if side == 'LONG' else 'buy'
            sl_order = self._place_stop_order(
                symbol=symbol,
                side=sl_side,
                amount=amount,
                trigger_price=stop_loss,
                exchange_name=exchange_name
            )
            
            # Place take profit order
            tp_order = self._place_take_profit_order(
                symbol=symbol,
                side=sl_side,
                amount=amount,
                price=take_profit,
                exchange_name=exchange_name
            )
            
            # Create position object
            position = {
                'id': main_order['id'],
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'size': size,
                'amount': amount,
                'leverage': leverage,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sl_order_id': sl_order.get('id'),
                'tp_order_id': tp_order.get('id'),
                'exchange': exchange_name,
                'opened_at': datetime.now(),
                'status': 'open'
            }
            
            self.open_positions.append(position)
            
            logger.info("="*60)
            logger.info("📈 POSITION OPENED")
            logger.info("="*60)
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Side: {side}")
            logger.info(f"Entry: ₹{entry_price:.2f}")
            logger.info(f"Size: ₹{size:.2f} ({amount:.4f} contracts)")
            logger.info(f"Leverage: {leverage}x")
            logger.info(f"Stop Loss: ₹{stop_loss:.2f}")
            logger.info(f"Take Profit: ₹{take_profit:.2f}")
            logger.info("="*60)
            
            return position
        
        except Exception as e:
            logger.error(f"❌ Failed to open position: {e}")
            return {'error': str(e)}
    
    def close_position(self, position_id: str, reason: str = 'manual') -> Dict:
        """
        Close an open position
        
        Args:
            position_id: Position ID
            reason: Reason for closing (manual, tp, sl, timeout)
        """
        try:
            # Find position
            position = None
            for pos in self.open_positions:
                if pos['id'] == position_id:
                    position = pos
                    break
            
            if not position:
                return {'error': 'Position not found'}
            
            exchange = self.exchanges[position['exchange']]
            
            # Close position (opposite side)
            close_side = 'sell' if position['side'] == 'LONG' else 'buy'
            close_order = self.place_order(
                symbol=position['symbol'],
                side=close_side,
                order_type='market',
                amount=position['amount'],
                exchange_name=position['exchange'],
                params={'reduceOnly': True}
            )
            
            # Cancel TP/SL orders
            if position.get('sl_order_id'):
                try:
                    exchange.cancel_order(position['sl_order_id'], position['symbol'])
                except:
                    pass
            
            if position.get('tp_order_id'):
                try:
                    exchange.cancel_order(position['tp_order_id'], position['symbol'])
                except:
                    pass
            
            # Get exit price
            time.sleep(1)
            filled = exchange.fetch_order(close_order['id'], position['symbol'])
            exit_price = filled.get('average', 0)
            
            # Calculate P&L
            if position['side'] == 'LONG':
                pnl = (exit_price - position['entry_price']) / position['entry_price'] * position['size']
            else:  # SHORT
                pnl = (position['entry_price'] - exit_price) / position['entry_price'] * position['size']
            
            pnl = pnl * position['leverage']  # Apply leverage
            
            # Update position
            position['status'] = 'closed'
            position['exit_price'] = exit_price
            position['pnl'] = pnl
            position['closed_at'] = datetime.now()
            position['close_reason'] = reason
            
            # Remove from open positions
            self.open_positions = [p for p in self.open_positions if p['id'] != position_id]
            
            logger.info("="*60)
            logger.info(f"📉 POSITION CLOSED - {reason.upper()}")
            logger.info("="*60)
            logger.info(f"Symbol: {position['symbol']}")
            logger.info(f"Side: {position['side']}")
            logger.info(f"Entry: ₹{position['entry_price']:.2f}")
            logger.info(f"Exit: ₹{exit_price:.2f}")
            logger.info(f"P&L: ₹{pnl:.2f} ({pnl/position['size']*100:.2f}%)")
            logger.info("="*60)
            
            return position
        
        except Exception as e:
            logger.error(f"❌ Failed to close position: {e}")
            return {'error': str(e)}
    
    def monitor_positions(self) -> list:
        """
        Monitor all open positions
        Check if TP/SL hit, time limit exceeded, etc.
        """
        updates = []
        
        for position in self.open_positions[:]:  # Copy list to iterate safely
            try:
                exchange = self.exchanges[position['exchange']]
                
                # Get current price
                ticker = exchange.fetch_ticker(position['symbol'])
                current_price = ticker['last']
                
                # Calculate unrealized P&L
                if position['side'] == 'LONG':
                    unrealized_pnl = (current_price - position['entry_price']) / position['entry_price'] * position['size']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) / position['entry_price'] * position['size']
                
                unrealized_pnl *= position['leverage']
                
                position['current_price'] = current_price
                position['unrealized_pnl'] = unrealized_pnl
                
                # Check time limit (4 hours default)
                time_open = (datetime.now() - position['opened_at']).total_seconds() / 3600
                if time_open > 4:  # 4 hours
                    logger.warning(f"⏰ Position {position['symbol']} open for {time_open:.1f}h - Closing")
                    result = self.close_position(position['id'], 'timeout')
                    updates.append(result)
                
                # Check if TP/SL orders are filled
                # (Exchange should handle this automatically, but we can monitor)
                
            except Exception as e:
                logger.error(f"❌ Error monitoring position: {e}")
        
        return updates
    
    def _set_leverage(self, symbol: str, leverage: int, exchange_name: str):
        """Set leverage for symbol"""
        try:
            exchange = self.exchanges[exchange_name]
            exchange.set_leverage(leverage, symbol)
            logger.info(f"📊 Leverage set: {leverage}x for {symbol}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to set leverage: {e}")
    
    def _place_stop_order(self, symbol: str, side: str, amount: float,
                          trigger_price: float, exchange_name: str) -> Dict:
        """Place stop loss order"""
        try:
            exchange = self.exchanges[exchange_name]
            
            params = {
                'stopPrice': trigger_price,
                'reduceOnly': True
            }
            
            order = exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side=side,
                amount=amount,
                params=params
            )
            
            logger.info(f"🛑 Stop loss placed @ ₹{trigger_price:.2f}")
            return order
        
        except Exception as e:
            logger.error(f"❌ Failed to place stop loss: {e}")
            return {'error': str(e)}
    
    def _place_take_profit_order(self, symbol: str, side: str, amount: float,
                                  price: float, exchange_name: str) -> Dict:
        """Place take profit order"""
        try:
            exchange = self.exchanges[exchange_name]
            
            params = {
                'reduceOnly': True
            }
            
            order = exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            
            logger.info(f"🎯 Take profit placed @ ₹{price:.2f}")
            return order
        
        except Exception as e:
            logger.error(f"❌ Failed to place take profit: {e}")
            return {'error': str(e)}
    
    def get_balance(self, exchange_name: str = 'bybit') -> Dict:
        """Get account balance"""
        try:
            exchange = self.exchanges[exchange_name]
            balance = exchange.fetch_balance()
            
            usdt_balance = balance.get('USDT', {})
            
            return {
                'free': usdt_balance.get('free', 0),
                'used': usdt_balance.get('used', 0),
                'total': usdt_balance.get('total', 0)
            }
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
            return {'error': str(e)}
    
    def emergency_close_all(self) -> bool:
        """Emergency close all positions"""
        logger.critical("🚨 EMERGENCY: Closing all positions!")
        
        success = True
        for position in self.open_positions[:]:
            result = self.close_position(position['id'], 'emergency')
            if 'error' in result:
                success = False
        
        return success


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    from config.settings import EXCHANGES
    
    # Initialize executor
    executor = OrderExecutor(EXCHANGES)
    
    # Check balance
    balance = executor.get_balance('bybit')
    print(f"Balance: {balance}")
    
    # Test position opening (PAPER TRADING - use testnet!)
    # position = executor.open_position(
    #     symbol='BTC/USDT:USDT',
    #     side='LONG',
    #     size=1000,
    #     leverage=5,
    #     stop_loss=49000,
    #     take_profit=51000,
    #     exchange_name='bybit'
    # )
    # print(f"Position: {position}") 