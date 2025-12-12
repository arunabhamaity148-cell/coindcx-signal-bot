"""
MAIN TRADING BOT - FINAL VERSION
Complete integration with all components:
- 45 unique logics
- ML ensemble (LSTM + XGBoost + RF)
- Multi-exchange support
- Risk management
- Telegram notifications
- Database logging
- Real-time monitoring
"""
import sys
import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
import argparse
import pandas as pd
import signal

# Import all components
from config.settings import (
    EXCHANGES, TRADING_CONFIG, RISK_CONFIG, ML_CONFIG, 
    UNIQUE_LOGICS, TELEGRAM_CONFIG, DATA_CONFIG
)
from data.data_collector import DataCollector
from strategy.unique_logics import LogicEvaluator
from ml.lstm_model import LSTMModel
from ml.xgboost_model import XGBoostModel
from ml.ensemble import EnsembleModel
from risk.risk_manager import RiskManager
from execution.order_executor import OrderExecutor

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, mode='paper', capital=10000):
        """
        Initialize Trading Bot - FINAL VERSION
        
        Args:
            mode: 'paper' or 'live'
            capital: Starting capital in INR
        """
        self.mode = mode
        self.capital = capital
        self.initial_capital = capital
        
        logger.info("="*100)
        logger.info("🤖 ADVANCED CRYPTO TRADING BOT - FINAL VERSION")
        logger.info("="*100)
        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"Capital: ₹{capital:,.2f}")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*100)
        
        # Initialize components
        self._initialize_components()
        
        # Initialize Telegram (async)
        self.telegram_notifier = None
        self._initialize_telegram()
        
        # Bot state
        self.is_running = False
        self.is_paused = False
        self.trades_today = []
        self.open_positions = []
        self.last_signal_time = None
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'daily_pnl': 0,
            'max_drawdown': 0,
            'equity_curve': []
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_components(self):
        """Initialize all bot components"""
        try:
            logger.info("\n📊 INITIALIZING COMPONENTS...")
            logger.info("="*100)
            
            # 1. Data Collector
            logger.info("1/7 📊 Data Collector...")
            self.data_collector = DataCollector(EXCHANGES)
            logger.info("    ✅ Multi-exchange data collector ready")
            
            # 2. Logic Evaluator (45 unique logics)
            logger.info("2/7 🧠 Logic Evaluator (45 unique logics)...")
            self.logic_evaluator = LogicEvaluator(UNIQUE_LOGICS)
            logger.info("    ✅ 45 unique trading logics loaded")
            
            # 3. ML Models
            logger.info("3/7 🤖 ML Models...")
            self.lstm_model = LSTMModel(
                lookback=ML_CONFIG['lookback_candles'],
                features=ML_CONFIG['features_count'],
                model_path=ML_CONFIG['model_path']
            )
            self.xgboost_model = XGBoostModel(
                model_path=ML_CONFIG['model_path']
            )
            
            # Load trained models
            try:
                self.lstm_model.load('lstm_model.h5')
                self.xgboost_model.load('xgboost_model.pkl')
                logger.info("    ✅ Pre-trained ML models loaded")
            except Exception as e:
                logger.warning(f"    ⚠️  ML models not found: {e}")
                logger.warning("    ℹ️  Run: python scripts/train_models.py")
            
            # 4. Ensemble Model
            logger.info("4/7 🎯 Ensemble Model...")
            self.ensemble = EnsembleModel(
                lstm_model=self.lstm_model,
                xgboost_model=self.xgboost_model,
                lstm_weight=ML_CONFIG['lstm_weight'],
                xgb_weight=ML_CONFIG['xgboost_weight'],
                rf_weight=ML_CONFIG['rf_weight'],
                model_path=ML_CONFIG['model_path']
            )
            
            try:
                self.ensemble.load_rf('random_forest.pkl')
                logger.info("    ✅ Ensemble model ready (LSTM + XGBoost + RF)")
            except:
                logger.warning("    ⚠️  Random Forest model not found")
            
            # 5. Risk Manager
            logger.info("5/7 🛡️ Risk Manager...")
            self.risk_manager = RiskManager(RISK_CONFIG)
            self.risk_manager.initial_daily_capital = self.capital
            logger.info("    ✅ Risk management system active")
            
            # 6. Order Executor
            logger.info("6/7 ⚡ Order Executor...")
            if self.mode == 'live':
                self.executor = OrderExecutor(EXCHANGES)
                logger.info("    ✅ Live order executor initialized")
            else:
                logger.info("    ✅ Paper trading mode (simulation)")
                self.executor = None
            
            # 7. Database Connection (optional)
            logger.info("7/7 🗄️  Database...")
            try:
                import psycopg2
                self.db_conn = psycopg2.connect(
                    host=DATA_CONFIG['db_host'],
                    port=DATA_CONFIG['db_port'],
                    database=DATA_CONFIG['db_name'],
                    user=DATA_CONFIG['db_user'],
                    password=DATA_CONFIG['db_password']
                )
import os, psycopg2
db_url = os.getenv("DATABASE_URL")          # Railway auto-provides
self.db_conn = psycopg2.connect(db_url) if db_url else None

                logger.info("    ✅ Database connected")
            except Exception as e:
                logger.warning(f"    ⚠️  Database not available: {e}")
                self.db_conn = None
            
            logger.info("="*100)
            logger.info("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
            logger.info("="*100)
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    def _initialize_telegram(self):
        """Initialize Telegram bot"""
        try:
            if TELEGRAM_CONFIG['enable_notifications']:
                from monitoring.telegram_bot import TradingBotNotifier
                
                bot_token = TELEGRAM_CONFIG['bot_token']
                chat_id = TELEGRAM_CONFIG['chat_id']
                
                if bot_token and chat_id and bot_token != 'YOUR_BOT_TOKEN':
                    self.telegram_notifier = TradingBotNotifier(bot_token, chat_id)
                    logger.info("✅ Telegram notifications enabled")
                else:
                    logger.warning("⚠️  Telegram not configured")
        except Exception as e:
            logger.warning(f"⚠️  Telegram initialization failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("\n⏹️  Shutdown signal received...")
        self.stop()
        sys.exit(0)
    
    async def _send_telegram(self, message):
        """Send Telegram notification"""
        if self.telegram_notifier:
            try:
                await self.telegram_notifier.send_message(message)
            except Exception as e:
                logger.warning(f"Telegram notification failed: {e}")
    
    def fetch_market_data(self, symbol='BTC/USDT:USDT', timeframe='15m', limit=200):
        """Fetch latest market data with all required info"""
        try:
            # OHLCV data
            df = self.data_collector.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                exchange_name='bybit',
                limit=limit
            )
            
            if df.empty:
                logger.error("❌ No OHLCV data received")
                return None
            
            # Orderbook
            orderbook = self.data_collector.fetch_orderbook(symbol, 'bybit', limit=20)
            
            # Funding rate
            funding_rate = self.data_collector.fetch_funding_rate(symbol, 'bybit')
            
            # Open interest
            open_interest = self.data_collector.fetch_open_interest(symbol, 'bybit')
            
            # Large orders
            large_orders = self.data_collector.detect_large_orders(
                symbol, 
                threshold=UNIQUE_LOGICS['large_order_threshold'],
                exchange_name='bybit'
            )
            
            # Cross-exchange divergence
            divergence = self.data_collector.calculate_price_divergence(symbol)
            
            return {
                'df': df,
                'orderbook': orderbook,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'large_orders': large_orders,
                'divergence': divergence,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to fetch market data: {e}")
            return None
    
    def generate_features(self, df):
        """Generate features for ML prediction"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Basic price features
            features['close'] = df['close']
            features['high'] = df['high']
            features['low'] = df['low']
            features['volume'] = df['volume']
            features['returns'] = df['close'].pct_change()
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Volatility
            features['volatility'] = df['close'].rolling(20).std()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            
            # Fill NaN
            features = features.fillna(method='bfill').fillna(0)
            
            return features
        
        except Exception as e:
            logger.error(f"❌ Feature generation failed: {e}")
            return pd.DataFrame()
    
    def analyze_trade_opportunity(self, symbol='BTC/USDT:USDT'):
        """
        MAIN ANALYSIS FUNCTION
        Combines ML prediction + 45 unique logics
        """
        logger.info("\n" + "="*100)
        logger.info(f"🔍 ANALYZING TRADE OPPORTUNITY: {symbol}")
        logger.info("="*100)
        
        try:
            # 1. Fetch market data
            market_data = self.fetch_market_data(symbol)
            if not market_data:
                return None
            
            df = market_data['df']
            orderbook = market_data['orderbook']
            funding_rate = market_data['funding_rate']
            
            logger.info(f"📊 Market Data: {len(df)} candles, Price: ₹{df['close'].iloc[-1]:,.2f}")
            
            # 2. Generate features for ML
            features = self.generate_features(df)
            if features.empty:
                logger.error("❌ Feature generation failed")
                return None
            
            # 3. ML Prediction
            logger.info("🤖 ML Prediction...")
            try:
                signal, confidence = self.ensemble.generate_signal(
                    features.tail(100).values,
                    confidence_threshold=ML_CONFIG['confidence_threshold']
                )
                
                signal_name = ['BUY', 'HOLD', 'SELL'][signal]
                logger.info(f"   Signal: {signal_name} | Confidence: {confidence:.2%}")
                
                if signal == 1:  # HOLD
                    logger.info("⏸️  ML says HOLD - No trade")
                    return None
                
            except Exception as e:
                logger.error(f"❌ ML prediction failed: {e}")
                logger.info("   Falling back to technical analysis only")
                return None
            
            # 4. Apply 45 Unique Logics
            logger.info("🧠 Evaluating 45 unique trading logics...")
            
            logic_result = self.logic_evaluator.evaluate_all_logics(
                df=df,
                orderbook=orderbook,
                funding_rate=funding_rate,
                oi_history=[market_data['open_interest']] * 20,
                recent_trades=self.trades_today,
                fear_greed_index=50,  # Would fetch from API in production
                news_times=[],  # Would fetch from news API
                liquidation_clusters=[]  # Would fetch from exchange
            )
            
            logger.info(f"   Logic Score: {logic_result['final_score']:.1f}%")
            logger.info(f"   Trade Allowed: {logic_result['trade_allowed']}")
            
            # 5. Final Decision
            if not logic_result['trade_allowed']:
                logger.warning("❌ Trade blocked by logic filters:")
                for reason in logic_result['reasons']:
                    logger.warning(f"   • {reason}")
                return None
            
            # 6. Determine trade parameters
            current_price = df['close'].iloc[-1]
            side = 'LONG' if signal == 0 else 'SHORT'
            
            market_regime = logic_result['market_health']['market_regime']
            atr = logic_result['price_action']['atr']
            
            # Position sizing
            position_size = self.risk_manager.calculate_position_size(
                capital=self.capital,
                signal_confidence=confidence,
                market_regime=market_regime
            )
            
            # Leverage calculation
            leverage = self.risk_manager.calculate_leverage(
                market_regime=market_regime,
                volatility=atr / current_price if current_price > 0 else 0.02,
                signal_confidence=confidence
            )
            
            # Stop loss
            stop_loss = self.risk_manager.calculate_stop_loss(
                entry_price=current_price,
                side=side,
                leverage=leverage,
                atr=atr
            )
            
            # Take profit
            take_profit = self.risk_manager.calculate_take_profit(
                entry_price=current_price,
                side=side,
                signal_confidence=confidence
            )
            
            # 7. Validate trade
            is_valid, reason = self.risk_manager.validate_trade(
                capital=self.capital,
                position_size=position_size,
                leverage=leverage,
                entry_price=current_price,
                stop_loss=stop_loss,
                side=side
            )
            
            if not is_valid:
                logger.warning(f"❌ Trade validation failed: {reason}")
                return None
            
            # 8. Create trade plan
            trade_plan = {
                'symbol': symbol,
                'side': side,
                'signal': signal_name,
                'confidence': confidence,
                'entry_price': current_price,
                'position_size': position_size,
                'leverage': leverage,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'logic_score': logic_result['final_score'],
                'market_regime': market_regime,
                'timestamp': datetime.now(),
                'large_orders_detected': len(market_data['large_orders']),
                'funding_rate': funding_rate
            }
            
            # 9. Print summary
            logger.info("\n" + "="*100)
            logger.info("✅ TRADE OPPORTUNITY FOUND!")
            logger.info("="*100)
            logger.info(f"📊 Signal: {signal_name} (ML Confidence: {confidence:.2%})")
            logger.info(f"🎯 Logic Score: {logic_result['final_score']:.1f}%")
            logger.info(f"💰 Entry: ₹{current_price:,.2f}")
            logger.info(f"📈 Size: ₹{position_size:,.2f} ({leverage}x leverage)")
            logger.info(f"🛑 Stop Loss: ₹{stop_loss:,.2f}")
            logger.info(f"🎯 Take Profit: ₹{take_profit:,.2f}")
            logger.info(f"📊 Market: {market_regime}")
            logger.info(f"🐋 Large Orders: {len(market_data['large_orders'])}")
            logger.info("="*100)
            
            return trade_plan
        
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return None
    
    def execute_trade(self, trade_plan):
        """Execute the trade (paper or live)"""
        try:
            if self.mode == 'paper':
                logger.info("📝 PAPER TRADE - Simulating execution...")
                
                trade = {
                    **trade_plan,
                    'id': f"paper_{int(time.time())}",
                    'status': 'open',
                    'pnl': 0,
                    'opened_at': datetime.now()
                }
                
                self.trades_today.append(trade)
                self.open_positions.append(trade)
                
                # Send Telegram notification
                if self.telegram_notifier:
                    asyncio.run(self.telegram_notifier.notify_trade_opened(trade))
                
                return trade
            
            elif self.mode == 'live':
                logger.info("⚡ LIVE TRADE - Executing on exchange...")
                
                position = self.executor.open_position(
                    symbol=trade_plan['symbol'],
                    side=trade_plan['side'],
                    size=trade_plan['position_size'],
                    leverage=trade_plan['leverage'],
                    stop_loss=trade_plan['stop_loss'],
                    take_profit=trade_plan['take_profit'],
                    exchange_name='bybit'
                )
                
                if 'error' not in position:
                    self.trades_today.append(position)
                    self.open_positions.append(position)
                    self.risk_manager.add_position(position)
                    
                    # Send Telegram notification
                    if self.telegram_notifier:
                        asyncio.run(self.telegram_notifier.notify_trade_opened(position))
                    
                    # Log to database
                    if self.db_conn:
                        self._log_trade_to_db(position)
                else:
                    logger.error(f"❌ Trade execution failed: {position['error']}")
                
                return position
        
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return {'error': str(e)}
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        if not self.open_positions:
            return
        
        logger.info(f"👀 Monitoring {len(self.open_positions)} open position(s)...")
        
        for position in self.open_positions[:]:
            try:
                # Get current price
                symbol = position['symbol']
                market_data = self.fetch_market_data(symbol, limit=50)
                
                if not market_data:
                    continue
                
                current_price = market_data['df']['close'].iloc[-1]
                
                # Calculate unrealized P&L
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                unrealized_pnl = pnl_pct * position['position_size'] * position['leverage']
                
                position['current_price'] = current_price
                position['unrealized_pnl'] = unrealized_pnl
                
                logger.info(f"   {symbol} {position['side']}: "
                          f"₹{current_price:,.2f} | P&L: ₹{unrealized_pnl:+,.2f}")
                
                # Check time limit (4 hours)
                time_open = (datetime.now() - position['opened_at']).total_seconds() / 3600
                
                if time_open >= 4:
                    logger.warning(f"⏰ Position open for {time_open:.1f}h - Closing...")
                    self._close_position(position, 'time_limit')
                
            except Exception as e:
                logger.error(f"❌ Position monitoring error: {e}")
    
    def _close_position(self, position, reason='manual'):
        """Close a position"""
        logger.info(f"📉 Closing position: {position['symbol']} ({reason})")
        
        # Simulate close for paper trading
        if self.mode == 'paper':
            position['status'] = 'closed'
            position['exit_price'] = position.get('current_price', position['entry_price'])
            position['pnl'] = position.get('unrealized_pnl', 0)
            position['closed_at'] = datetime.now()
            position['close_reason'] = reason
            
            # Update performance
            self.performance['total_pnl'] += position['pnl']
            self.performance['daily_pnl'] += position['pnl']
            self.risk_manager.update_trade_result(position['pnl'])
            
            # Remove from open positions
            self.open_positions = [p for p in self.open_positions if p['id'] != position['id']]
            
            # Send notification
            if self.telegram_notifier:
                asyncio.run(self.telegram_notifier.notify_trade_closed(position))
            
            logger.info(f"   P&L: ₹{position['pnl']:+,.2f}")
    
    def _log_trade_to_db(self, trade):
        """Log trade to database"""
        if not self.db_conn:
            return
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    trade_id, symbol, side, entry_time, entry_price,
                    size, leverage, stop_loss, take_profit,
                    ml_confidence, logic_score, status, exchange
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                trade['id'], trade['symbol'], trade['side'], trade['opened_at'],
                trade['entry_price'], trade['position_size'], trade['leverage'],
                trade['stop_loss'], trade['take_profit'], trade['confidence'],
                trade['logic_score'], 'open', 'bybit'
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.warning(f"Database logging failed: {e}")
    
    def _update_performance(self):
        """Update performance metrics"""
        if self.trades_today:
            closed = [t for t in self.trades_today if t.get('status') == 'closed']
            
            if closed:
                wins = [t for t in closed if t.get('pnl', 0) > 0]
                self.performance['wins'] = len(wins)
                self.performance['losses'] = len(closed) - len(wins)
                self.performance['win_rate'] = len(wins) / len(closed) if closed else 0
                self.performance['total_trades'] = len(closed)
    
    def _print_daily_summary(self):
        """Print daily performance summary"""
        logger.info("\n" + "="*100)
        logger.info("📊 DAILY SUMMARY")
        logger.info("="*100)
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        logger.info(f"Total Trades: {self.performance['total_trades']}")
        logger.info(f"Wins: {self.performance['wins']} | Losses: {self.performance['losses']}")
        logger.info(f"Win Rate: {self.performance['win_rate']:.1%}")
        logger.info(f"Daily P&L: ₹{self.performance['daily_pnl']:+,.2f}")
        logger.info(f"Total P&L: ₹{self.performance['total_pnl']:+,.2f}")
        logger.info(f"Current Capital: ₹{self.capital:,.2f}")
        logger.info(f"ROI: {(self.capital - self.initial_capital) / self.initial_capital:.2%}")
        logger.info("="*100)
    
    def run(self, symbol='BTC/USDT:USDT', interval=900):
        """
        MAIN BOT LOOP
        
        Args:
            symbol: Trading pair
            interval: Check interval in seconds (default 900 = 15 min)
        """
        self.is_running = True
        
        logger.info("\n" + "="*100)
        logger.info("🚀 TRADING BOT STARTED - FINAL VERSION")
        logger.info("="*100)
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Interval: {interval}s ({interval/60:.0f} minutes)")
        logger.info(f"Capital: ₹{self.capital:,.2f}")
        logger.info(f"45 Unique Logics: ACTIVE")
        logger.info(f"ML Ensemble: ACTIVE")
        logger.info(f"Risk Management: ACTIVE")
        logger.info("="*100)
        
        # Send start notification
        if self.telegram_notifier:
            asyncio.run(self.telegram_notifier.notify_bot_started(self.capital, self.mode))
        
        try:
            iteration = 0
            while self.is_running:
                iteration += 1
                
                try:
                    logger.info(f"\n{'='*100}")
                    logger.info(f"🔄 ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"{'='*100}")
                    
                    # Skip if paused
                    if self.is_paused:
                        logger.info("⏸️  Bot paused - Waiting...")
                        time.sleep(interval)
                        continue
                    
                    # 1. Check daily limits
                    can_trade, reason = self.risk_manager.check_daily_limits()
                    if not can_trade:
                        logger.warning(f"⏸️  Trading paused: {reason}")
                        
                        if self.telegram_notifier:
                            asyncio.run(self.telegram_notifier.notify_daily_loss_limit(
                                self.risk_manager.daily_pnl,
                                self.initial_capital * RISK_CONFIG['max_daily_loss_percent']
                            ))
                        
                        time.sleep(interval)
                        continue
                    
                    # 2. Monitor existing positions
                    if self.open_positions:
                        self.monitor_positions()
                    
                    # 3. Check for new trade opportunity (if no open positions)
                    max_positions = TRADING_CONFIG['max_positions']
                    if len(self.open_positions) < max_positions:
                        trade_plan = self.analyze_trade_opportunity(symbol)
                        
                        # 4. Execute if opportunity found
                        if trade_plan:
                            trade = self.execute_trade(trade_plan)
                            
                            if 'error' not in trade:
                                logger.info(f"✅ Trade executed: {trade['id']}")
                                self.last_signal_time = datetime.now()
                            else:
                                logger.error(f"❌ Trade failed: {trade.get('error')}")
                        else:
                            logger.info("⏭️  No trade opportunity - Waiting for next cycle...")
                    else:
                        logger.info(f"⏸️  Max positions ({max_positions}) reached - No new trades")
                    
                    # 5. Update performance
                    self._update_performance()
                    
                    # 6. Print summary every 10 iterations
                    if iteration % 10 == 0:
                        self._print_daily_summary()
                    
                    # 7. Sleep until next check
                    logger.info(f"\n😴 Sleeping for {interval}s ({interval/60:.0f} minutes)...")
                    logger.info(f"Next check: {(datetime.now() + timedelta(seconds=interval)).strftime('%H:%M:%S')}")
                    time.sleep(interval)
                
                except KeyboardInterrupt:
                    logger.info("\n⏹️  Keyboard interrupt received")
                    break
                
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
                    
                    if self.telegram_notifier:
                        asyncio.run(self.telegram_notifier.notify_error(str(e)))
                    
                    logger.info("⏳ Waiting 60s before retry...")
                    time.sleep(60)
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the bot gracefully"""
        logger.info("\n" + "="*100)
        logger.info("⏹️  STOPPING BOT...")
        logger.info("="*100)
        
        self.is_running = False
        
        # Close any open positions (optional - you might want to keep them)
        if self.open_positions:
            logger.warning(f"⚠️  {len(self.open_positions)} position(s) still open!")
            
            response = input("Close all positions before stopping? (y/n): ")
            if response.lower() == 'y':
                for position in self.open_positions[:]:
                    self._close_position(position, 'bot_shutdown')
        
        # Print final performance
        self._print_daily_summary()
        
        # Send stop notification
        if self.telegram_notifier:
            asyncio.run(self.telegram_notifier.notify_bot_stopped())
        
        # Close database connection
        if self.db_conn:
            self.db_conn.close()
            logger.info("🗄️  Database connection closed")
        
        logger.info("\n" + "="*100)
        logger.info("👋 BOT STOPPED SUCCESSFULLY")
        logger.info("="*100)
        logger.info(f"Total Runtime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Final Capital: ₹{self.capital:,.2f}")
        logger.info(f"Total P&L: ₹{self.performance['total_pnl']:+,.2f}")
        logger.info(f"Total Trades: {self.performance['total_trades']}")
        logger.info(f"Win Rate: {self.performance['win_rate']:.1%}")
        logger.info("="*100)
        logger.info("\nThank you for using Advanced Crypto Trading Bot!")
        logger.info("Trade safely and responsibly! 🚀")
        logger.info("="*100)


# ==================== MAIN ====================
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Advanced Crypto Trading Bot - Final Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading (recommended first)
  python main.py --mode paper --capital 10000
  
  # Live trading with custom settings
  python main.py --mode live --capital 2000 --symbol BTC/USDT:USDT --interval 900
  
  # Quick test with short interval
  python main.py --mode paper --capital 5000 --interval 300
  
Features:
  ✅ 45 Unique Trading Logics
  ✅ ML Ensemble (LSTM + XGBoost + RF)
  ✅ Multi-Exchange Support
  ✅ Advanced Risk Management
  ✅ Telegram Notifications
  ✅ Database Logging
  ✅ Real-time Monitoring
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='paper',
        choices=['paper', 'live'],
        help='Trading mode (paper=simulation, live=real trading)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Starting capital in INR (default: 10000)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT:USDT',
        help='Trading symbol (default: BTC/USDT:USDT)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=900,
        help='Check interval in seconds (default: 900 = 15 minutes)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run configuration validation before starting'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("🐛 Debug logging enabled")
    
    # Run validation if requested
    if args.validate:
        logger.info("🔍 Running configuration validation...")
        try:
            from scripts.validate_config import ConfigValidator
            validator = ConfigValidator()
            success = validator.run_all_checks()
            
            if not success:
                logger.error("❌ Configuration validation failed!")
                logger.error("Please fix the errors before starting the bot")
                sys.exit(1)
            
            logger.info("✅ Configuration validation passed!")
        except Exception as e:
            logger.warning(f"⚠️  Validation check failed: {e}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Warning for live mode
    if args.mode == 'live':
        logger.warning("\n" + "="*100)
        logger.warning("⚠️  WARNING: LIVE TRADING MODE")
        logger.warning("="*100)
        logger.warning("You are about to start LIVE TRADING with REAL MONEY!")
        logger.warning(f"Capital at risk: ₹{args.capital:,.2f}")
        logger.warning("")
        logger.warning("Please confirm:")
        logger.warning("1. You have tested thoroughly in paper trading mode")
        logger.warning("2. You understand the risks involved")
        logger.warning("3. You can afford to lose this capital")
        logger.warning("4. You have proper API keys configured")
        logger.warning("="*100)
        
        confirmation = input("\nType 'I UNDERSTAND THE RISKS' to continue: ")
        
        if confirmation != 'I UNDERSTAND THE RISKS':
            logger.info("❌ Live trading cancelled")
            sys.exit(0)
        
        # Double confirmation
        final_confirm = input(f"\nFinal confirmation: Start live trading with ₹{args.capital:,.2f}? (yes/no): ")
        
        if final_confirm.lower() != 'yes':
            logger.info("❌ Live trading cancelled")
            sys.exit(0)
    
    # Print banner
    print("\n" + "="*100)
    print("     █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗ ██████╗███████╗██████╗ ")
    print("    ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗")
    print("    ███████║██║  ██║██║   ██║███████║██╔██╗ ██║██║     █████╗  ██║  ██║")
    print("    ██╔══██║██║  ██║╚██╗ ██╔╝██╔══██║██║╚██╗██║██║     ██╔══╝  ██║  ██║")
    print("    ██║  ██║██████╔╝ ╚████╔╝ ██║  ██║██║ ╚████║╚██████╗███████╗██████╔╝")
    print("    ╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═════╝ ")
    print("")
    print("                   CRYPTO TRADING BOT - FINAL VERSION")
    print("                        45 Unique Logics | ML Ensemble")
    print("="*100)
    print("")
    
    # Create and run bot
    try:
        bot = TradingBot(mode=args.mode, capital=args.capital)
        bot.run(symbol=args.symbol, interval=args.interval)
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user")
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# ==================== USAGE EXAMPLES ====================
"""
═══════════════════════════════════════════════════════════════════════════════
                              USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

1. PAPER TRADING (Recommended first!)
   python main.py --mode paper --capital 10000
   
   - Simulates real trading
   - No real money risk
   - Tests all systems
   - Builds confidence

2. PAPER TRADING WITH VALIDATION
   python main.py --mode paper --capital 10000 --validate
   
   - Runs pre-flight checks
   - Validates configuration
   - Checks API keys
   - Verifies ML models

3. CUSTOM INTERVAL
   python main.py --mode paper --capital 5000 --interval 300
   
   - Check every 5 minutes (300 seconds)
   - Good for testing
   - More active trading

4. DIFFERENT SYMBOL
   python main.py --mode paper --symbol ETH/USDT:USDT --capital 10000
   
   - Trade Ethereum instead of Bitcoin
   - Same logic applies
   - Works with any futures pair

5. LIVE TRADING (After paper success)
   python main.py --mode live --capital 2000
   
   ⚠️ WARNING: Real money!
   - Start with small capital
   - Use 3x leverage max initially
   - Monitor closely
   - Scale gradually

6. LIVE TRADING WITH ALL OPTIONS
   python main.py --mode live --capital 5000 --symbol BTC/USDT:USDT --interval 900 --validate
   
   - Full configuration
   - Pre-flight checks
   - 15-minute intervals
   - Real trading

7. DEBUG MODE
   python main.py --mode paper --capital 10000 --debug
   
   - Detailed logging
   - Useful for troubleshooting
   - Shows all calculations

═══════════════════════════════════════════════════════════════════════════════
                           MONITORING COMMANDS
═══════════════════════════════════════════════════════════════════════════════

While bot is running:
- Ctrl+C to stop gracefully
- Check logs in: logs/trading_bot_YYYYMMDD.log
- Monitor Telegram bot for real-time updates

Telegram commands:
/status  - Check current status
/balance - Show P&L
/trades  - Today's trades
/pause   - Pause trading
/resume  - Resume trading
/emergency - Close all positions

═══════════════════════════════════════════════════════════════════════════════
                              IMPORTANT NOTES
═══════════════════════════════════════════════════════════════════════════════

✅ BEFORE LIVE TRADING:
   1. Run paper trading for at least 1-2 weeks
   2. Verify win rate is 60%+
   3. Check all systems work properly
   4. Start with minimal capital (₹2,000)
   5. Use conservative settings (3x leverage)
   6. Monitor closely for first week

✅ BEST PRACTICES:
   - Never risk more than you can afford to lose
   - Set daily loss limits and respect them
   - Take profits regularly (50% withdrawal rule)
   - Keep detailed records
   - Update software regularly
   - Monitor bot daily
   - Have emergency procedures ready

✅ RISK WARNINGS:
   - Cryptocurrency trading is highly risky
   - Past performance ≠ future results
   - Bot can fail due to technical issues
   - Markets can be unpredictable
   - Always use proper risk management
   - Never trade with borrowed money

✅ SUPPORT:
   - Documentation: README.md
   - Quick start: QUICKSTART.md
   - Validation: python scripts/validate_config.py
   - Issues: GitHub Issues
   - Logs: logs/trading_bot_*.log

═══════════════════════════════════════════════════════════════════════════════
                           GOOD LUCK & TRADE SAFELY! 🚀
═══════════════════════════════════════════════════════════════════════════════
"""
