"""
MAIN TRADING BOT – FINAL (Full-length, Railway-deploy)
All 45 logics, ML ensemble, multi-exchange, risk, Telegram, DB intact
"""
import os, sys, time, asyncio, logging, signal, argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ----------  imports  ----------
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
from monitoring.telegram_bot import TradingBotNotifier   # lightweight helper

# ----------  logging  ----------
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trading_bot_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Final bot – Railway ready (no polling, full features)."""
    def __init__(self, mode='paper', capital=10000):
        self.mode, self.capital, self.initial_capital = mode, capital, capital
        self.is_running = False
        self.open_positions: list = []
        self.trades_today:  list = []
        self.performance = {
            'total_trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0,
            'win_rate': 0.0, 'daily_pnl': 0, 'max_drawdown': 0,
            'equity_curve': []
        }

        # Banner
        logger.info("="*100)
        logger.info("🤖 ADVANCED CRYPTO TRADING BOT – FINAL")
        logger.info("="*100)
        logger.info("Mode : %s  |  Capital : ₹%,.2f  |  Date : %s",
                    mode.upper(), capital, datetime.now())
        logger.info("="*100)

        self._init_components()
        self._init_telegram()
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ------------------------------------------------------------------
    # Component init
    # ------------------------------------------------------------------
    def _init_components(self):
        logger.info("\n📊 INITIALISING COMPONENTS...")
        logger.info("="*100)

        # 1. Data
        logger.info("1/7 📊 Data Collector...")
        self.data_collector = DataCollector(EXCHANGES)
        logger.info("    ✅ Multi-exchange ready")

        # 2. Logics
        logger.info("2/7 🧠 Logic Evaluator (45 unique logics)...")
        self.logic_evaluator = LogicEvaluator(UNIQUE_LOGICS)
        logger.info("    ✅ 45 unique logics loaded")

        # 3. ML
        logger.info("3/7 🤖 ML Models...")
        self.lstm_model   = LSTMModel(ML_CONFIG['lookback_candles'],
                                      ML_CONFIG['features_count'],
                                      ML_CONFIG['model_path'])
        self.xgb_model    = XGBoostModel(ML_CONFIG['model_path'])
        try:
            self.lstm_model.load('lstm_model.h5')
            self.xgb_model.load('xgboost_model.pkl')
            logger.info("    ✅ Pre-trained ML loaded")
        except Exception as e:
            logger.warning("    ⚠️  ML models not found: %s", e)

        # 4. Ensemble
        logger.info("4/7 🎯 Ensemble Model...")
        self.ensemble = EnsembleModel(self.lstm_model, self.xgb_model,
                                     ML_CONFIG['lstm_weight'],
                                     ML_CONFIG['xgboost_weight'],
                                     ML_CONFIG['rf_weight'],
                                     ML_CONFIG['model_path'])
        try:
            self.ensemble.load_rf('random_forest.pkl')
            logger.info("    ✅ Ensemble ready")
        except:
            logger.warning("    ⚠️  RF model missing")

        # 5. Risk
        logger.info("5/7 🛡️ Risk Manager...")
        self.risk_manager = RiskManager(RISK_CONFIG)
        self.risk_manager.initial_daily_capital = self.capital
        logger.info("    ✅ Risk system active")

        # 6. Executor
        logger.info("6/7 ⚡ Order Executor...")
        if self.mode == 'live':
            self.executor = OrderExecutor(EXCHANGES)
            logger.info("    ✅ Live executor")
        else:
            self.executor = None
            logger.info("    ✅ Paper mode")

        # 7. DB (Railway PostgreSQL)
        logger.info("7/7 🗄️  Database...")
        try:
            import psycopg2
            db_url = os.getenv("DATABASE_URL")      # Railway gives this
            self.db_conn = psycopg2.connect(db_url) if db_url else None
            logger.info("    ✅ DB connected" if self.db_conn else "    ⚠️  DB URL not set")
        except Exception as e:
            logger.warning("    ⚠️  DB error: %s", e)
            self.db_conn = None

        logger.info("="*100)
        logger.info("✅ ALL COMPONENTS INITIALISED")
        logger.info("="*100)

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------
    def _init_telegram(self):
        self.telegram_notifier = None
        if not TELEGRAM_CONFIG.get('enable_notifications', True):
            return

        token = os.getenv("TELEGRAM_BOT_TOKEN") or TELEGRAM_CONFIG.get('bot_token')
        chat  = os.getenv("TELEGRAM_CHAT_ID")   or TELEGRAM_CONFIG.get('chat_id')

        if token and token != "YOUR_BOT_TOKEN" and chat:
            try:
                self.telegram_notifier = TradingBotNotifier(token, chat)
                logger.info("✅ Telegram notifier ready")
            except Exception as e:
                logger.warning("⚠️  Telegram init failed: %s", e)
        else:
            logger.warning("⚠️  Telegram not configured")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _signal_handler(self, signum, frame):
        logger.info("\n⏹️  Shutdown signal received…")
        self.stop()
        sys.exit(0)

    async def _telegram(self, coro):
        """Fire-and-forget telegram call (never blocks trading loop)."""
        if not self.telegram_notifier:
            return
        try:
            await coro
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def fetch_market_data(self, symbol='BTC/USDT:USDT', timeframe='15m', limit=200):
        try:
            df = self.data_collector.fetch_ohlcv(symbol, timeframe, 'bybit', limit)
            if df.empty:
                return None
            orderbook   = self.data_collector.fetch_orderbook(symbol, 'bybit', 20)
            funding     = self.data_collector.fetch_funding_rate(symbol, 'bybit')
            oi          = self.data_collector.fetch_open_interest(symbol, 'bybit')
            large_orders= self.data_collector.detect_large_orders(
                symbol, UNIQUE_LOGICS['large_order_threshold'], 'bybit')
            divergence  = self.data_collector.calculate_price_divergence(symbol)
            return {
                'df':df, 'orderbook':orderbook, 'funding_rate':funding,
                'open_interest':oi, 'large_orders':large_orders,
                'divergence':divergence, 'timestamp':datetime.now()
            }
        except Exception as e:
            logger.error("❌ fetch_market_data: %s", e)
            return None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def generate_features(self, df):
        try:
            feats = pd.DataFrame(index=df.index)
            feats['close']    = df['close']
            feats['high']     = df['high']
            feats['low']      = df['low']
            feats['volume']   = df['volume']
            feats['returns']  = df['close'].pct_change()

            for p in [5,10,20,50]:
                feats[f'sma_{p}'] = df['close'].rolling(p).mean()
                feats[f'ema_{p}'] = df['close'].ewm(span=p).mean()

            feats['volatility'] = df['close'].rolling(20).std()
            delta = df['close'].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss
            feats['rsi'] = 100 - (100 / (1 + rs))

            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            feats['macd']        = ema12 - ema26
            feats['macd_signal'] = feats['macd'].ewm(span=9).mean()

            return feats.bfill().fillna(0)        # Railway-fix: no method=
        except Exception as e:
            logger.error("❌ generate_features: %s", e)
            return pd.DataFrame()
    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------
    def analyze_trade_opportunity(self, symbol='BTC/USDT:USDT'):
        logger.info("\n" + "="*100)
        logger.info("🔍 ANALYSING: %s", symbol)
        logger.info("="*100)

        mdata = self.fetch_market_data(symbol)
        if not mdata:
            return None
        df, orderbook, funding = mdata['df'], mdata['orderbook'], mdata['funding_rate']

        logger.info("📊 %d candles, Price ₹%,.2f", len(df), df['close'].iloc[-1])

        # 1. ML signal
        logger.info("🤖 ML Prediction…")
        features = self.generate_features(df)
        if features.empty:
            return None
        try:
            signal, conf = self.ensemble.generate_signal(
                features.tail(100).values,
                ML_CONFIG['confidence_threshold']
            )
            signal_name = ['BUY', 'HOLD', 'SELL'][signal]
            logger.info("   Signal: %s | Confidence: %.1f%%", signal_name, conf*100)
            if signal == 1:   # HOLD
                return None
        except Exception as e:
            logger.error("❌ ML failed: %s – fallback to TA only", e)
            return None

        # 2. 45-logics filter
        logger.info("🧠 45-logics evaluation…")
        logic_res = self.logic_evaluator.evaluate_all_logics(
            df=df, orderbook=orderbook, funding_rate=funding,
            oi_history=[mdata['open_interest']]*20,
            recent_trades=self.trades_today,
            fear_greed_index=50, news_times=[], liquidation_clusters=[]
        )
        logger.info("   Logic score: %.1f%% | Trade allowed: %s",
                    logic_res['final_score'], logic_res['trade_allowed'])
        if not logic_res['trade_allowed']:
            logger.warning("❌ Trade blocked by filters")
            return None

        # 3. Build trade plan
        curr_p     = df['close'].iloc[-1]
        side       = 'LONG' if signal == 0 else 'SHORT'
        regime     = logic_res['market_health']['market_regime']
        atr        = logic_res['price_action']['atr']
        pos_size   = self.risk_manager.calculate_position_size(
            self.capital, conf, regime)
        leverage   = self.risk_manager.calculate_leverage(
            regime, atr/curr_p if curr_p else 0.02, conf)
        stop_loss  = self.risk_manager.calculate_stop_loss(
            curr_p, side, leverage, atr)
        take_profit= self.risk_manager.calculate_take_profit(
            curr_p, side, conf)

        is_valid, reason = self.risk_manager.validate_trade(
            self.capital, pos_size, leverage, curr_p, stop_loss, side)
        if not is_valid:
            logger.warning("❌ Validation failed: %s", reason)
            return None

        plan = {
            'symbol':symbol, 'side':side, 'signal':signal_name,
            'confidence':conf, 'entry_price':curr_p,
            'position_size':pos_size, 'leverage':leverage,
            'stop_loss':stop_loss, 'take_profit':take_profit,
            'logic_score':logic_res['final_score'],
            'market_regime':regime, 'timestamp':datetime.now(),
            'large_orders_detected':len(mdata['large_orders']),
            'funding_rate':funding
        }

        logger.info("\n" + "="*100)
        logger.info("✅ TRADE PLAN READY")
        logger.info("="*100)
        logger.info("📊 %s (ML %.1f%%) | Logic %.1f%%", signal_name, conf*100,
                    logic_res['final_score'])
        logger.info("💰 Entry ₹%,.2f | Size ₹%,.2f (%dx)", curr_p, pos_size, leverage)
        logger.info("🛑 SL ₹%,.2f | TP ₹%,.2f", stop_loss, take_profit)
        logger.info("="*100)
        return plan

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute_trade(self, plan):
        try:
            if self.mode == 'paper':
                logger.info("📝 PAPER trade – simulating…")
                trade = {**plan,
                         'id': f"paper_{int(time.time())}",
                         'status':'open','pnl':0,
                         'opened_at':datetime.now()}
                self.trades_today.append(trade)
                self.open_positions.append(trade)
                asyncio.create_task(
                    self._telegram(self.telegram_notifier.notify_trade_opened(trade))
                )
                return trade

            elif self.mode == 'live' and self.executor:
                logger.info("⚡ LIVE trade – executing…")
                pos = self.executor.open_position(
                    plan['symbol'], plan['side'], plan['position_size'],
                    plan['leverage'], plan['stop_loss'], plan['take_profit'],
                    'bybit')
                if 'error' not in pos:
                    self.trades_today.append(pos)
                    self.open_positions.append(pos)
                    self.risk_manager.add_position(pos)
                    asyncio.create_task(
                        self._telegram(self.telegram_notifier.notify_trade_opened(pos))
                    )
                    if self.db_conn:
                        self._log_trade_to_db(pos)
                else:
                    logger.error("❌ Live exec failed: %s", pos.get('error'))
                return pos
        except Exception as e:
            logger.error("❌ execute_trade: %s", e)
            return {'error':str(e)}

    # ------------------------------------------------------------------
    # Position monitor
    # ------------------------------------------------------------------
    def monitor_positions(self):
        if not self.open_positions:
            return
        logger.info("👀 Monitoring %d open position(s)…", len(self.open_positions))
        for pos in self.open_positions[:]:
            try:
                mdata = self.fetch_market_data(pos['symbol'], limit=50)
                if not mdata:
                    continue
                curr_p = mdata['df']['close'].iloc[-1]
                side_mul = 1 if pos['side']=='LONG' else -1
                pnl_pct  = (curr_p - pos['entry_price'])/pos['entry_price']*side_mul
                unreal   = pnl_pct * pos['position_size'] * pos['leverage']
                pos['current_price'] = curr_p
                pos['unrealized_pnl'] = unreal
                logger.info("   %s %s: ₹%,.2f | P&L ₹%+,.2f",
                            pos['symbol'], pos['side'], curr_p, unreal)

                # time-limit close (4 h)
                hrs_open = (datetime.now()-pos['opened_at']).total_seconds()/3600
                if hrs_open >= 4:
                    logger.warning("⏰ Time-limit – closing %s", pos['id'])
                    self._close_position(pos, 'time_limit')
            except Exception as e:
                logger.error("❌ monitor_positions: %s", e)

    def _close_position(self, pos, reason='manual'):
        logger.info("📉 Closing %s (%s)", pos['id'], reason)
        if self.mode == 'paper':
            pos['status']='closed'; pos['exit_price']=pos.get('current_price',pos['entry_price'])
            pos['pnl']=pos.get('unrealized_pnl',0); pos['closed_at']=datetime.now()
            pos['close_reason']=reason
            self.performance['total_pnl'] += pos['pnl']
            self.performance['daily_pnl'] += pos['pnl']
            self.risk_manager.update_trade_result(pos['pnl'])
            self.open_positions = [p for p in self.open_positions if p['id']!=pos['id']]
            asyncio.create_task(
                self._telegram(self.telegram_notifier.notify_trade_closed(pos))
            )
            logger.info("   P&L ₹%+,.2f", pos['pnl'])

    def _log_trade_to_db(self, trade):
        if not self.db_conn:
            return
        try:
            cur = self.db_conn.cursor()
            cur.execute("""
                INSERT INTO trades (trade_id,symbol,side,entry_time,entry_price,size,leverage,stop_loss,take_profit,ml_confidence,logic_score,status,exchange)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (trade['id'], trade['symbol'], trade['side'], trade['opened_at'],
                  trade['entry_price'], trade['position_size'], trade['leverage'],
                  trade['stop_loss'], trade['take_profit'], trade['confidence'],
                  trade['logic_score'], 'open', 'bybit'))
            self.db_conn.commit()
        except Exception as e:
            logger.warning("DB log fail: %s", e)

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------
    def _update_performance(self):
        closed = [t for t in self.trades_today if t.get('status')=='closed']
        if closed:
            wins = [t for t in closed if t.get('pnl',0)>0]
            self.performance.update({
                'wins':len(wins), 'losses':len(closed)-len(wins),
                'win_rate':len(wins)/len(closed) if closed else 0,
                'total_trades':len(closed)
            })

    def _print_daily_summary(self):
        logger.info("\n" + "="*100)
        logger.info("📊 DAILY SUMMARY")
        logger.info("="*100)
        logger.info("Date : %s", datetime.now().date())
        logger.info("Trades : %d  (W %d | L %d)  Win-rate %.1f%%",
                    self.performance['total_trades'],
                    self.performance['wins'], self.performance['losses'],
                    self.performance['win_rate']*100)
        logger.info("Daily P&L : ₹%+,.2f", self.performance['daily_pnl'])
        logger.info("Total P&L : ₹%+,.2f", self.performance['total_pnl'])
        logger.info("Capital   : ₹%,.2f", self.capital)
        logger.info("ROI       : %+.2f%%",
                    (self.capital-self.initial_capital)/self.initial_capital*100)
        logger.info("="*100)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, symbol='BTC/USDT:USDT', interval=900):
        self.is_running = True
        logger.info("\n" + "="*100)
        logger.info("🚀 TRADING BOT STARTED – FINAL")
        logger.info("="*100)
        logger.info("Mode : %s | Symbol : %s | Interval : %ds (%d min)",
                    self.mode.upper(), symbol, interval, interval//60)
        logger.info("Capital : ₹%,.2f | Logics : 45 | ML : ON | Risk : ON",
                    self.capital)
        logger.info("="*100)

        # Notify Telegram
        asyncio.run(self._telegram(
            self.telegram_notifier.notify_bot_started(self.capital, self.mode)))

        iteration = 0
        while self.is_running:
            iteration +=1
            try:
                logger.info("\n" + "="*100)
                logger.info("🔄 ITERATION #%d – %s", iteration, datetime.now())
                logger.info("="*100)

                # Daily limit check
                can_trade, reason = self.risk_manager.check_daily_limits()
                if not can_trade:
                    logger.warning("⏸️  Trading paused: %s", reason)
                    asyncio.run(self._telegram(
                        self.telegram_notifier.notify_daily_loss_limit(
                            self.risk_manager.daily_pnl,
                            self.initial_capital * RISK_CONFIG['max_daily_loss_percent'])))
                    time.sleep(interval); continue

                # Monitor existing
                if self.open_positions:
                    self.monitor_positions()

                # New trade?
                max_pos = TRADING_CONFIG['max_positions']
                if len(self.open_positions) < max_pos:
                    plan = self.analyze_trade_opportunity(symbol)
                    if plan:
                        trade = self.execute_trade(plan)
                        if 'error' not in trade:
                            self.last_signal_time = datetime.now()
                        else:
                            logger.error("❌ Trade failed: %s", trade.get('error'))
                    else:
                        logger.info("⏭️  No opportunity – waiting…")
                else:
                    logger.info("⏸️  Max positions (%d) reached", max_pos)

                self._update_performance()
                if iteration % 10 == 0:
                    self._print_daily_summary()

                logger.info("\n😴 Sleeping %d s – next : %s",
                            interval, (datetime.now()+timedelta(seconds=interval)).strftime('%H:%M:%S'))
                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("\n⏹️  Keyboard interrupt"); break
            except Exception as e:
                logger.error("❌ Main-loop error: %s", e)
                asyncio.run(self._telegram(
                    self.telegram_notifier.notify_error(str(e))))
                time.sleep(60)

        self.stop()

    # ------------------------------------------------------------------
    # Graceful stop
    # ------------------------------------------------------------------
    def stop(self):
        logger.info("\n" + "="*100)
        logger.info("⏹️  STOPPING BOT…")
        logger.info("="*100)
        self.is_running = False

        if self.open_positions:
            logger.warning("⚠️  %d position(s) still open!", len(self.open_positions))
        self._print_daily_summary()

        asyncio.run(self._telegram(self.telegram_notifier.notify_bot_stopped()))
        if self.db_conn:
            self.db_conn.close(); logger.info("🗄️  DB closed")

        logger.info("👋 BOT STOPPED – FINAL"); logger.info("="*100)


# ==========================================================================
# Entry-point
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="Advanced Crypto Trading Bot – Final")
    parser.add_argument('--mode',    default='paper', choices=['paper','live'])
    parser.add_argument('--capital', type=float, default=10_000)
    parser.add_argument('--symbol',  default='BTC/USDT:USDT')
    parser.add_argument('--interval',type=int,  default=900, help="sec (default 900)")
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--debug',  action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.validate:
        logger.info("🔍 Running validation…")
        try:
            from scripts.validate_config import ConfigValidator
            if not ConfigValidator().run_all_checks():
                logger.error("❌ Validation failed"); sys.exit(1)
            logger.info("✅ Validation passed")
        except Exception as e:
            logger.warning("⚠️  Validation skip: %s", e)

    if args.mode == 'live':
        logger.warning("\n" + "="*100)
        logger.warning("⚠️  LIVE MODE – REAL MONEY AT RISK")
        logger.warning("Capital : ₹%,.2f", args.capital)
        logger.warning("="*100)
        confirm = input("Type 'LIVE' to continue: ")
        if confirm != 'LIVE':
            logger.info("❌ Live cancelled"); sys.exit(0)

    bot = TradingBot(mode=args.mode, capital=args.capital)
    bot.run(symbol=args.symbol, interval=args.interval)


if __name__ == "__main__":
    main()
