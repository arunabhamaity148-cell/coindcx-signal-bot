#!/usr/bin/env python3
"""
MAIN TRADING BOT – FINAL (TA-ONLY READY)
- ENV switch MODE=live/paper
- Health-check for Railway
- Contract symbols (OI/funding ready)
- Premium Telegram msgs
"""
import os
import sys
import time
import asyncio
import logging
import signal
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from aiohttp import web      # ← health server

# ---------- project imports ----------
from config.settings import (
    EXCHANGES, TRADING_CONFIG, RISK_CONFIG, ML_CONFIG,
    UNIQUE_LOGICS, TELEGRAM_CONFIG, DATA_CONFIG
)
from data.data_collector import DataCollector
from strategy.unique_logics import LogicEvaluator
from risk.risk_manager import RiskManager
from execution.order_executor import OrderExecutor
from monitoring.telegram_bot import TradingBotNotifier

# ---------- logging ----------
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


# ---------- health-check server (Railway keeps alive) ----------
async def start_health_server():
    app = web.Application()
    app.router.add_get('/health', lambda req: web.Response(text="OK"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', int(os.getenv("PORT", 8080)))
    await site.start()
    logger.info("Health-check server started on port %s", os.getenv("PORT", 8080))


# ---------- TradingBot ----------
class TradingBot:
    def __init__(self, mode=None, capital=None):
        # ENV preference → fallback arg
        self.mode = mode or os.getenv("MODE", "paper")
        self.capital = capital if capital is not None else TRADING_CONFIG.get('initial_capital', 10000)
        self.initial_capital = self.capital
        self.is_running = False
        self.open_positions = []
        self.trades_today = []
        self.performance = {
            'total_trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0,
            'win_rate': 0.0, 'daily_pnl': 0, 'max_drawdown': 0,
            'equity_curve': []
        }

        logger.info("="*100)
        logger.info("🤖 ADVANCED CRYPTO TRADING BOT – FINAL")
        logger.info("Mode : %s  |  Capital : ₹%0.2f  |  Date : %s",
                    self.mode.upper(), self.capital, datetime.now())
        logger.info("="*100)

        self._init_components()
        self._init_telegram()
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ---------- components ----------
    def _init_components(self):
        logger.info("📊 INITIALISING COMPONENTS...")

        # 1) Data
        logger.info("1/6 📊 Data Collector...")
        self.data_collector = DataCollector(EXCHANGES)
        logger.info("    ✅ Multi-exchange ready")

        # 2) 45-logics
        logger.info("2/6 🧠 Logic Evaluator (45 unique logics)...")
        self.logic_evaluator = LogicEvaluator(UNIQUE_LOGICS)
        logger.info("    ✅ 45 unique logics loaded")

        # 3) ML (only if required)
        ml_required = TRADING_CONFIG.get('ml_required', True)
        self.ml_available = False
        if ml_required:
            try:
                from ml.lstm_model import LSTMModel
                from ml.xgboost_model import XGBoostModel
                from ml.ensemble import EnsembleModel

                self.lstm_model = LSTMModel(
                    ML_CONFIG.get('lookback_candles', 60),
                    ML_CONFIG.get('features_count', 50),
                    ML_CONFIG.get('model_path', 'models/')
                )
                self.xgb_model = XGBoostModel(ML_CONFIG.get('model_path', 'models/'))
                self.ensemble = EnsembleModel(
                    self.lstm_model, self.xgb_model,
                    ML_CONFIG.get('lstm_weight', 0.5),
                    ML_CONFIG.get('xgboost_weight', 0.3),
                    ML_CONFIG.get('rf_weight', 0.2),
                    ML_CONFIG.get('model_path', 'models/')
                )

                self.lstm_model.load('lstm_model.h5')
                self.xgb_model.load('xgboost_model.pkl')
                self.ensemble.load_rf('random_forest.pkl')
                self.ml_available = True
                logger.info("    ✅ ML models loaded")
            except Exception as e:
                logger.warning("    ⚠️  ML models failed (%s) → TA-only mode", e)
                self.ml_available = False
        else:
            logger.info("    🚫 ML disabled by config → TA-only mode")

        # 4) Risk
        logger.info("4/6 🛡️ Risk Manager...")
        self.risk_manager = RiskManager(RISK_CONFIG)
        self.risk_manager.initial_daily_capital = self.capital
        logger.info("    ✅ Risk system active")

        # 5) Executor
        logger.info("5/6 ⚡ Order Executor...")
        if self.mode == 'live':
            self.executor = OrderExecutor(EXCHANGES)
            logger.info("    ✅ Live executor")
        else:
            self.executor = None
            logger.info("    ✅ Paper mode")

        # 6) DB
        logger.info("6/6 🗄️  Database...")
        try:
            import psycopg2
            db_url = os.getenv("DATABASE_URL")
            self.db_conn = psycopg2.connect(db_url) if db_url else None
            logger.info("    ✅ DB connected" if self.db_conn else "    ⚠️  DB URL not set")
        except Exception as e:
            logger.warning("    ⚠️  DB error: %s", e)
            self.db_conn = None

        logger.info("✅ ALL COMPONENTS INITIALISED")

    # ---------- telegram ----------
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

    # ---------- helpers ----------
    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received… stopping bot")
        self.is_running = False

    async def _telegram(self, coro):
        if not self.telegram_notifier:
            return
        try:
            await coro
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)

    # ---------- market fetch ----------
    def fetch_market_data(self, symbol, timeframe=None, limit=200):
        tf = timeframe or TRADING_CONFIG.get('timeframe', '15m')
        try:
            df = self.data_collector.fetch_ohlcv(symbol, tf, 'bybit', limit)
            if df is None or df.empty:
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
            logger.error("fetch_market_data error: %s", e)
            return None

    # ---------- feature engineering ----------
    def generate_features(self, df):
        try:
            feats = pd.DataFrame(index=df.index)
            feats['close']  = df['close']
            feats['high']   = df['high']
            feats['low']    = df['low']
            feats['volume'] = df['volume']
            feats['returns']= df['close'].pct_change().fillna(0)
            for p in [5,10,20,50]:
                feats[f'sma_{p}'] = df['close'].rolling(p).mean()
                feats[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
            feats['volatility'] = df['close'].rolling(20).std()
            delta = df['close'].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / (loss.replace(0, np.nan))
            feats['rsi'] = 100 - (100 / (1 + rs))
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            feats['macd'] = ema12 - ema26
            feats['macd_signal'] = feats['macd'].ewm(span=9, adjust=False).mean()
            return feats.bfill().ffill().fillna(0)
        except Exception as e:
            logger.error("generate_features: %s", e)
            return pd.DataFrame()

    # ---------- analyze & trade plan ----------
    def analyze_trade_opportunity(self, symbol):
        logger.info("\n🔍 ANALYSING: %s", symbol)
        mdata = self.fetch_market_data(symbol)
        if not mdata:
            logger.info("No market data for %s", symbol)
            return None

        df = mdata['df']
        orderbook = mdata['orderbook']
        funding = mdata['funding_rate']

        logger.info("📊 %d candles, Price ₹%0.2f", len(df), df['close'].iloc[-1])

        features = self.generate_features(df)
        if features.empty:
            return None

        # 1) 45-logics gate
        logic_res = self.logic_evaluator.evaluate_all_logics(
            df=df, orderbook=orderbook, funding_rate=funding,
            oi_history=[mdata['open_interest']]*20,
            recent_trades=self.trades_today,
            fear_greed_index=50, news_times=[], liquidation_clusters=[]
        )
        logger.info("   Logic Score: %.1f%% | Trade allowed: %s",
                    logic_res.get('final_score', 0.0), logic_res.get('trade_allowed', False))

        if not logic_res.get('trade_allowed', False):
            logger.warning("Blocked by logics")
            return None

        # 2) ML or TA signal
        ml_required = TRADING_CONFIG.get('ml_required', True)
        signal = None
        conf = 0.0

        if ml_required and self.ml_available:
            try:
                sig, confidence = self.ensemble.generate_signal(
                    features.tail(ML_CONFIG.get('lookback_candles', 100)).values,
                    ML_CONFIG.get('confidence_threshold', 0.7)
                )
                signal = ['BUY', 'HOLD', 'SELL'][sig]
                conf = confidence
                logger.info("   ML Signal: %s | Conf: %.1f%%", signal, conf*100)
                if signal == 'HOLD':
                    logger.info("ML says HOLD")
                    return None
            except Exception as e:
                logger.warning("ML failed (%s) → TA fallback", e)
                ml_required = False

        if not ml_required or not self.ml_available:
            # TA-only
            ema_align = self.logic_evaluator.check_ema_alignment(df)
            structure = self.logic_evaluator.detect_market_structure_shift(df)
            if ema_align == 'bullish' or structure == 'bullish':
                signal = 'BUY'
            elif ema_align == 'bearish' or structure == 'bearish':
                signal = 'SELL'
            else:
                logger.info("TA indecisive")
                return None
            conf = max(0.05, min(1.0, logic_res.get('final_score', 50) / 100.0))
            logger.info("   TA Signal: %s | Conf: %.1f%%", signal, conf*100)

        # 3) Build plan
        curr_p = df['close'].iloc[-1]
        side = 'LONG' if signal == 'BUY' else 'SHORT'
        regime = logic_res.get('market_health', {}).get('market_regime', 'ranging')
        atr = logic_res.get('price_action', {}).get('atr', 0.0)

        pos_size = self.risk_manager.calculate_position_size(self.capital, conf, regime)
        leverage = self.risk_manager.calculate_leverage(regime, (atr / curr_p) if curr_p else 0.02, conf)
        stop_loss = self.risk_manager.calculate_stop_loss(curr_p, side, leverage, atr)
        take_profit = self.risk_manager.calculate_take_profit(curr_p, side, conf)

        is_valid, reason = self.risk_manager.validate_trade(self.capital, pos_size, leverage, curr_p, stop_loss, side)
        if not is_valid:
            logger.warning("Validation fail: %s", reason)
            return None

        plan = {
            'symbol': symbol,
            'side': side,
            'signal': signal,
            'confidence': conf,
            'entry_price': curr_p,
            'position_size': pos_size,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'logic_score': logic_res.get('final_score', 0.0),
            'market_regime': regime,
            'timestamp': datetime.now(),
            'large_orders_detected': len(mdata.get('large_orders', [])),
            'funding_rate': funding
        }

        logger.info("✅ TRADE PLAN READY: %s %s (conf %.1f%%)", symbol, side, conf*100)
        return plan

    # ---------- execute ----------
    def execute_trade(self, plan):
        try:
            if self.mode == 'paper':
                trade = {**plan, 'id': f"paper_{int(time.time())}", 'status':'open',
                         'pnl':0, 'opened_at': datetime.now()}
                self.trades_today.append(trade)
                self.open_positions.append(trade)
                if self.telegram_notifier:
                    asyncio.create_task(self._telegram(self.telegram_notifier.notify_trade_opened(trade)))
                logger.info("Simulated PAPER trade opened: %s", trade['id'])
                return trade
            else:
                # live
                pos = self.executor.open_position(
                    plan['symbol'], plan['side'], plan['position_size'],
                    plan['leverage'], plan['stop_loss'], plan['take_profit'], 'bybit'
                )
                if 'error' not in pos:
                    self.trades_today.append(pos)
                    self.open_positions.append(pos)
                    self.risk_manager.add_position(pos)
                    if self.telegram_notifier:
                        asyncio.create_task(self._telegram(self.telegram_notifier.notify_trade_opened(pos)))
                    if self.db_conn:
                        self._log_trade_to_db(pos)
                    logger.info("LIVE trade executed: %s", pos.get('id', 'unknown'))
                else:
                    logger.error("Live exec error: %s", pos.get('error'))
                return pos
        except Exception as e:
            logger.error("execute_trade error: %s", e)
            return {'error': str(e)}

    # ---------- monitor ----------
    def monitor_positions(self):
        if not self.open_positions:
            return
        logger.info("Monitoring %d open position(s)", len(self.open_positions))
        for pos in list(self.open_positions):
            try:
                mdata = self.fetch_market_data(pos['symbol'], limit=50)
                if not mdata:
                    continue
                curr_p = mdata['df']['close'].iloc[-1]
                side_mul = 1 if pos['side']=='LONG' else -1
                pnl_pct = (curr_p - pos['entry_price'])/pos['entry_price'] * side_mul
                unreal = pnl_pct * pos['position_size'] * pos.get('leverage',1)
                pos['current_price'] = curr_p
                pos['unrealized_pnl'] = unreal
                logger.info("  %s %s: ₹%0.2f | Unreal P&L ₹%+0.2f", pos['symbol'], pos['side'], curr_p, unreal)

                # time limit
                hrs_open = (datetime.now() - pos['opened_at']).total_seconds() / 3600.0
                if hrs_open >= TRADING_CONFIG.get('max_trade_duration_hours', 4):
                    logger.warning("Time-limit closing %s", pos.get('id'))
                    self._close_position(pos, 'time_limit')
            except Exception as e:
                logger.error("monitor_positions error: %s", e)

    def _close_position(self, pos, reason='manual'):
        try:
            logger.info("Closing %s (%s)", pos.get('id'), reason)
            if self.mode == 'paper':
                pos['status'] = 'closed'
                pos['exit_price'] = pos.get('current_price', pos['entry_price'])
                pos['pnl'] = pos.get('unrealized_pnl', 0)
                pos['closed_at'] = datetime.now()
                pos['close_reason'] = reason
                self.performance['total_pnl'] += pos['pnl']
                self.performance['daily_pnl'] += pos['pnl']
                self.risk_manager.update_trade_result(pos['pnl'])
                self.open_positions = [p for p in self.open_positions if p.get('id') != pos.get('id')]
                if self.telegram_notifier:
                    asyncio.create_task(self._telegram(self.telegram_notifier.notify_trade_closed(pos)))
                logger.info("Closed paper pos %s P&L ₹%+0.2f", pos.get('id'), pos['pnl'])
            else:
                # live close (integrate executor)
                logger.info("Live close requested (executor integration required)")
        except Exception as e:
            logger.error("_close_position error: %s", e)

    # ---------- performance helpers ----------
    def _update_performance(self):
        total = self.performance['wins'] + self.performance['losses']
        if total:
            self.performance['win_rate'] = self.performance['wins'] / total
            self.performance['equity_curve'].append(self.capital)

    def _print_daily_summary(self):
        self._update_performance()
        logger.info("="*60)
        logger.info("📈 DAILY SUMMARY – %s", datetime.now().date())
        logger.info("Trades today : %d", len(self.trades_today))
        logger.info("Win / Loss   : %d / %d", self.performance['wins'], self.performance['losses'])
        logger.info("Win Rate     : %.1f%%", self.performance['win_rate']*100)
        logger.info("Daily P&L    : ₹%+0.2f", self.performance['daily_pnl'])
        logger.info("Total P&L    : ₹%+0.2f", self.performance['total_pnl'])
        logger.info("Capital      : ₹%0.2f", self.capital)
        logger.info("="*60)

    # ---------- run loop ----------
    async def _run_loop(self):
        await start_health_server()   # ← health-check start
        self.is_running = True
        symbols = TRADING_CONFIG.get('symbols', [TRADING_CONFIG.get('primary_symbol', 'BTC/USDT:USDT')])
        interval = TRADING_CONFIG.get('check_interval', 900)
        max_positions = TRADING_CONFIG.get('max_positions', 3)

        logger.info("Starting async loop: %d symbols | interval: %ds | ml_required: %s",
                    len(symbols), interval, TRADING_CONFIG.get('ml_required', True))

        if self.telegram_notifier:
            try:
                await self.telegram_notifier.notify_bot_started(self.capital, self.mode)
            except Exception as e:
                logger.warning("Telegram start msg failed: %s", e)

        while self.is_running:
            start_cycle = datetime.now()
            logger.info("Cycle start: %s", start_cycle)

            for symbol in symbols:
                if len(self.open_positions) >= max_positions:
                    logger.info("Max positions reached (%d). Skipping further symbols.", max_positions)
                    break

                try:
                    plan = self.analyze_trade_opportunity(symbol)
                    if plan:
                        self.execute_trade(plan)
                except Exception as e:
                    logger.error("Error analyzing/executing %s: %s", symbol, e)

                await asyncio.sleep(1)

            try:
                self.monitor_positions()
            except Exception as e:
                logger.error("monitor_positions loop error: %s", e)

            # daily summary
            try:
                ds_time = TELEGRAM_CONFIG.get('daily_summary_time', '18:00')
                hh, mm = map(int, ds_time.split(':'))
                now = datetime.now()
                if now.hour == hh and now.minute == mm:
                    self._print_daily_summary()
                    if self.telegram_notifier:
                        await self._telegram(self.telegram_notifier.notify_daily_summary({
                            'daily_pnl': self.performance.get('daily_pnl', 0),
                            'total_trades': self.performance.get('total_trades', 0),
                            'wins': self.performance.get('wins', 0),
                            'losses': self.performance.get('losses', 0),
                            'win_rate': self.performance.get('win_rate', 0),
                            'capital': self.capital,
                            'roi': (self.capital - self.initial_capital) / max(1, self.initial_capital)
                        }))
                    await asyncio.sleep(61)
            except Exception as e:
                logger.error("Daily summary error: %s", e)

            elapsed = (datetime.now() - start_cycle).total_seconds()
            to_sleep = max(1, interval - elapsed)
            logger.info("Cycle complete. Sleeping %0.1fs until next cycle.", to_sleep)
            await asyncio.sleep(to_sleep)

    def run(self):
        try:
            asyncio.run(self._run_loop())
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Stopping...")
        except Exception as e:
            logger.error("Run error: %s", e)
        finally:
            self.is_running = False


# ---------- CLI (ENV preference) ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['paper','live'], default=os.getenv("MODE", "paper"))
    parser.add_argument('--capital', type=float, default=None)
    args = parser.parse_args()

    bot = TradingBot(mode=args.mode, capital=args.capital)
    bot.run()

if __name__ == "__main__":
    main()
