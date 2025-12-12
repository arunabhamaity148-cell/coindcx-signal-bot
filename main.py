"""
MAIN TRADING BOT – FINAL (Railway-deploy)
- 45 logics, ML ensemble, multi-exchange, risk, Telegram, DB
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
    """Final bot – Railway ready (no polling, minimal memory)."""
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
