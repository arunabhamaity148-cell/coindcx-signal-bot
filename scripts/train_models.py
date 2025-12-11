"""
ML Model Training Script
Trains LSTM, XGBoost, and Random Forest on historical data
Performs train/validation/test split and saves trained models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import logging
import argparse
from datetime import datetime

from ml.lstm_model import LSTMModel
from ml.xgboost_model import XGBoostModel
from ml.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, data_path, lookback=60, features_count=50):
        self.data_path = data_path
        self.lookback = lookback
        self.features_count = features_count
        self.df = None
        
    def load_and_prepare_data(self):
        """Load data and prepare features"""
        logger.info("="*80)
        logger.info("📊 LOADING AND PREPARING DATA")
        logger.info("="*80)
        
        # Load data
        self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        logger.info(f"✅ Loaded {len(self.df):,} rows")
        logger.info(f"Date range: {self.df.index[0]} to {self.df.index[-1]}")
        
        # Generate features
        logger.info("\n🔧 Engineering features...")
        features_df = self._create_features(self.df)
        
        logger.info(f"✅ Created {len(features_df.columns)} features")
        logger.info("="*80)
        
        return features_df
    
    def _create_features(self, df):
        """Create comprehensive feature set"""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['close'] = df['close']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume']
        
        # Returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Volatility
        features['volatility_10'] = df['close'].rolling(10).std()
        features['volatility_20'] = df['close'].rolling(20).std()
        features['volatility_50'] = df['close'].rolling(50).std()
        
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + (2 * std)
            features[f'bb_lower_{period}'] = sma - (2 * std)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - 
                                              features[f'bb_lower_{period}']) / sma
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        features['atr_14'] = ranges.max(axis=1).rolling(14).mean()
        
        # Volume indicators
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        
        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        features['obv'] = obv
        
        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Fill NaN values
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return features
    
    def create_labels(self, df, threshold=0.005):
        """
        Create labels for classification
        0 = BUY (price will drop, good for LONG entry)
        1 = HOLD (price stable)
        2 = SELL (price will rise, good for SHORT entry)
        """
        logger.info(f"🏷️ Creating labels (threshold: {threshold:.2%})...")
        
        # Future return
        future_return = df['close'].shift(-1).pct_change()
        
        def create_label(ret):
            if pd.isna(ret):
                return 1
            elif ret < -threshold:  # Price drops
                return 0  # BUY signal
            elif ret > threshold:   # Price rises
                return 2  # SELL signal
            else:
                return 1  # HOLD
        
        labels = future_return.apply(create_label)
        
        # Convert to one-hot encoding
        labels_onehot = keras.utils.to_categorical(labels, 3)
        
        # Distribution
        unique, counts = np.unique(labels, return_counts=True)
        logger.info("Label distribution:")
        for label, count in zip(unique, counts):
            pct = count / len(labels) * 100
            label_name = ['BUY', 'HOLD', 'SELL'][int(label)]
            logger.info(f"  {label_name} ({int(label)}): {count:,} ({pct:.1f}%)")
        
        return labels_onehot
    
    def split_data(self, features, labels, train_ratio=0.70, val_ratio=0.15):
        """Split data into train/val/test sets"""
        logger.info("\n✂️ Splitting data...")
        
        total = len(features)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # Time-based split (important for time series!)
        X_train = features[:train_size]
        y_train = labels[:train_size]
        
        X_val = features[train_size:train_size + val_size]
        y_val = labels[train_size:train_size + val_size]
        
        X_test = features[train_size + val_size:]
        y_test = labels[train_size + val_size:]
        
        logger.info(f"Train set: {len(X_train):,} samples ({train_ratio:.0%})")
        logger.info(f"Val set:   {len(X_val):,} samples ({val_ratio:.0%})")
        logger.info(f"Test set:  {len(X_test):,} samples ({1-train_ratio-val_ratio:.0%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_all_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train all three models"""
        logger.info("\n" + "="*80)
        logger.info("🤖 TRAINING ML MODELS")
        logger.info("="*80)
        
        # 1. Train LSTM
        logger.info("\n1️⃣ Training LSTM Model...")
        lstm = LSTMModel(lookback=self.lookback, features=X_train.shape[1])
        lstm.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
        lstm.save('lstm_model.h5')
        lstm_results = lstm.evaluate(X_test, y_test)
        
        # 2. Train XGBoost
        logger.info("\n2️⃣ Training XGBoost Model...")
        xgb = XGBoostModel(n_estimators=200, max_depth=8, learning_rate=0.1)
        xgb.train(X_train, y_train, X_val, y_val)
        xgb.save('xgboost_model.pkl')
        xgb_results = xgb.evaluate(X_test, y_test)
        
        # 3. Train Random Forest (part of ensemble)
        logger.info("\n3️⃣ Training Random Forest Model...")
        ensemble = EnsembleModel(
            lstm_model=lstm,
            xgboost_model=xgb,
            lstm_weight=0.50,
            xgb_weight=0.30,
            rf_weight=0.20
        )
        ensemble.train_rf(X_train, y_train)
        ensemble.save_rf('random_forest.pkl')
        
        # 4. Evaluate Ensemble
        logger.info("\n4️⃣ Evaluating Ensemble Model...")
        ensemble_results = ensemble.evaluate(X_test, y_test)
        
        logger.info("\n" + "="*80)
        logger.info("✅ ALL MODELS TRAINED SUCCESSFULLY")
        logger.info("="*80)
        
        return {
            'lstm': lstm_results,
            'xgboost': xgb_results,
            'ensemble': ensemble_results
        }
    
    def print_comparison(self, results):
        """Print model comparison"""
        logger.info("\n" + "="*80)
        logger.info("📊 MODEL COMPARISON")
        logger.info("="*80)
        
        print("\n{:<15} {:<12} {:<12} {:<12} {:<12}".format(
            "Model", "Accuracy", "BUY Acc", "HOLD Acc", "SELL Acc"
        ))
        print("-" * 65)
        
        for model_name, result in results.items():
            acc = result['accuracy']
            buy_acc = result['class_accuracy'].get(0, 0)
            hold_acc = result['class_accuracy'].get(1, 0)
            sell_acc = result['class_accuracy'].get(2, 0)
            
            print("{:<15} {:<12.2%} {:<12.2%} {:<12.2%} {:<12.2%}".format(
                model_name.upper(), acc, buy_acc, hold_acc, sell_acc
            ))
        
        logger.info("\n" + "="*80)
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        logger.info(f"🏆 Best Model: {best_model[0].upper()} ({best_model[1]['accuracy']:.2%})")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Train ML Models')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to historical data CSV')
    parser.add_argument('--lookback', type=int, default=60,
                        help='Lookback period (default: 60)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs for LSTM (default: 50)')
    parser.add_argument('--threshold', type=float, default=0.005,
                        help='Label threshold (default: 0.005 = 0.5%)')
    
    args = parser.parse_args()
    
    # Start training
    logger.info("="*80)
    logger.info("🚀 ML MODEL TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Data: {args.data}")
    logger.info(f"Lookback: {args.lookback}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("="*80)
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_path=args.data,
        lookback=args.lookback
    )
    
    # Load and prepare data
    features_df = trainer.load_and_prepare_data()
    labels = trainer.create_labels(trainer.df, threshold=args.threshold)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(
        features_df.values, labels
    )
    
    # Train all models
    results = trainer.train_all_models(
        X_train, X_val, X_test, 
        y_train, y_val, y_test
    )
    
    # Print comparison
    trainer.print_comparison(results)
    
    logger.info("\n✅ Training complete! Models saved in models/ directory")
    logger.info("\nNext steps:")
    logger.info("1. Run backtest: python backtest/backtester.py --data your_data.csv")
    logger.info("2. Start paper trading: python main.py --mode paper")


if __name__ == "__main__":
    main()


# ==================== USAGE EXAMPLES ====================
"""
# Basic training
python scripts/train_models.py --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv

# Custom parameters
python scripts/train_models.py --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv --lookback 60 --epochs 100 --threshold 0.008

# Quick test (fewer epochs)
python scripts/train_models.py --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv --epochs 10
"""