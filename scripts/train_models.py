#!/usr/bin/env python3
"""
ML Model Training Script (improved)

- Accepts either --data (single CSV file) or --data_dir (directory containing many CSVs)
- Prepares features, creates sequences for LSTM, trains LSTM/XGBoost/RandomForest ensemble
- Saves models to ./models/
"""
import sys
import os
from glob import glob
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# adjust path for project imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Placeholders for your model classes (import your real implementations)
from ml.lstm_model import LSTMModel
from ml.xgboost_model import XGBoostModel
from ml.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_models")


class ModelTrainer:
    def __init__(self, data_paths, lookback=60, features_count=None):
        """
        data_paths: list of csv file paths (already validated non-empty)
        lookback: number of timesteps for LSTM sequences
        """
        self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]
        self.lookback = lookback
        self.df = None
        self.features_df = None
        self.scaler = None

    def load_and_concat(self):
        """Load CSVs, concat by datetime index, sort and dedupe"""
        dfs = []
        for p in self.data_paths:
            logger.info(f"Loading {p} ...")
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        self.df = df
        logger.info(f"✅ Loaded total {len(self.df):,} rows from {len(self.data_paths)} file(s)")
        logger.info(f"Date range: {self.df.index[0]} -> {self.df.index[-1]}")
        return self.df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw OHLCV DataFrame"""
        features = pd.DataFrame(index=df.index)

        # Basic columns (expecting columns named close, high, low, volume)
        features['close'] = df['close']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume']

        # Returns & log returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages and EMAs
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Volatility
        features['volatility_10'] = df['close'].rolling(10).std()
        features['volatility_20'] = df['close'].rolling(20).std()
        features['volatility_50'] = df['close'].rolling(50).std()

        # RSI simple implementation for multiple periods
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.clip(lower=0).rolling(window=period).mean()
            loss = (-delta.clip(upper=0)).rolling(window=period).mean()
            rs = gain / (loss.replace(0, np.nan))
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # Bollinger bands
        for period in [20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + (2 * std)
            features[f'bb_lower_{period}'] = sma - (2 * std)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / sma

        # ATR (simplified)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()

        # Volume
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']

        # OBV (on-balance volume)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        features['obv'] = obv

        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)

        # Final clean: use bfill then ffill then zero
        features = features.bfill().ffill().fillna(0)

        return features

    def prepare(self):
        """Load, feature-engineer and store features"""
        if self.df is None:
            self.load_and_concat()
        self.features_df = self._create_features(self.df)
        logger.info(f"✅ Prepared features: {self.features_df.shape[1]} columns, {len(self.features_df):,} rows")
        return self.features_df

    def create_labels(self, df: pd.DataFrame, threshold=0.005):
        """
        Create labels using future return between next close and current close:
        future_return = (close_next / close_now) - 1
        Labels:
          0 = BUY (price will drop -> good for LONG entry)  <-- NOTE: your doc had reversed meaning; here we keep conventional:
          0 = BUY (future_return > threshold)  [price rises -> buy]
          1 = HOLD
          2 = SELL (future_return < -threshold)
        (But you can flip mapping to match your main bot's convention.)
        """
        logger.info(f"Creating labels with threshold {threshold:.4f}")
        future_close = df['close'].shift(-1)
        future_return = (future_close / df['close']) - 1.0

        def label_fn(x):
            if pd.isna(x):
                return 1
            if x > threshold:
                return 0  # BUY
            elif x < -threshold:
                return 2  # SELL
            else:
                return 1  # HOLD

        labels = future_return.apply(label_fn).astype(int)
        # One-hot
        labels_onehot = pd.get_dummies(labels).reindex(columns=[0, 1, 2], fill_value=0).values.astype(np.float32)

        # Log distribution
        unique, counts = np.unique(labels.values, return_counts=True)
        logger.info("Label distribution:")
        for u, c in zip(unique, counts):
            name = {0: "BUY", 1: "HOLD", 2: "SELL"}.get(u, str(u))
            logger.info(f"  {name} ({u}): {c:,} ({c/len(labels)*100:.2f}%)")

        return labels_onehot, labels.values

    def create_sequences(self, features: pd.DataFrame, labels_array: np.ndarray):
        """
        Create sequences for LSTM:
          X_seq shape: (n_samples, lookback, n_features)
          y_seq shape: (n_samples, n_classes)
        We align y at the end of each sequence (the label corresponding to timestep i+lookback)
        """
        X = features.values
        n_samples = len(X) - self.lookback
        if n_samples <= 0:
            raise ValueError("Not enough data to create sequences with the requested lookback")

        n_features = X.shape[1]
        # If labels_array is 1D (class ints), convert to indices; but we expect one-hot later.
        # We'll align both one-hot and int labels outside.
        X_seq = np.zeros((n_samples, self.lookback, n_features), dtype=np.float32)
        for i in range(n_samples):
            X_seq[i] = X[i:i + self.lookback]

        # align labels: use label at position i+lookback (we expect one-hot in full_labels)
        # So caller should pass full_labels_onehot (len = len(features))
        return X_seq

    def scale_sequences(self, X_train, X_val, X_test):
        """Fit scaler on flattened X_train then transform all and reshape back"""
        # Flatten train to 2D: (samples*timesteps, features)
        s_train = X_train.reshape(-1, X_train.shape[-1])
        self.scaler = StandardScaler()
        logger.info("Fitting scaler on training data...")
        self.scaler.fit(s_train)

        def transform_array(X):
            shape = X.shape
            X2 = X.reshape(-1, shape[-1])
            X2 = self.scaler.transform(X2)
            return X2.reshape(shape)

        X_train_s = transform_array(X_train)
        X_val_s = transform_array(X_val)
        X_test_s = transform_array(X_test)
        return X_train_s, X_val_s, X_test_s

    def split_and_prepare_sequences(self, features_df, labels_onehot_full, labels_int_full, train_ratio=0.70, val_ratio=0.15):
        """
        Create sequences and split into train/val/test.
        labels_onehot_full length == len(features_df)
        """
        # create sequences (X) and aligned labels (y)
        X_seq = self.create_sequences(features_df, labels_int_full)  # shape (N, lookback, n_features)
        # aligned labels start at index lookback .. end-1
        y_idx_start = self.lookback
        y_onehot = labels_onehot_full[y_idx_start: y_idx_start + X_seq.shape[0]]
        y_int = labels_int_full[y_idx_start: y_idx_start + X_seq.shape[0]]

        total = X_seq.shape[0]
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        X_train = X_seq[:train_size]
        X_val = X_seq[train_size: train_size + val_size]
        X_test = X_seq[train_size + val_size:]

        y_train = y_onehot[:train_size]
        y_val = y_onehot[train_size: train_size + val_size]
        y_test = y_onehot[train_size + val_size:]

        logger.info(f"Sequences: total {total:,}, train {len(X_train):,}, val {len(X_val):,}, test {len(X_test):,}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def ensure_models_dir(self):
        os.makedirs("models", exist_ok=True)

    def train_all(self, X_train, X_val, X_test, y_train, y_val, y_test, epochs=50):
        """Train and save LSTM, XGBoost and an ensemble RF"""
        self.ensure_models_dir()

        # Train LSTM
        logger.info("Training LSTM model...")
        n_features = X_train.shape[-1]
        lstm = LSTMModel(lookback=self.lookback, features=n_features)
        lstm.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=32)
        lstm.save(os.path.join("models", "lstm_model.h5"))
        lstm_res = lstm.evaluate(X_test, y_test)

        # For tree-based models, flatten sequences to feature vector (e.g., last timestep features or averages)
        # Here we use last timestep features as simple approach
        def flatten_for_tree(X):
            # X shape (samples, lookback, features) -> take last timestep features
            return X[:, -1, :]

        X_train_flat = flatten_for_tree(X_train)
        X_val_flat = flatten_for_tree(X_val)
        X_test_flat = flatten_for_tree(X_test)

        # Train XGBoost
        logger.info("Training XGBoost model...")
        xgb = XGBoostModel(n_estimators=200, max_depth=8, learning_rate=0.1)
        xgb.train(X_train_flat, y_train, X_val_flat, y_val)
        xgb.save(os.path.join("models", "xgboost_model.pkl"))
        xgb_res = xgb.evaluate(X_test_flat, y_test)

        # Ensemble (train RF)
        logger.info("Training ensemble RandomForest...")
        ensemble = EnsembleModel(lstm_model=lstm, xgboost_model=xgb, lstm_weight=0.5, xgb_weight=0.3, rf_weight=0.2)
        ensemble.train_rf(X_train_flat, y_train)
        ensemble.save_rf(os.path.join("models", "random_forest.pkl"))
        ensemble_res = ensemble.evaluate(X_test_flat, y_test)

        logger.info("✅ Training complete. Models saved to ./models/")
        return {"lstm": lstm_res, "xgboost": xgb_res, "ensemble": ensemble_res}


def build_file_list(data_arg, data_dir_arg):
    """Return list of CSV paths based on either --data or --data_dir"""
    if data_arg:
        if not os.path.isfile(data_arg):
            raise FileNotFoundError(f"--data file not found: {data_arg}")
        return [data_arg]
    if data_dir_arg:
        if not os.path.isdir(data_dir_arg):
            raise FileNotFoundError(f"--data_dir not found: {data_dir_arg}")
        csvs = sorted(glob(os.path.join(data_dir_arg.rstrip("/"), "BTCUSDT-15m-*.csv")))
        if len(csvs) == 0:
            # fallback: any csv in dir
            csvs = sorted(glob(os.path.join(data_dir_arg.rstrip("/"), "*.csv")))
        if len(csvs) == 0:
            raise FileNotFoundError(f"No CSV files found in {data_dir_arg}")
        return csvs
    raise ValueError("Either --data or --data_dir must be provided.")


def main():
    parser = argparse.ArgumentParser(description="Train ML models (LSTM / XGBoost / RF ensemble)")
    parser.add_argument("--data", type=str, default=None, help="Single CSV file path")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing CSV files (preferred for multi-file)")
    parser.add_argument("--lookback", type=int, default=60, help="LSTM lookback length")
    parser.add_argument("--epochs", type=int, default=50, help="LSTM epochs")
    parser.add_argument("--threshold", type=float, default=0.005, help="Label threshold")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train split fraction")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split fraction")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ML TRAINING PIPELINE START")
    logger.info(f"args: data={args.data}, data_dir={args.data_dir}, lookback={args.lookback}, epochs={args.epochs}")
    logger.info("=" * 80)

    files = build_file_list(args.data, args.data_dir)
    trainer = ModelTrainer(files, lookback=args.lookback)

    # load and prepare
    trainer.load_and_concat()
    features_df = trainer.prepare()

    # create labels (returns-based)
    labels_onehot, labels_int = trainer.create_labels(trainer.df, threshold=args.threshold)

    # sequences + splits
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_and_prepare_sequences(
        features_df, labels_onehot, labels_int, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    # scale sequences
    X_train_s, X_val_s, X_test_s = trainer.scale_sequences(X_train, X_val, X_test)

    # train models
    results = trainer.train_all(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, epochs=args.epochs)

    # summary
    logger.info("Training results summary:")
    for name, res in results.items():
        logger.info(f" - {name}: {res}")

    logger.info("DONE.")


if __name__ == "__main__":
    main()