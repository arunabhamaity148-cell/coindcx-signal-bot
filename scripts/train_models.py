#!/usr/bin/env python3
"""
scripts/train_models.py

Final robust training script.

Features:
- Accepts either --data (single CSV) or --data_dir (directory with many CSVs)
- Robust CSV loading (handles epoch timestamps in ms/s, common column name variants)
- Concatenates, deduplicates, sorts by datetime index
- Feature engineering (many technical features)
- Creates sequences for LSTM (lookback) and aligns labels
- Scales using StandardScaler fitted on training data
- Trains LSTM (keras), XGBoost (tree), and RandomForest as ensemble component
- Saves models to ./models/
- Designed to be used in Docker builds (small epochs) or offline/full training
"""
import sys
import os
from glob import glob
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ensure project imports work (adjust path as necessary)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model wrappers (these must exist in the repo)
# If signatures differ, adapt LSTMModel/XGBoostModel/EnsembleModel wrappers accordingly.
from ml.lstm_model import LSTMModel
from ml.xgboost_model import XGBoostModel
from ml.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_models")


class ModelTrainer:
    def __init__(self, data_paths, lookback=60):
        """
        data_paths: list of csv file paths
        lookback: number of timesteps for LSTM sequences
        """
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.data_paths = data_paths
        self.lookback = lookback
        self.df = None
        self.features_df = None
        self.scaler = None

    def load_and_concat(self):
        """
        Robust loader for many CSV formats:
        - tries to parse a timestamp column (common names)
        - handles epoch in ms or s
        - normalizes column names to open/high/low/close/volume
        - concatenates files, dedupes by index, sorts
        """
        dfs = []
        for p in self.data_paths:
            logger.info(f"Loading {p} ...")
            tmp = pd.read_csv(p, low_memory=False)
            # candidate timestamp columns
            ts_candidates = [c for c in tmp.columns if c.lower() in ("timestamp", "time", "date", "datetime", "open_time", "start_time")]
            if len(ts_candidates) == 0:
                # fallback: first column may be timestamp-like
                ts_candidates = [tmp.columns[0]]

            ts_col = ts_candidates[0]

            # attempt conversion to datetime
            tmp['___dt'] = pd.NaT
            try:
                col = tmp[ts_col]
                if np.issubdtype(col.dtype, np.number):
                    sample = int(col.iloc[0]) if len(col) > 0 and not pd.isna(col.iloc[0]) else 0
                    # heuristics: ms > 1e12, s > 1e9
                    if sample > 1e12:
                        tmp['___dt'] = pd.to_datetime(col, unit='ms', origin='unix', errors='coerce')
                    elif sample > 1e9:
                        tmp['___dt'] = pd.to_datetime(col, unit='s', origin='unix', errors='coerce')
                    else:
                        # small numbers, try as seconds
                        tmp['___dt'] = pd.to_datetime(col, unit='s', origin='unix', errors='coerce')
                else:
                    tmp['___dt'] = pd.to_datetime(col, errors='coerce')
            except Exception:
                tmp['___dt'] = pd.to_datetime(tmp.iloc[:, 0], errors='coerce')

            # If parsing failed, try parsing first column directly
            if tmp['___dt'].isna().all():
                try:
                    tmp['___dt'] = pd.to_datetime(tmp.iloc[:, 0], errors='coerce')
                except Exception:
                    pass

            # If still all NaT, try pandas' parse on the whole frame (rare)
            if tmp['___dt'].isna().all():
                try:
                    tmp.index = pd.to_datetime(tmp.index, errors='coerce')
                    tmp['___dt'] = tmp.index
                except Exception:
                    pass

            # set index
            tmp = tmp.set_index('___dt')
            tmp.index.name = None

            # normalize column names
            colmap = {}
            for col in tmp.columns:
                lc = col.lower()
                if lc in ('open', 'open_price'):
                    colmap[col] = 'open'
                elif lc in ('high', 'high_price'):
                    colmap[col] = 'high'
                elif lc in ('low', 'low_price'):
                    colmap[col] = 'low'
                elif lc in ('close', 'close_price', 'last', 'trade_price'):
                    colmap[col] = 'close'
                elif 'volume' in lc or lc == 'vol':
                    colmap[col] = 'volume'
                elif lc in ('o', 'h', 'l', 'c'):
                    # single-letter columns map if present
                    if lc == 'o':
                        colmap[col] = 'open'
                    elif lc == 'h':
                        colmap[col] = 'high'
                    elif lc == 'l':
                        colmap[col] = 'low'
                    elif lc == 'c':
                        colmap[col] = 'close'
            tmp = tmp.rename(columns=colmap)

            # If 'close' missing but 'price' exists, map it
            if 'close' not in tmp.columns and 'price' in tmp.columns:
                tmp = tmp.rename(columns={'price': 'close'})

            # Ensure required columns exist
            for needed in ['open', 'high', 'low', 'close', 'volume']:
                if needed not in tmp.columns:
                    tmp[needed] = np.nan

            # Keep only these columns (order)
            tmp = tmp[['open', 'high', 'low', 'close', 'volume']]

            dfs.append(tmp)

        if len(dfs) == 0:
            raise FileNotFoundError("No CSV files loaded (empty file list).")

        df = pd.concat(dfs, axis=0)
        # drop rows with NaT index
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        self.df = df
        logger.info(f"✅ Loaded total {len(self.df):,} rows from {len(self.data_paths)} file(s)")
        if len(self.df) > 0:
            logger.info(f"Date range: {self.df.index[0]} -> {self.df.index[-1]}")
        return self.df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from OHLCV DataFrame"""
        features = pd.DataFrame(index=df.index)

        features['close'] = df['close']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume']

        # returns & log returns
        features['returns'] = df['close'].pct_change().fillna(0)
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

        # moving averages & EMAs
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # volatility
        features['volatility_10'] = df['close'].rolling(10).std()
        features['volatility_20'] = df['close'].rolling(20).std()
        features['volatility_50'] = df['close'].rolling(50).std()

        # RSI
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

        # Bollinger
        for period in [20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + (2 * std)
            features[f'bb_lower_{period}'] = sma - (2 * std)
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / (sma.replace(0, np.nan))

        # ATR simplified
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()

        # Volume features
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']

        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        features['obv'] = obv

        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)

        # Final fill
        features = features.bfill().ffill().fillna(0)
        return features

    def prepare(self):
        """Load data and prepare feature matrix"""
        if self.df is None:
            self.load_and_concat()
        self.features_df = self._create_features(self.df)
        logger.info(f"Prepared features: {self.features_df.shape[1]} columns, {len(self.features_df):,} rows")
        return self.features_df

    def create_labels(self, threshold=0.005):
        """
        Create labels from future returns:
        0 = BUY (future_return > threshold)
        1 = HOLD
        2 = SELL (future_return < -threshold)
        Returns: (onehot array, int labels)
        """
        future_close = self.df['close'].shift(-1)
        future_return = (future_close / self.df['close']) - 1.0

        def lbl_fn(v):
            if pd.isna(v):
                return 1
            if v > threshold:
                return 0
            elif v < -threshold:
                return 2
            else:
                return 1

        labels_int = future_return.apply(lbl_fn).astype(int).values
        # one-hot
        labels_df = pd.get_dummies(labels_int).reindex(columns=[0, 1, 2], fill_value=0)
        labels_onehot = labels_df.values.astype(np.float32)

        # log distribution
        unique, counts = np.unique(labels_int, return_counts=True)
        logger.info("Label distribution:")
        for u, c in zip(unique, counts):
            name = {0: "BUY", 1: "HOLD", 2: "SELL"}.get(u, str(u))
            logger.info(f"  {name} ({u}): {c:,} ({c / len(labels_int) * 100:.2f}%)")

        return labels_onehot, labels_int

    def create_sequences(self, features: pd.DataFrame):
        """Create sequences for LSTM from features DataFrame"""
        X = features.values.astype(np.float32)
        n_samples = len(X) - self.lookback
        if n_samples <= 0:
            raise ValueError("Not enough data to create sequences with requested lookback")
        n_features = X.shape[1]
        X_seq = np.zeros((n_samples, self.lookback, n_features), dtype=np.float32)
        for i in range(n_samples):
            X_seq[i] = X[i:i + self.lookback]
        return X_seq

    def split_and_prepare_sequences(self, labels_onehot, labels_int, train_ratio=0.70, val_ratio=0.15):
        """
        Create sequences, align labels and split into train/val/test.
        labels_onehot/int length == len(features_df)
        """
        X_seq = self.create_sequences(self.features_df)
        y_start = self.lookback
        y_onehot = labels_onehot[y_start: y_start + X_seq.shape[0]]
        y_int = labels_int[y_start: y_start + X_seq.shape[0]]

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

    def scale_sequences(self, X_train, X_val, X_test):
        """Fit scaler on flattened training sequences and transform all"""
        s_train = X_train.reshape(-1, X_train.shape[-1])
        self.scaler = StandardScaler()
        logger.info("Fitting StandardScaler on training data...")
        self.scaler.fit(s_train)

        def transform_arr(X):
            shape = X.shape
            X2 = X.reshape(-1, shape[-1])
            X2 = self.scaler.transform(X2)
            return X2.reshape(shape)

        return transform_arr(X_train), transform_arr(X_val), transform_arr(X_test)

    def ensure_models_dir(self):
        os.makedirs("models", exist_ok=True)

    def train_all(self, X_train, X_val, X_test, y_train, y_val, y_test, epochs=50):
        """Train LSTM, XGBoost, and RF (ensemble) and save models"""
        self.ensure_models_dir()

        # 1) LSTM
        logger.info("Training LSTM model...")
        n_features = X_train.shape[-1]
        lstm = LSTMModel(lookback=self.lookback, features=n_features)
        lstm.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=32)
        lstm.save(os.path.join("models", "lstm_model.h5"))
        lstm_res = lstm.evaluate(X_test, y_test)

        # 2) Flatten last timestep for tree models
        def flatten_last(X):
            return X[:, -1, :]

        X_train_flat = flatten_last(X_train)
        X_val_flat = flatten_last(X_val)
        X_test_flat = flatten_last(X_test)

        # 3) XGBoost
        logger.info("Training XGBoost model...")
        xgb = XGBoostModel(n_estimators=200, max_depth=8, learning_rate=0.1)
        xgb.train(X_train_flat, y_train, X_val_flat, y_val)
        xgb.save(os.path.join("models", "xgboost_model.pkl"))
        xgb_res = xgb.evaluate(X_test_flat, y_test)

        # 4) Ensemble RF
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
        pattern = os.path.join(data_dir_arg.rstrip("/"), "BTCUSDT-15m-*.csv")
        csvs = sorted(glob(pattern))
        if len(csvs) == 0:
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
    logger.info(f"Using files: {files}")
    trainer = ModelTrainer(files, lookback=args.lookback)

    # load and prepare
    trainer.load_and_concat()
    features_df = trainer.prepare()

    # create labels
    labels_onehot, labels_int = trainer.create_labels(threshold=args.threshold)

    # sequences + split
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_and_prepare_sequences(
        labels_onehot, labels_int, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    # scale
    X_train_s, X_val_s, X_test_s = trainer.scale_sequences(X_train, X_val, X_test)

    # train
    results = trainer.train_all(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, epochs=args.epochs)

    logger.info("Training results summary:")
    for name, res in results.items():
        logger.info(f" - {name}: {res}")

    logger.info("DONE.")


if __name__ == "__main__":
    main()
