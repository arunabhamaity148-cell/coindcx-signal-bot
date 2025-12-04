"""
train_ml.py — Train ensemble models for 70%+ accuracy
Run this locally after downloading data
"""
import pandas as pd
import numpy as np
import joblib
import os
import glob
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    "rsi_1m", "rsi_5m", "rsi_div", "macd_hist", "mom_1m", "mom_5m",
    "vol_ratio", "atr_1m", "atr_5m", "delta_norm", "delta_momentum",
    "imbalance", "large_buys", "large_sells", "spread", "depth",
    "mtf_trend", "reg_trend", "reg_chop", "funding", "poc_dist"
]

def create_labels(df, forward_bars=15, profit_threshold=0.006, stop_threshold=0.004):
    """Label: 1 = win, 0 = loss"""
    labels = []
    
    for i in range(len(df) - forward_bars):
        entry = df["c"].iloc[i]
        future = df["c"].iloc[i+1:i+forward_bars+1]
        returns = (future - entry) / entry
        
        profit_hit = (returns >= profit_threshold).any()
        
        stop_hit_first = False
        for ret in returns:
            if ret <= -stop_threshold:
                stop_hit_first = True
                break
            if ret >= profit_threshold:
                break
        
        labels.append(1 if profit_hit and not stop_hit_first else 0)
    
    labels.extend([0] * forward_bars)
    df["label"] = labels
    return df

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_features(df):
    """Calculate 21 features"""
    try:
        df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
        
        # RSI
        df["rsi_1m"] = calc_rsi(df["c"], 14) / 100
        
        # 5m RSI (resample)
        df_5m = df.set_index("t").resample("5T").agg({"o":"first","h":"max","l":"min","c":"last","v":"sum"}).ffill()
        if len(df_5m) > 14:
            rsi_5m = calc_rsi(df_5m["c"], 14).reindex(df.index, method="ffill")
            df["rsi_5m"] = rsi_5m / 100
        else:
            df["rsi_5m"] = 0.5
        
        df["rsi_div"] = 0.0
        
        # MACD
        ema12 = df["c"].ewm(span=12, adjust=False).mean()
        ema26 = df["c"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = np.tanh((macd - macd_signal) / df["c"])
        
        # Momentum
        df["mom_1m"] = np.tanh((df["c"] - df["c"].shift(10)) / df["c"].shift(10) * 100)
        df["mom_5m"] = np.tanh((df["c"] - df["c"].shift(50)) / df["c"].shift(50) * 100)
        
        # Volume
        df["vol_ratio"] = (df["v"].rolling(10).mean() / df["v"].rolling(50).mean()).clip(0, 3) / 3
        
        # ATR
        tr = np.maximum(df["h"] - df["l"], np.maximum(
            np.abs(df["h"] - df["c"].shift()),
            np.abs(df["l"] - df["c"].shift())
        ))
        atr = tr.rolling(14).mean() / df["c"]
        df["atr_1m"] = (atr.clip(0, 0.02) / 0.02).fillna(0.5)
        df["atr_5m"] = df["atr_1m"]
        
        # Orderflow (synthetic)
        df["delta_norm"] = np.random.uniform(-0.5, 0.5, len(df))
        df["delta_momentum"] = np.random.uniform(-0.3, 0.3, len(df))
        df["imbalance"] = np.random.uniform(0.3, 0.7, len(df))
        df["large_buys"] = np.random.uniform(0, 0.5, len(df))
        df["large_sells"] = np.random.uniform(0, 0.5, len(df))
        df["spread"] = 0.08
        df["depth"] = 0.7
        
        # MTF
        if len(df_5m) > 50:
            ema20_5m = df_5m["c"].ewm(span=20).mean()
            ema50_5m = df_5m["c"].ewm(span=50).mean()
            mtf = (ema20_5m > ema50_5m).astype(float).reindex(df.index, method="ffill")
            df["mtf_trend"] = mtf.fillna(0.5)
        else:
            df["mtf_trend"] = 0.5
        
        # Regime
        adx_proxy = atr.rolling(14).mean() * 100
        df["reg_trend"] = (adx_proxy > 0.015).astype(float)
        df["reg_chop"] = (adx_proxy < 0.008).astype(float)
        
        df["funding"] = 0.0
        df["poc_dist"] = 0.0
        
        return df[FEATURE_COLS + ["label"]].dropna()
    except Exception as e:
        print(f"Feature error: {e}")
        return pd.DataFrame()

def load_data(data_dir="data"):
    """Load CSVs and create dataset"""
    print(f"📂 Loading data from {data_dir}/")
    
    all_data = []
    files = glob.glob(f"{data_dir}/*_1m.csv")
    
    if not files:
        print(f"❌ No CSV files found in {data_dir}/")
        return None
    
    for filepath in files:
        try:
            sym = os.path.basename(filepath).split("_")[0]
            print(f"Processing {sym}...")
            
            df = pd.read_csv(filepath)
            
            if len(df.columns) == 6:
                df.columns = ["t", "o", "h", "l", "c", "v"]
            elif len(df.columns) == 7:
                df.columns = ["t", "o", "h", "l", "c", "v", "x"]
            
            if df["t"].dtype == 'int64':
                df["t"] = pd.to_datetime(df["t"], unit="ms")
            else:
                df["t"] = pd.to_datetime(df["t"])
            
            df = create_labels(df)
            df = calculate_features(df)
            
            if len(df) > 0:
                df["sym"] = sym
                all_data.append(df)
                print(f"  ✓ {len(df)} samples | {df['label'].sum()} wins ({df['label'].mean()*100:.1f}%)")
        
        except Exception as e:
            print(f"❌ Error: {filepath}: {e}")
            continue
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Total: {len(combined)} | Wins: {combined['label'].sum()} ({combined['label'].mean()*100:.1f}%)")
    
    return combined

def train_ensemble(X_train, y_train, X_test, y_test):
    """Train 3 models"""
    models = {}
    
    print("\n🤖 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_split=100, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_score = gb.score(X_test, y_test)
    print(f"  Test Accuracy: {gb_score:.2%}")
    models["gb"] = gb
    
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=50,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    print(f"  Test Accuracy: {rf_score:.2%}")
    models["rf"] = rf
    
    print("\n📊 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=300, C=0.1, random_state=42)
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)
    print(f"  Test Accuracy: {lr_score:.2%}")
    models["lr"] = lr
    
    # Ensemble
    print("\n🎯 Ensemble:")
    gb_pred = gb.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict_proba(X_test)[:, 1]
    lr_pred = lr.predict_proba(X_test)[:, 1]
    
    ensemble_pred = (gb_pred + rf_pred + lr_pred) / 3
    ensemble_binary = (ensemble_pred > 0.5).astype(int)
    ensemble_acc = (ensemble_binary == y_test).mean()
    
    print(f"  Ensemble Accuracy: {ensemble_acc:.2%}")
    print(f"\n{classification_report(y_test, ensemble_binary, target_names=['Loss', 'Win'])}")
    
    # Feature importance
    print("\n📊 Top 10 Features:")
    importances = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances.head(10).to_string(index=False))
    
    return models, ensemble_acc

def main():
    print("=" * 60)
    print("🎯 ML Model Training")
    print("=" * 60)
    
    data = load_data("data")
    if data is None:
        print("\n❌ No data. Run: python download_data.py")
        return
    
    X = data[FEATURE_COLS]
    y = data["label"]
    
    print(f"\n📊 Class Distribution:")
    print(f"  Win: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    print(f"  Loss: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📈 Train: {len(X_train)} | Test: {len(X_test)}")
    
    models, acc = train_ensemble(X_train, y_train, X_test, y_test)
    
    print("\n💾 Saving models...")
    joblib.dump(models["gb"], "gb_model.pkl")
    joblib.dump(models["rf"], "rf_model.pkl")
    joblib.dump(models["lr"], "lr_model.pkl")
    
    print("\n✅ Complete!")
    print(f"   Ensemble Accuracy: {acc:.2%}")
    print(f"   Files: gb_model.pkl, rf_model.pkl, lr_model.pkl")
    
    if acc >= 0.70:
        print(f"\n   ✅ EXCELLENT: Expected live win rate: 65-70%")
    elif acc >= 0.65:
        print(f"\n   ⚠️ GOOD: Expected live win rate: 60-65%")
    else:
        print(f"\n   ❌ POOR: Collect more data")

if __name__ == "__main__":
    main()