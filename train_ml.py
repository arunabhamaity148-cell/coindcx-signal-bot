import pandas as pd, numpy as np, joblib, os, glob
from sklearn.ensemble import GradientBoostingClassifier

def label(df):
    df["ret"] = df["c"].shift(-60) / df["c"] - 1
    df["label"] = ((df["ret"] > 0.008) & (df["ret"] > df["ret"].shift(-60) + 0.005)).astype(int)
def feat(df):
    df["rsi"] = df["c"].diff().agg(lambda x: 100 - 100/(1+x.clip(lower=0).rolling(14).mean()/x.clip(upper=0).abs().rolling(14).mean()))
    macd = df["c"].ewm(12).mean() - df["c"].ewm(26).mean()
    df["macd_slope"] = np.tanh(macd - macd.ewm(9).mean())
    df["imb"] = np.random.uniform(-1,1,len(df))  # replace with real
    df["delta_norm"] = np.tanh(np.random.uniform(-1e6,1e6,len(df))/1e6)
    df["sweep"] = np.random.randint(0,2,len(df))
    df["spread_atr"] = 0.02 / 0.5
    df["depth_norm"] = np.tanh(np.random.uniform(1e6,5e6,len(df))/5e6)
    df["btc_1m"] = 0.3
    df["reg_trend"] = (df["rsi"] > 50).astype(int)
    df["reg_chop"] = (df["rsi"] < 30).astype(int)
    return df[["rsi","macd_slope","imb","delta_norm","sweep","spread_atr","depth_norm","btc_1m","reg_trend","reg_chop","label"]].dropna()

dfs = []
for f in glob.glob("data/*_1m.csv"):
    df = pd.read_csv(f, names=["t","o","h","l","c","v"])
    df["sym"] = f.split("/")[1].split("_")[0]; label(df); dfs.append(feat(df))
Xy = pd.concat(dfs)
X, y = Xy.drop("label", axis=1), Xy["label"]
clf = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05)
clf.fit(X, y)
joblib.dump(clf, "ml_model.pkl")
print("model saved acc", clf.score(X, y))
