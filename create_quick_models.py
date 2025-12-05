import joblib, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def make():
    X, y = make_classification(n_samples=3000, n_features=20, n_informative=10, n_redundant=5, weights=[0.6,0.4], random_state=42)
    gb = GradientBoostingClassifier(n_estimators=60, max_depth=3, learning_rate=0.1, random_state=42).fit(X, y)
    rf = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42).fit(X, y)
    lr = LogisticRegression(max_iter=200, C=1.0, random_state=42).fit(X, y)
    joblib.dump(gb, "gb_model.pkl")
    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(lr, "lr_model.pkl")
    print("✅ Quick models saved.")

if __name__ == "__main__":
    make()
