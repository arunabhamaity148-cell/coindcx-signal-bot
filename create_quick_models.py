# create_quick_models.py
"""
Fast small-model generator for mobile/Railway.
This trains three lightweight sklearn models on synthetic data
and saves gb_model.pkl, rf_model.pkl, lr_model.pkl in repo root.
This is NOT a substitute for real trained models on historical market data,
but it makes helpers.load_ensemble() find models so ML-path runs.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

MODEL_FILES = ["gb_model.pkl", "rf_model.pkl", "lr_model.pkl"]

def simple_dataset(n_samples=5000, n_features=20, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],
        flip_y=0.03,
        random_state=random_state
    )
    return X, y

def train_and_save():
    print(">> Creating synthetic dataset...")
    X, y = simple_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(">> Training GradientBoosting (fast config)...")
    gb = GradientBoostingClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.05, random_state=42
    )
    gb.fit(X_train, y_train)
    print(f"   GB score: {gb.score(X_test, y_test):.3f}")

    print(">> Training RandomForest (fast config)...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    print(f"   RF score: {rf.score(X_test, y_test):.3f}")

    print(">> Training LogisticRegression (fast config)...")
    lr = LogisticRegression(max_iter=200, C=0.5, random_state=42)
    lr.fit(X_train, y_train)
    print(f"   LR score: {lr.score(X_test, y_test):.3f}")

    for name, model in [
        ("gb_model.pkl", gb),
        ("rf_model.pkl", rf),
        ("lr_model.pkl", lr),
    ]:
        joblib.dump(model, name)
        print(f">> Saved {name}")

    print(">> All 3 models created successfully.")

if __name__ == "__main__":
    exists = all(os.path.exists(f) for f in MODEL_FILES)
    if exists:
        print("Models already exist. Delete them if you want to regenerate.")
    else:
        train_and_save()