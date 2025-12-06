# ================================================================
# ml_filter.py - Machine Learning Signal Filter
# ================================================================

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import pickle
from datetime import datetime

log = logging.getLogger("ml_filter")


class MLSignalFilter:
    """
    Machine Learning filter to predict signal success probability
    
    Uses simple logistic regression or random forest to classify
    signals as likely profitable or not.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.trained = False
        
        # Feature importance
        self.feature_importance = {}
    
    def extract_features(self, signal: dict) -> np.ndarray:
        """
        Extract ML features from signal
        
        Features:
            - Confidence score
            - Logic score
            - BTC correlation
            - RSI
            - Volume metrics
            - Smart money direction
            - Risk/reward ratio
            - Time of day
            - Day of week
        """
        
        features = []
        
        # Basic signal features
        features.append(signal.get("confidence", 0))
        features.append(signal.get("score", 0))
        
        # Correlation
        corr_data = signal.get("correlation", {})
        features.append(corr_data.get("price_corr", 0))
        
        # Volume features
        volume = signal.get("volume", {})
        if volume and volume.get("smart_money"):
            sm = volume["smart_money"]
            features.append(1 if sm["smart_money_direction"] == "BUY" else -1 if sm["smart_money_direction"] == "SELL" else 0)
            features.append(sm["large_buys"])
            features.append(sm["large_sells"])
        else:
            features.extend([0, 0, 0])
        
        # Price levels
        levels = signal.get("levels", {})
        if levels and levels.get("analysis"):
            analysis = levels["analysis"]
            features.append(analysis.get("risk_reward_ratio", 0))
        else:
            features.append(0)
        
        # Time features
        timestamp = signal.get("timestamp", datetime.utcnow())
        features.append(timestamp.hour)  # Hour of day (0-23)
        features.append(timestamp.weekday())  # Day of week (0-6)
        
        # Strategy type (one-hot encoded)
        strat = signal.get("strategy", "")
        features.append(1 if strat == "QUICK" else 0)
        features.append(1 if strat == "MID" else 0)
        features.append(1 if strat == "TREND" else 0)
        
        return np.array(features)
    
    def prepare_training_data(
        self,
        signals: List[dict],
        outcomes: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from signals and their outcomes
        
        Args:
            signals: List of signal dicts
            outcomes: List of booleans (True = profitable, False = loss)
            
        Returns:
            (X, y) arrays
        """
        
        X = []
        y = []
        
        for signal, outcome in zip(signals, outcomes):
            try:
                features = self.extract_features(signal)
                X.append(features)
                y.append(1 if outcome else 0)
            except Exception as e:
                log.warning(f"Failed to extract features: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        log.info(f"Prepared {len(X)} training samples")
        log.info(f"Positive samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        
        return X, y
    
    def train_logistic_regression(
        self,
        signals: List[dict],
        outcomes: List[bool]
    ):
        """
        Train logistic regression model
        
        Simple but interpretable model
        """
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
        except ImportError:
            log.error("scikit-learn not installed. Run: pip install scikit-learn")
            return
        
        # Prepare data
        X, y = self.prepare_training_data(signals, outcomes)
        
        if len(X) < 20:
            log.warning("Not enough training data (need at least 20 samples)")
            return
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        self.model.fit(X_scaled, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
        log.info(f"✅ Logistic Regression trained")
        log.info(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Feature importance (coefficient magnitudes)
        self.feature_names = [
            "confidence", "score", "btc_corr", "sm_direction",
            "large_buys", "large_sells", "risk_reward",
            "hour", "weekday", "is_quick", "is_mid", "is_trend"
        ]
        
        coef = np.abs(self.model.coef_[0])
        self.feature_importance = dict(zip(self.feature_names, coef))
        
        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        log.info("Top 5 important features:")
        for name, importance in sorted_features[:5]:
            log.info(f"  {name}: {importance:.3f}")
        
        self.trained = True
    
    def train_random_forest(
        self,
        signals: List[dict],
        outcomes: List[bool]
    ):
        """
        Train random forest model
        
        More powerful but less interpretable
        """
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
        except ImportError:
            log.error("scikit-learn not installed")
            return
        
        # Prepare data
        X, y = self.prepare_training_data(signals, outcomes)
        
        if len(X) < 20:
            log.warning("Not enough training data")
            return
        
        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
        log.info(f"✅ Random Forest trained")
        log.info(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Feature importance
        self.feature_names = [
            "confidence", "score", "btc_corr", "sm_direction",
            "large_buys", "large_sells", "risk_reward",
            "hour", "weekday", "is_quick", "is_mid", "is_trend"
        ]
        
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_names, importance))
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        log.info("Top 5 important features:")
        for name, importance in sorted_features[:5]:
            log.info(f"  {name}: {importance:.3f}")
        
        self.trained = True
    
    def predict_success_probability(self, signal: dict) -> float:
        """
        Predict probability that signal will be profitable
        
        Returns:
            Probability between 0 and 1
        """
        
        if not self.trained or self.model is None:
            log.warning("Model not trained yet")
            return 0.5  # Neutral
        
        try:
            features = self.extract_features(signal)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict probability
            prob = self.model.predict_proba(features_scaled)[0][1]
            
            return float(prob)
        
        except Exception as e:
            log.error(f"Prediction error: {e}")
            return 0.5
    
    def enhance_signal_with_ml(self, signal: dict) -> dict:
        """
        Add ML predictions to signal
        
        Adds:
            - ml_probability: Success probability (0-1)
            - ml_confidence_adj: Adjustment to confidence (-20 to +20)
            - ml_recommendation: "TAKE" or "SKIP"
        """
        
        if not self.trained:
            return signal
        
        prob = self.predict_success_probability(signal)
        
        # Confidence adjustment based on ML probability
        # prob > 0.7: boost confidence
        # prob < 0.3: reduce confidence
        if prob > 0.7:
            confidence_adj = (prob - 0.5) * 40  # Up to +20
        elif prob < 0.3:
            confidence_adj = (prob - 0.5) * 40  # Down to -20
        else:
            confidence_adj = 0
        
        # Recommendation
        recommendation = "TAKE" if prob > 0.55 else "SKIP"
        
        signal["ml_probability"] = round(prob, 3)
        signal["ml_confidence_adj"] = round(confidence_adj, 1)
        signal["ml_recommendation"] = recommendation
        
        # Adjust overall confidence
        original_confidence = signal.get("confidence", 0)
        signal["confidence"] = max(0, min(100, original_confidence + confidence_adj))
        
        return signal
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.trained:
            log.warning("No trained model to save")
            return
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        log.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            self.feature_importance = model_data["feature_importance"]
            self.trained = True
            
            log.info(f"Model loaded from {filepath}")
        
        except Exception as e:
            log.error(f"Failed to load model: {e}")


# Convenience function
async def train_ml_filter_from_history(
    closed_trades: List[dict],
    min_samples: int = 50
) -> Optional[MLSignalFilter]:
    """
    Train ML filter from closed trades history
    
    Args:
        closed_trades: List of closed trade dicts with signal_data
        min_samples: Minimum number of trades needed
        
    Returns:
        Trained MLSignalFilter or None
    """
    
    if len(closed_trades) < min_samples:
        log.warning(f"Not enough trades ({len(closed_trades)}) for ML training")
        return None
    
    # Extract signals and outcomes
    signals = []
    outcomes = []
    
    for trade in closed_trades:
        if "signal_data" in trade:
            signals.append(trade["signal_data"])
            outcomes.append(trade["pnl"] > 0)
    
    if len(signals) < min_samples:
        log.warning("Not enough valid signal data")
        return None
    
    # Train filter
    ml_filter = MLSignalFilter()
    ml_filter.train_random_forest(signals, outcomes)
    
    return ml_filter