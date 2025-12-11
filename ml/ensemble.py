"""
Ensemble Model - Combines LSTM + XGBoost + Random Forest
Final prediction with weighted voting
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModel:
    def __init__(self, lstm_model, xgboost_model, 
                 lstm_weight=0.50, xgb_weight=0.30, rf_weight=0.20,
                 model_path='models/'):
        self.lstm_model = lstm_model
        self.xgboost_model = xgboost_model
        self.rf_model = None
        
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight
        self.rf_weight = rf_weight
        
        self.model_path = model_path
        self.rf_scaler = StandardScaler()
        
        # Validate weights
        total_weight = lstm_weight + xgb_weight + rf_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"⚠️ Weights don't sum to 1.0: {total_weight}")
    
    def build_rf_model(self, n_estimators=100, max_depth=10):
        """Build Random Forest model"""
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        logger.info("✅ Random Forest model built")
        return self.rf_model
    
    def train_rf(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("🚀 Training Random Forest...")
        
        # Convert one-hot to labels if needed
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
        
        # Scale features
        X_train_scaled = self.rf_scaler.fit_transform(X_train)
        
        # Build if not exists
        if self.rf_model is None:
            self.build_rf_model()
        
        # Train
        self.rf_model.fit(X_train_scaled, y_train)
        
        logger.info("✅ Random Forest training completed")
        
        # Save scaler
        joblib.dump(self.rf_scaler, f'{self.model_path}rf_scaler.pkl')
        
        return self.rf_model
    
    def predict_proba(self, X):
        """
        Get ensemble prediction probabilities
        Combines predictions from all 3 models
        """
        # Get predictions from each model
        lstm_proba = self.lstm_model.predict_proba(X)
        xgb_proba = self.xgboost_model.predict_proba(X)
        
        # RF prediction
        X_scaled = self.rf_scaler.transform(X)
        rf_proba = self.rf_model.predict_proba(X_scaled)
        
        # Weighted ensemble
        ensemble_proba = (
            self.lstm_weight * lstm_proba +
            self.xgb_weight * xgb_proba +
            self.rf_weight * rf_proba
        )
        
        return ensemble_proba
    
    def predict(self, X):
        """Make predictions"""
        return self.predict_proba(X)
    
    def predict_classes(self, X):
        """Get predicted classes"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_confidence(self, X):
        """Get confidence scores"""
        proba = self.predict_proba(X)
        confidence = np.max(proba, axis=1)
        return confidence
    
    def get_individual_predictions(self, X):
        """Get predictions from each model separately"""
        lstm_proba = self.lstm_model.predict_proba(X)
        xgb_proba = self.xgboost_model.predict_proba(X)
        X_scaled = self.rf_scaler.transform(X)
        rf_proba = self.rf_model.predict_proba(X_scaled)
        
        return {
            'lstm': lstm_proba,
            'xgboost': xgb_proba,
            'random_forest': rf_proba
        }
    
    def generate_signal(self, X, confidence_threshold=0.70):
        """
        Generate trading signal with confidence check
        Returns: signal (0=BUY, 1=HOLD, 2=SELL), confidence
        """
        proba = self.predict_proba(X)
        
        # Get last prediction
        if len(proba.shape) > 1:
            last_proba = proba[-1]
        else:
            last_proba = proba
        
        predicted_class = np.argmax(last_proba)
        confidence = np.max(last_proba)
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            signal = 1  # HOLD if confidence too low
            logger.info(f"⚠️ Low confidence ({confidence:.2%}) - Signal: HOLD")
        else:
            signal = predicted_class
            signal_name = ['BUY', 'HOLD', 'SELL'][signal]
            logger.info(f"✅ Signal: {signal_name} (confidence: {confidence:.2%})")
        
        return signal, confidence
    
    def evaluate_consensus(self, X):
        """
        Check if all models agree on the prediction
        High consensus = more reliable signal
        """
        individual = self.get_individual_predictions(X)
        
        lstm_pred = np.argmax(individual['lstm'], axis=1)
        xgb_pred = np.argmax(individual['xgboost'], axis=1)
        rf_pred = np.argmax(individual['random_forest'], axis=1)
        
        # Check agreement
        consensus = []
        for i in range(len(lstm_pred)):
            votes = [lstm_pred[i], xgb_pred[i], rf_pred[i]]
            # All agree
            if len(set(votes)) == 1:
                consensus.append('full')
            # Majority (2 out of 3)
            elif votes.count(lstm_pred[i]) >= 2 or \
                 votes.count(xgb_pred[i]) >= 2 or \
                 votes.count(rf_pred[i]) >= 2:
                consensus.append('majority')
            # No agreement
            else:
                consensus.append('split')
        
        return consensus
    
    def save_rf(self, filename='random_forest.pkl'):
        """Save Random Forest model"""
        filepath = f'{self.model_path}{filename}'
        joblib.dump(self.rf_model, filepath)
        joblib.dump(self.rf_scaler, f'{self.model_path}rf_scaler.pkl')
        logger.info(f"✅ RF model saved: {filepath}")
    
    def load_rf(self, filename='random_forest.pkl'):
        """Load Random Forest model"""
        filepath = f'{self.model_path}{filename}'
        self.rf_model = joblib.load(filepath)
        self.rf_scaler = joblib.load(f'{self.model_path}rf_scaler.pkl')
        logger.info(f"✅ RF model loaded: {filepath}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble model"""
        # Convert one-hot to labels if needed
        if len(y_test.shape) > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test
        
        predictions = self.predict_classes(X_test)
        accuracy = (predictions == y_test_labels).mean()
        
        # Per-class accuracy
        class_accuracy = {}
        for cls in [0, 1, 2]:
            mask = y_test_labels == cls
            if mask.sum() > 0:
                class_acc = (predictions[mask] == cls).mean()
                class_accuracy[cls] = class_acc
        
        # Consensus analysis
        consensus = self.evaluate_consensus(X_test)
        consensus_stats = {
            'full': consensus.count('full') / len(consensus),
            'majority': consensus.count('majority') / len(consensus),
            'split': consensus.count('split') / len(consensus)
        }
        
        results = {
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'consensus_stats': consensus_stats
        }
        
        logger.info("="*60)
        logger.info("ENSEMBLE MODEL EVALUATION")
        logger.info("="*60)
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        logger.info(f"Class 0 (BUY) Accuracy: {class_accuracy.get(0, 0):.4f}")
        logger.info(f"Class 1 (HOLD) Accuracy: {class_accuracy.get(1, 0):.4f}")
        logger.info(f"Class 2 (SELL) Accuracy: {class_accuracy.get(2, 0):.4f}")
        logger.info("─"*60)
        logger.info("Model Consensus:")
        logger.info(f"  Full agreement: {consensus_stats['full']:.1%}")
        logger.info(f"  Majority agreement: {consensus_stats['majority']:.1%}")
        logger.info(f"  Split decision: {consensus_stats['split']:.1%}")
        logger.info("="*60)
        
        return results


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    from ml.lstm_model import LSTMModel
    from ml.xgboost_model import XGBoostModel
    
    # Mock data
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    
    # Initialize models (mock)
    lstm = LSTMModel(lookback=60, features=50)
    xgb = XGBoostModel()
    
    # Create ensemble
    ensemble = EnsembleModel(
        lstm_model=lstm,
        xgboost_model=xgb,
        lstm_weight=0.50,
        xgb_weight=0.30,
        rf_weight=0.20
    )
    
    # Train RF component
    ensemble.train_rf(X, y)
    
    print("✅ Ensemble model ready")
    print(f"Weights: LSTM={ensemble.lstm_weight}, XGB={ensemble.xgb_weight}, RF={ensemble.rf_weight}") 