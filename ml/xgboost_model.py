"""
XGBoost Model for Trading Signals
Fast gradient boosting for feature-based predictions
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel:
    def __init__(self, n_estimators=200, max_depth=8, learning_rate=0.1, 
                 model_path='models/'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        
        # Create model directory
        os.makedirs(model_path, exist_ok=True)
    
    def build_model(self):
        """Build XGBoost classifier"""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='multi:softprob',
            num_class=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("✅ XGBoost model built successfully")
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        logger.info("🚀 Starting XGBoost training...")
        
        # Convert one-hot to labels if needed
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
            y_val = np.argmax(y_val, axis=1)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Train with early stopping
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=20,
            verbose=True
        )
        
        logger.info("✅ XGBoost training completed")
        
        # Save scaler
        joblib.dump(self.scaler, f'{self.model_path}xgb_scaler.pkl')
        
        return self.model
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            logger.error("❌ Model not loaded!")
            return None
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict_proba(X_scaled)
        return predictions
    
    def predict(self, X):
        """Make predictions"""
        return self.predict_proba(X)
    
    def predict_classes(self, X):
        """Get predicted classes"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_confidence(self, X):
        """Get confidence scores"""
        proba = self.predict_proba(X)
        confidence = np.max(proba, axis=1)
        return confidence
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            return None
        
        importance = self.model.feature_importances_
        return importance
    
    def save(self, filename='xgboost_model.pkl'):
        """Save model"""
        filepath = f'{self.model_path}{filename}'
        joblib.dump(self.model, filepath)
        joblib.dump(self.scaler, f'{self.model_path}xgb_scaler.pkl')
        logger.info(f"✅ Model saved: {filepath}")
    
    def load(self, filename='xgboost_model.pkl'):
        """Load model"""
        filepath = f'{self.model_path}{filename}'
        self.model = joblib.load(filepath)
        self.scaler = joblib.load(f'{self.model_path}xgb_scaler.pkl')
        logger.info(f"✅ Model loaded: {filepath}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        # Convert one-hot to labels if needed
        if len(y_test.shape) > 1:
            y_test = np.argmax(y_test, axis=1)
        
        predictions = self.predict_classes(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for cls in [0, 1, 2]:
            mask = y_test == cls
            if mask.sum() > 0:
                class_acc = (predictions[mask] == cls).mean()
                class_accuracy[cls] = class_acc
        
        results = {
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'report': classification_report(y_test, predictions, 
                                           target_names=['BUY', 'HOLD', 'SELL'])
        }
        
        logger.info("="*60)
        logger.info("XGBOOST MODEL EVALUATION")
        logger.info("="*60)
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Class 0 (BUY) Accuracy: {class_accuracy.get(0, 0):.4f}")
        logger.info(f"Class 1 (HOLD) Accuracy: {class_accuracy.get(1, 0):.4f}")
        logger.info(f"Class 2 (SELL) Accuracy: {class_accuracy.get(2, 0):.4f}")
        logger.info("="*60)
        
        return results


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Mock data
    n_samples = 10000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Initialize and train
    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train, X_val, y_val)
    
    # Test prediction
    test_X = X[:100]
    predictions = xgb_model.predict(test_X)
    confidence = xgb_model.get_confidence(test_X)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Confidence scores: {confidence[:5]}")
    
    # Feature importance
    importance = xgb_model.get_feature_importance()
    print(f"Top 5 features: {np.argsort(importance)[-5:]}") 