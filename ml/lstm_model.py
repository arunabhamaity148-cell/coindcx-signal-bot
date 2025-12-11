"""
LSTM Model for Time Series Prediction
60 timesteps lookback, 50 features input
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel:
    def __init__(self, lookback=60, features=50, model_path='models/'):
        self.lookback = lookback
        self.features = features
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        
        # Create model directory
        os.makedirs(model_path, exist_ok=True)
    
    def build_model(self):
        """Build LSTM architecture"""
        model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, 
                 input_shape=(self.lookback, self.features)),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense layer
            Dense(16, activation='relu'),
            
            # Output layer (3 classes: BUY, HOLD, SELL)
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("✅ LSTM model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def prepare_sequences(self, X, y=None):
        """
        Prepare sequences for LSTM
        X shape: (samples, features)
        Output shape: (samples - lookback, lookback, features)
        """
        X_scaled = self.scaler.fit_transform(X)
        
        sequences = []
        targets = []
        
        for i in range(len(X_scaled) - self.lookback):
            seq = X_scaled[i:i + self.lookback]
            sequences.append(seq)
            
            if y is not None:
                targets.append(y[i + self.lookback])
        
        X_seq = np.array(sequences)
        
        if y is not None:
            y_seq = np.array(targets)
            return X_seq, y_seq
        
        return X_seq
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train LSTM model"""
        logger.info("🚀 Starting LSTM training...")
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        
        logger.info(f"Training sequences: {X_train_seq.shape}")
        logger.info(f"Validation sequences: {X_val_seq.shape}")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            f'{self.model_path}lstm_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        logger.info("✅ LSTM training completed")
        
        # Save scaler
        joblib.dump(self.scaler, f'{self.model_path}lstm_scaler.pkl')
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            logger.error("❌ Model not loaded!")
            return None
        
        X_seq = self.prepare_sequences(X)
        predictions = self.model.predict(X_seq, verbose=0)
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.predict(X)
    
    def predict_classes(self, X):
        """Get predicted classes"""
        proba = self.predict(X)
        return np.argmax(proba, axis=1)
    
    def get_confidence(self, X):
        """Get confidence scores"""
        proba = self.predict(X)
        confidence = np.max(proba, axis=1)
        return confidence
    
    def save(self, filename='lstm_model.h5'):
        """Save model"""
        filepath = f'{self.model_path}{filename}'
        self.model.save(filepath)
        joblib.dump(self.scaler, f'{self.model_path}lstm_scaler.pkl')
        logger.info(f"✅ Model saved: {filepath}")
    
    def load(self, filename='lstm_model.h5'):
        """Load model"""
        filepath = f'{self.model_path}{filename}'
        self.model = keras.models.load_model(filepath)
        self.scaler = joblib.load(f'{self.model_path}lstm_scaler.pkl')
        logger.info(f"✅ Model loaded: {filepath}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test)
        
        loss, accuracy = self.model.evaluate(X_test_seq, y_test_seq, verbose=0)
        
        predictions = self.predict_classes(X_test)
        y_true = np.argmax(y_test_seq, axis=1)
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for cls in [0, 1, 2]:
            mask = y_true == cls
            if mask.sum() > 0:
                class_acc = (predictions[mask] == cls).mean()
                class_accuracy[cls] = class_acc
        
        results = {
            'loss': loss,
            'accuracy': accuracy,
            'class_accuracy': class_accuracy
        }
        
        logger.info("="*60)
        logger.info("LSTM MODEL EVALUATION")
        logger.info("="*60)
        logger.info(f"Loss: {loss:.4f}")
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
    y = keras.utils.to_categorical(np.random.randint(0, 3, n_samples), 3)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Initialize and train
    lstm = LSTMModel(lookback=60, features=50)
    history = lstm.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)
    
    # Test prediction
    test_X = X[:100]
    predictions = lstm.predict(test_X)
    confidence = lstm.get_confidence(test_X)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Confidence scores: {confidence[:5]}") 