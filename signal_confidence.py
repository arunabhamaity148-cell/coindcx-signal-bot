# ================================================================
# signal_confidence.py — Advanced Signal Validation
# ================================================================

import numpy as np
from datetime import datetime, timedelta


class SignalConfidence:
    """Calculate confidence percentage for each signal"""
    
    def __init__(self):
        self.weight_map = {
            # Trend strength (40% weight)
            "mtf_bull": 8,
            "mtf_bear": 8,
            "atr_stable": 6,
            "low_atr": 6,
            
            # Momentum (30% weight)
            "mom1": 7,
            "mom5": 7,
            "imbalance": 8,
            "vwap_momentum": 8,
            
            # Price position (20% weight)
            "vwap_close": 5,
            "vwap_small_dev": 5,
            "rsi_mid": 5,
            "rsi_flip": 5,
            
            # Market structure (10% weight)
            "spread_ok": 3,
            "depth_ok": 3,
            "rsi_div": 4,
        }
        
        self.total_weight = sum(self.weight_map.values())
    
    def calculate(self, logic_dict: dict, params: dict) -> dict:
        """
        Calculate confidence percentage and quality metrics
        
        Returns:
            {
                "confidence": 0-100,
                "quality": "EXCELLENT|GOOD|FAIR|WEAK",
                "strength": "STRONG|MODERATE|WEAK",
                "risk_level": "LOW|MEDIUM|HIGH"
            }
        """
        
        # Weighted score
        weighted_sum = sum(
            self.weight_map.get(k, 0) * v 
            for k, v in logic_dict.items()
        )
        
        confidence = (weighted_sum / self.total_weight) * 100
        
        # Quality assessment
        if confidence >= 75:
            quality = "EXCELLENT"
            strength = "STRONG"
        elif confidence >= 60:
            quality = "GOOD"
            strength = "MODERATE"
        elif confidence >= 45:
            quality = "FAIR"
            strength = "MODERATE"
        else:
            quality = "WEAK"
            strength = "WEAK"
        
        # Risk level based on volatility & spread
        risk_level = self._assess_risk(params)
        
        # Additional metrics
        trend_alignment = self._check_trend_alignment(logic_dict)
        momentum_strength = self._check_momentum(logic_dict)
        
        return {
            "confidence": round(confidence, 1),
            "quality": quality,
            "strength": strength,
            "risk_level": risk_level,
            "trend_aligned": trend_alignment,
            "momentum_score": momentum_strength,
            "passed_count": sum(logic_dict.values()),
            "total_checks": len(logic_dict)
        }
    
    def _assess_risk(self, params: dict) -> str:
        """Assess risk level based on market conditions"""
        atr = params.get("atr", 0)
        spread = params.get("spread", 0)
        last = params.get("last", 1)
        
        atr_pct = (atr / last) * 100 if last > 0 else 0
        
        # High risk indicators
        if atr_pct > 0.25 or spread > 0.4:
            return "HIGH"
        elif atr_pct > 0.15 or spread > 0.25:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _check_trend_alignment(self, logic: dict) -> bool:
        """Check if multiple timeframes agree"""
        mtf_signals = logic.get("mtf_bull", 0) or logic.get("mtf_bear", 0)
        return mtf_signals == 1
    
    def _check_momentum(self, logic: dict) -> int:
        """Score momentum strength (0-10)"""
        score = 0
        if logic.get("mom1"): score += 3
        if logic.get("mom5"): score += 3
        if logic.get("imbalance"): score += 2
        if logic.get("vwap_momentum"): score += 2
        return score


class SignalHistory:
    """Track signal performance history"""
    
    def __init__(self, max_history=100):
        self.history = []
        self.max_history = max_history
    
    def add_signal(self, signal: dict):
        """Add new signal to history"""
        signal["timestamp"] = datetime.utcnow()
        self.history.append(signal)
        
        # Keep only recent signals
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_symbol_stats(self, symbol: str, hours: int = 24) -> dict:
        """Get statistics for a specific symbol"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        recent = [
            s for s in self.history 
            if s["symbol"] == symbol and s["timestamp"] > cutoff
        ]
        
        if not recent:
            return None
        
        avg_confidence = np.mean([s.get("confidence", 0) for s in recent])
        
        return {
            "signal_count": len(recent),
            "avg_confidence": round(avg_confidence, 1),
            "last_signal": recent[-1]["timestamp"].isoformat(),
            "strategies_used": list(set(s["strategy"] for s in recent))
        }
    
    def get_recent_signals(self, limit: int = 10) -> list:
        """Get most recent signals"""
        return sorted(
            self.history, 
            key=lambda x: x["timestamp"], 
            reverse=True
        )[:limit]


class SignalValidator:
    """Validate signals before sending"""
    
    @staticmethod
    def should_send(signal: dict, confidence_data: dict) -> tuple[bool, str]:
        """
        Determine if signal should be sent to user
        
        Returns:
            (bool: should_send, str: reason)
        """
        
        confidence = confidence_data["confidence"]
        quality = confidence_data["quality"]
        risk = confidence_data["risk_level"]
        
        # Filter rules
        if confidence < 40:
            return False, "Confidence too low"
        
        if quality == "WEAK":
            return False, "Signal quality weak"
        
        if risk == "HIGH" and confidence < 70:
            return False, "High risk, low confidence"
        
        # Check minimum passed logic
        if confidence_data["passed_count"] < 6:
            return False, "Insufficient logic checks passed"
        
        return True, "Signal validated"


# Usage example
def enhance_signal(signal: dict, params: dict) -> dict:
    """Add confidence metrics to existing signal"""
    
    conf_calc = SignalConfidence()
    confidence_data = conf_calc.calculate(signal["logic"], params)
    
    # Merge confidence data into signal
    signal.update(confidence_data)
    
    # Validate
    validator = SignalValidator()
    should_send, reason = validator.should_send(signal, confidence_data)
    
    signal["validated"] = should_send
    signal["validation_reason"] = reason
    
    return signal