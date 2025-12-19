import json
import os
from datetime import datetime, timedelta

class SignalTracker:
    """
    Track signal performance and adjust scoring dynamically
    """
    
    def __init__(self):
        self.tracker_file = 'signal_history.json'
        self.history = self.load_history()
    
    def load_history(self):
        """Load signal history from file"""
        try:
            if os.path.exists(self.tracker_file):
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            return {
                'signals': [],
                'market_performance': {},
                'time_performance': {},
                'setup_performance': {}
            }
        except:
            return {
                'signals': [],
                'market_performance': {},
                'time_performance': {},
                'setup_performance': {}
            }
    
    def save_history(self):
        """Save signal history"""
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Save history error: {e}")
    
    def add_signal(self, signal_data):
        """Record a new signal"""
        signal_record = {
            'timestamp': datetime.now().isoformat(),
            'market': signal_data['market'],
            'direction': signal_data['direction'],
            'score': signal_data['score'],
            'entry': signal_data['entry'],
            'sl': signal_data['sl'],
            'tp1': signal_data['tp1'],
            'tp2': signal_data['tp2'],
            'hour': datetime.now().hour
        }
        
        self.history['signals'].append(signal_record)
        
        # Keep only last 100 signals
        if len(self.history['signals']) > 100:
            self.history['signals'] = self.history['signals'][-100:]
        
        self.save_history()
    
    def get_market_win_rate(self, market):
        """Get historical win rate for a market (placeholder)"""
        # In production, track actual outcomes
        if market in self.history['market_performance']:
            return self.history['market_performance'][market].get('win_rate', 0.5)
        return 0.5  # Default 50%
    
    def get_time_multiplier(self, current_hour):
        """
        Get scoring multiplier based on time of day
        High liquidity hours = higher multiplier
        """
        # IST market hours (converted from UTC)
        # 4:30 - 9:30 UTC = 10:00 - 15:00 IST (peak hours)
        
        if 4 <= current_hour <= 9:
            return 1.2  # Peak Indian market hours
        elif 10 <= current_hour <= 14:
            return 1.1  # Active global hours
        elif 0 <= current_hour <= 3 or 15 <= current_hour <= 23:
            return 0.9  # Low activity
        
        return 1.0
    
    def get_recent_streak(self):
        """
        Get recent win/loss streak (simplified)
        Returns: streak count (positive = wins, negative = losses)
        """
        # Placeholder - in production, track actual outcomes
        return 0
    
    def should_pause_trading(self):
        """
        Determine if bot should pause due to poor performance
        """
        # Placeholder - implement based on tracked results
        return False
    
    def get_market_penalty(self, market):
        """
        Get penalty score for markets with recent losses
        Returns: penalty (0 to -20)
        """
        # Placeholder - in production, track losses
        if market in self.history['market_performance']:
            recent_losses = self.history['market_performance'][market].get('recent_losses', 0)
            if recent_losses >= 3:
                return -20
            elif recent_losses >= 2:
                return -10
        return 0
    
    def get_signals_in_window(self, hours=4):
        """Get number of signals sent in last N hours"""
        if not self.history['signals']:
            return 0
        
        cutoff = datetime.now() - timedelta(hours=hours)
        count = 0
        
        for signal in reversed(self.history['signals']):
            try:
                signal_time = datetime.fromisoformat(signal['timestamp'])
                if signal_time >= cutoff:
                    count += 1
                else:
                    break
            except:
                continue
        
        return count