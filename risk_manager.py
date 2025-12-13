class RiskManager:
    def __init__(self, config):
        self.config = config
        self.open_positions = {}

    def calculate_position_size(self, capital, confidence):
        risk_pct = self.config.get("position_size_percent", 0.1)
        return capital * risk_pct * confidence

    def validate_trade(self, capital, position_size):
        if position_size > capital * 0.2:
            return False, "Position too big"
        return True, "OK"

    def add_position(self, trade):
        self.open_positions[trade["id"]] = trade

    def remove_position(self, trade_id):
        if trade_id in self.open_positions:
            del self.open_positions[trade_id]

    def check_daily_limits(self):
        return True, "OK"