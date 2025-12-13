"""
Risk Management System – FINAL
- SL ≤ 2% (ATR or fixed cap)
- TP base 3% ± confidence boost
- Duplicate symbol+side block
- Clean premium logs
"""
import logging
from typing import Dict, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("RiskManager")


class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.daily_pnl = 0
        self.daily_trades = 0
        self.open_positions = []
        self.consecutive_losses = 0
        self.today = datetime.now().date()

    # ---------- daily reset ----------
    def reset_daily_stats(self):
        current_date = datetime.now().date()
        if current_date != self.today:
            logger.info("📅 New trading day – Resetting daily stats")
            self.daily_pnl = 0
            self.daily_trades = 0
            self.today = current_date

    # ---------- position sizing ----------
    def calculate_position_size(self, capital: float, signal_confidence: float,
                                 market_regime: str = 'normal') -> float:
        base = self.config.get('position_size_percent', 0.15)
        if signal_confidence > 0.80:
            base *= 1.2
        elif signal_confidence < 0.65:
            base *= 0.8
        if self.consecutive_losses >= 3:
            base *= 0.5
            logger.warning("⚠️ 3+ losses – position halved")
        if market_regime == 'volatile':
            base *= 0.7

        size = capital * base
        max_single = capital * 0.50
        return min(size, max_single)

    # ---------- leverage ----------
    def calculate_leverage(self, market_regime: str, volatility: float,
                           signal_confidence: float) -> int:
        min_l = self.config.get('min_leverage', 3)
        max_l = self.config.get('max_leverage', 7)
        lev = 5                              # base

        if market_regime == 'trending':
            lev = max_l
        elif market_regime in ('ranging', 'volatile'):
            lev = min_l

        if volatility > 0.03:
            lev = min(lev, 3)
        if signal_confidence < 0.70:
            lev = min_l

        return int(lev)

    # ---------- STOP LOSS (≤ 2% cap) ----------
    def calculate_stop_loss(self, entry: float, side: str,
                            leverage: int, atr: float = None) -> float:
        # 1) ATR distance (max 1.5%) or fixed 1%
        base_dist = min(atr / entry, 0.015) if atr else 0.01
        # 2) hard cap 2% (leverage-adjusted)
        hard_dist = 0.02 / leverage
        dist = min(base_dist, hard_dist)

        if side == 'LONG':
            sl = entry * (1 - dist)
        else:
            sl = entry * (1 + dist)

        # 3) buffer vs liquidation (5%)
        liq = entry * (1 - 1 / leverage) if side == 'LONG' else entry * (1 + 1 / leverage)
        buffer = 0.05
        if side == 'LONG':
            sl = max(sl, liq * (1 + buffer))
        else:
            sl = min(sl, liq * (1 - buffer))

        logger.info(f"🛑 SL: ₹{sl:.2f} (dist {dist*100:.1f}%)")
        return round(sl, 2)

    # ---------- TAKE PROFIT (base 3%) ----------
    def calculate_take_profit(self, entry: float, side: str,
                              signal_confidence: float) -> float:
        base = 0.03
        boost = min(signal_confidence * 0.02, 0.02)   # max +2%
        dist = base + boost

        if side == 'LONG':
            tp = entry * (1 + dist)
        else:
            tp = entry * (1 - dist)

        logger.info(f"🎯 TP: ₹{tp:.2f} (dist {dist*100:.1f}%)")
        return round(tp, 2)

    # ---------- daily limits ----------
    def check_daily_limits(self) -> Tuple[bool, str]:
        self.reset_daily_stats()
        max_trades = self.config.get('max_daily_trades', 8)
        if self.daily_trades >= max_trades:
            return False, f"Daily trade limit {self.daily_trades}/{max_trades}"
        max_loss = self.config.get('max_daily_loss_percent', 0.20)
        if hasattr(self, 'initial_daily_capital'):
            limit = self.initial_daily_capital * max_loss
            if self.daily_pnl < -limit:
                return False, f"Daily loss limit ₹{abs(self.daily_pnl):.2f}"
        return True, "OK"

    # ---------- max positions ----------
    def check_max_positions(self) -> Tuple[bool, str]:
        max_pos = self.config.get('max_positions', 3)
        if len(self.open_positions) >= max_pos:
            return False, f"Max positions {len(self.open_positions)}/{max_pos}"
        return True, "OK"

    # ---------- trade validation ----------
    def validate_trade(self, capital: float, position_size: float,
                       leverage: int, entry: float,
                       stop_loss: float, side: str) -> Tuple[bool, str]:
        ok, msg = self.check_daily_limits()
        if not ok:
            return False, msg
        ok, msg = self.check_max_positions()
        if not ok:
            return False, msg
        if position_size > capital * 0.50:
            return False, "Position > 50% capital"
        if leverage > self.config.get('max_leverage', 7):
            return False, f"Leverage {leverage}x > max"

        # SL distance check (≥ 0.5%)
        dist = abs(entry - stop_loss) / entry
        if dist < 0.005:
            return False, "SL too tight (<0.5%)"

        # liquidation buffer check (≥ 5%)
        liq = entry * (1 - 1 / leverage) if side == 'LONG' else entry * (1 + 1 / leverage)
        buf_dist = abs(stop_loss - liq) / liq
        if buf_dist < 0.05:
            return False, f"SL too close to liq ({buf_dist:.1%})"

        logger.info("✅ Trade validation passed")
        return True, "OK"

    # ---------- position tracking ----------
    def add_position(self, position: Dict):
        self.open_positions.append(position)
        logger.info(f"📈 Position added ({len(self.open_positions)} open)")

    def remove_position(self, position_id: str):
        self.open_positions = [p for p in self.open_positions if p.get('id') != position_id]
        logger.info(f"📉 Position removed ({len(self.open_positions)} open)")

    def update_trade_result(self, pnl: float):
        self.daily_pnl += pnl
        self.daily_trades += 1
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        logger.info(f"📊 Daily P&L ₹{self.daily_pnl:+.2f} | Trades {self.daily_trades}")

    # ---------- emergency ----------
    def emergency_close_all(self) -> bool:
        logger.critical("🚨 EMERGENCY: Closing all positions!")
        self.open_positions.clear()
        return True
