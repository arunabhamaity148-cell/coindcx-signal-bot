# ================================================================
# position_tracker.py — Track Manual Trades & Performance
# ================================================================

import json
from datetime import datetime
from typing import Optional
import aiofiles


class Position:
    """Represents a single trading position"""
    
    def __init__(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        notes: str = ""
    ):
        self.id = f"{symbol}_{int(datetime.utcnow().timestamp())}"
        self.symbol = symbol
        self.side = side.upper()
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = datetime.utcnow()
        self.exit_time = None
        self.exit_price = None
        self.status = "OPEN"
        self.pnl = 0.0
        self.pnl_percent = 0.0
        self.notes = notes
        self.max_profit = 0.0
        self.max_loss = 0.0
    
    def update_current_price(self, current_price: float):
        """Update position metrics with current price"""
        
        if self.side == "LONG":
            unrealized_pnl = (current_price - self.entry_price) * self.quantity
            pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            unrealized_pnl = (self.entry_price - current_price) * self.quantity
            pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
        
        self.pnl = unrealized_pnl
        self.pnl_percent = pnl_pct
        
        # Track max profit/loss
        if unrealized_pnl > self.max_profit:
            self.max_profit = unrealized_pnl
        if unrealized_pnl < self.max_loss:
            self.max_loss = unrealized_pnl
        
        # Check if stop loss or take profit hit
        if self.stop_loss:
            if (self.side == "LONG" and current_price <= self.stop_loss) or \
               (self.side == "SHORT" and current_price >= self.stop_loss):
                return "STOP_LOSS_HIT"
        
        if self.take_profit:
            if (self.side == "LONG" and current_price >= self.take_profit) or \
               (self.side == "SHORT" and current_price <= self.take_profit):
                return "TAKE_PROFIT_HIT"
        
        return "OK"
    
    def close(self, exit_price: float, notes: str = ""):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_time = datetime.utcnow()
        self.status = "CLOSED"
        
        if self.side == "LONG":
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_percent = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_percent = ((self.entry_price - exit_price) / self.entry_price) * 100
        
        if notes:
            self.notes += f" | Exit: {notes}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "status": self.status,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "notes": self.notes
        }


class PositionTracker:
    """Track and manage all positions"""
    
    def __init__(self, storage_file: str = "positions.json"):
        self.storage_file = storage_file
        self.open_positions = {}
        self.closed_positions = []
    
    async def load(self):
        """Load positions from file"""
        try:
            async with aiofiles.open(self.storage_file, 'r') as f:
                data = json.loads(await f.read())
                # Reconstruct positions
                # (Implementation depends on your needs)
        except FileNotFoundError:
            pass
    
    async def save(self):
        """Save positions to file"""
        data = {
            "open": [p.to_dict() for p in self.open_positions.values()],
            "closed": self.closed_positions
        }
        async with aiofiles.open(self.storage_file, 'w') as f:
            await f.write(json.dumps(data, indent=2))
    
    def add_position(self, position: Position):
        """Add new position"""
        self.open_positions[position.id] = position
    
    def close_position(self, position_id: str, exit_price: float, notes: str = ""):
        """Close a position"""
        if position_id in self.open_positions:
            pos = self.open_positions[position_id]
            pos.close(exit_price, notes)
            self.closed_positions.append(pos.to_dict())
            del self.open_positions[position_id]
            return pos
        return None
    
    async def update_all(self, price_dict: dict):
        """Update all open positions with current prices"""
        alerts = []
        
        for pos_id, pos in list(self.open_positions.items()):
            if pos.symbol in price_dict:
                current_price = price_dict[pos.symbol]
                status = pos.update_current_price(current_price)
                
                if status == "STOP_LOSS_HIT":
                    alerts.append({
                        "type": "STOP_LOSS",
                        "position": pos.to_dict()
                    })
                elif status == "TAKE_PROFIT_HIT":
                    alerts.append({
                        "type": "TAKE_PROFIT",
                        "position": pos.to_dict()
                    })
        
        return alerts
    
    def get_statistics(self) -> dict:
        """Calculate trading statistics"""
        
        if not self.closed_positions:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0
            }
        
        total_trades = len(self.closed_positions)
        wins = [p for p in self.closed_positions if p["pnl"] > 0]
        losses = [p for p in self.closed_positions if p["pnl"] < 0]
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(p["pnl"] for p in self.closed_positions)
        
        avg_win = sum(p["pnl"] for p in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(p["pnl"] for p in losses) / len(losses)) if losses else 0
        
        gross_profit = sum(p["pnl"] for p in wins)
        gross_loss = abs(sum(p["pnl"] for p in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "best_trade": max((p["pnl"] for p in self.closed_positions), default=0),
            "worst_trade": min((p["pnl"] for p in self.closed_positions), default=0)
        }
    
    def get_open_positions_summary(self) -> dict:
        """Get summary of open positions"""
        
        if not self.open_positions:
            return {
                "count": 0,
                "total_unrealized_pnl": 0,
                "positions": []
            }
        
        total_pnl = sum(p.pnl for p in self.open_positions.values())
        
        return {
            "count": len(self.open_positions),
            "total_unrealized_pnl": round(total_pnl, 2),
            "positions": [p.to_dict() for p in self.open_positions.values()]
        }


# Telegram command handlers for position management
class PositionCommands:
    """Handle position management via Telegram commands"""
    
    @staticmethod
    def format_open_positions(tracker: PositionTracker) -> str:
        """Format open positions for Telegram"""
        
        summary = tracker.get_open_positions_summary()
        
        if summary["count"] == 0:
            return "📊 <b>No open positions</b>"
        
        msg = f"""
📊 <b>OPEN POSITIONS</b> ({summary["count"]})
━━━━━━━━━━━━━━━━━━━━━━

💰 Total Unrealized P&L: <b>${summary["total_unrealized_pnl"]:.2f}</b>

"""
        
        for pos in summary["positions"]:
            emoji = "🟢" if pos["side"] == "LONG" else "🔴"
            pnl_emoji = "✅" if pos["pnl"] > 0 else "❌"
            
            msg += f"""
{emoji} <b>{pos["symbol"]}</b> {pos["side"]}
├ Entry: <code>${pos["entry_price"]:.6f}</code>
├ Qty: <code>{pos["quantity"]}</code>
├ P&L: {pnl_emoji} <b>${pos["pnl"]:.2f}</b> ({pos["pnl_percent"]:.2f}%)
"""
            if pos["stop_loss"]:
                msg += f"├ SL: <code>${pos['stop_loss']:.6f}</code>\n"
            if pos["take_profit"]:
                msg += f"└ TP: <code>${pos['take_profit']:.6f}</code>\n"
            msg += "\n"
        
        return msg
    
    @staticmethod
    def format_statistics(tracker: PositionTracker) -> str:
        """Format trading statistics for Telegram"""
        
        stats = tracker.get_statistics()
        
        if stats["total_trades"] == 0:
            return "📊 <b>No closed trades yet</b>"
        
        wr_emoji = "🟢" if stats["win_rate"] >= 50 else "🔴"
        pnl_emoji = "✅" if stats["total_pnl"] > 0 else "❌"
        
        msg = f"""
📊 <b>TRADING STATISTICS</b>
━━━━━━━━━━━━━━━━━━━━━━

📈 Total Trades: <b>{stats["total_trades"]}</b>
├ Wins: {stats["wins"]} ✅
├ Losses: {stats["losses"]} ❌
└ Win Rate: {wr_emoji} <b>{stats["win_rate"]}%</b>

💰 <b>P&L Summary</b>
├ Total P&L: {pnl_emoji} <b>${stats["total_pnl"]:.2f}</b>
├ Avg Win: <b>${stats["avg_win"]:.2f}</b>
├ Avg Loss: <b>${stats["avg_loss"]:.2f}</b>
└ Profit Factor: <b>{stats["profit_factor"]:.2f}</b>

🏆 Best Trade: <b>${stats["best_trade"]:.2f}</b>
💔 Worst Trade: <b>${stats["worst_trade"]:.2f}</b>

━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return msg