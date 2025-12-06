# ================================================================
# dashboard.py - Web Dashboard Backend API
# ================================================================

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import json
import asyncio
from datetime import datetime, timedelta


class DashboardManager:
    """Manage dashboard data and WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.signal_history = []
        self.performance_data = {}
    
    async def connect(self, websocket: WebSocket):
        """Add new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass
    
    async def send_signal_update(self, signal: dict):
        """Broadcast new signal to dashboard"""
        await self.broadcast({
            "type": "new_signal",
            "data": signal,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def send_performance_update(self, stats: dict):
        """Broadcast performance update"""
        await self.broadcast({
            "type": "performance",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        })


# Dashboard routes
def create_dashboard_routes(app: FastAPI, dashboard_manager: DashboardManager):
    """Add dashboard routes to FastAPI app"""
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard():
        """Serve dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f23;
            color: #e0e0e0;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .header h1 { font-size: 32px; margin-bottom: 10px; }
        .header .status { font-size: 14px; opacity: 0.9; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #2a2a3e;
        }
        .stat-card h3 {
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .stat-card .value {
            font-size: 32px;
            font-weight: bold;
            color: #fff;
        }
        .stat-card .change {
            font-size: 14px;
            margin-top: 5px;
        }
        .change.positive { color: #10b981; }
        .change.negative { color: #ef4444; }
        .signals-container {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .signals-container h2 {
            margin-bottom: 20px;
            font-size: 20px;
        }
        .signal-item {
            background: #0f0f23;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        .signal-item.long { border-left-color: #10b981; }
        .signal-item.short { border-left-color: #ef4444; }
        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .signal-symbol {
            font-size: 18px;
            font-weight: bold;
        }
        .signal-confidence {
            background: #667eea;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        .signal-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            font-size: 14px;
            color: #888;
        }
        .chart-container {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        canvas { max-height: 300px; }
        .live-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 8px;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Trading Bot Dashboard v5.0</h1>
        <div class="status">
            <span class="live-indicator"></span>
            <span id="status">Connecting...</span>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <h3>Total Signals</h3>
            <div class="value" id="totalSignals">0</div>
            <div class="change" id="signalsChange">--</div>
        </div>
        <div class="stat-card">
            <h3>Win Rate</h3>
            <div class="value" id="winRate">0%</div>
            <div class="change positive" id="winRateChange">--</div>
        </div>
        <div class="stat-card">
            <h3>Total P&L</h3>
            <div class="value" id="totalPnl">$0</div>
            <div class="change" id="pnlChange">--</div>
        </div>
        <div class="stat-card">
            <h3>Active Positions</h3>
            <div class="value" id="activePositions">0</div>
            <div class="change" id="positionsChange">--</div>
        </div>
    </div>
    
    <div class="chart-container">
        <h2>Equity Curve</h2>
        <canvas id="equityChart"></canvas>
    </div>
    
    <div class="signals-container">
        <h2>Recent Signals</h2>
        <div id="signalsList"></div>
    </div>
    
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws/dashboard`);
        
        ws.onopen = () => {
            document.getElementById('status').textContent = 'Connected';
        };
        
        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            
            if (message.type === 'new_signal') {
                addSignalToList(message.data);
            } else if (message.type === 'performance') {
                updateStats(message.data);
            }
        };
        
        ws.onerror = () => {
            document.getElementById('status').textContent = 'Connection Error';
        };
        
        function addSignalToList(signal) {
            const list = document.getElementById('signalsList');
            const item = document.createElement('div');
            item.className = `signal-item ${signal.side}`;
            
            item.innerHTML = `
                <div class="signal-header">
                    <span class="signal-symbol">${signal.symbol} ${signal.side.toUpperCase()}</span>
                    <span class="signal-confidence">${signal.confidence}%</span>
                </div>
                <div class="signal-details">
                    <div>Price: $${signal.last.toFixed(6)}</div>
                    <div>Score: ${signal.score.toFixed(1)}/17</div>
                    <div>Strategy: ${signal.strategy}</div>
                    <div>Time: ${new Date().toLocaleTimeString()}</div>
                </div>
            `;
            
            list.insertBefore(item, list.firstChild);
            
            // Keep only last 10 signals
            while (list.children.length > 10) {
                list.removeChild(list.lastChild);
            }
        }
        
        function updateStats(stats) {
            document.getElementById('totalSignals').textContent = stats.total_trades || 0;
            document.getElementById('winRate').textContent = (stats.win_rate || 0).toFixed(1) + '%';
            document.getElementById('totalPnl').textContent = '$' + (stats.total_pnl || 0).toFixed(2);
            document.getElementById('activePositions').textContent = stats.open_positions || 0;
        }
        
        // Initialize equity chart
        const ctx = document.getElementById('equityChart').getContext('2d');
        const equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Equity',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: { 
                        grid: { color: '#2a2a3e' },
                        ticks: { color: '#888' }
                    },
                    x: { 
                        grid: { color: '#2a2a3e' },
                        ticks: { color: '#888' }
                    }
                }
            }
        });
        
        // Fetch initial data
        fetch('/stats')
            .then(r => r.json())
            .then(data => updateStats(data.trading_stats || {}));
        
        fetch('/signals')
            .then(r => r.json())
            .then(data => {
                (data.signals || []).slice(0, 10).forEach(signal => {
                    addSignalToList(signal);
                });
            });
    </script>
</body>
</html>
        """
    
    @app.websocket("/ws/dashboard")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        await dashboard_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            dashboard_manager.disconnect(websocket)
    
    @app.get("/api/dashboard/summary")
    async def get_dashboard_summary():
        """Get dashboard summary data"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "signals_today": len(dashboard_manager.signal_history),
            "performance": dashboard_manager.performance_data
        }


# Global dashboard manager instance
dashboard_manager = DashboardManager()