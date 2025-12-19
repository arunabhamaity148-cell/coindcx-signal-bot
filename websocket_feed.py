import json
import threading
import time
from typing import Dict, Optional
import websocket
from config import config

class WebSocketFeed:
    """
    Real-time price feed via WebSocket
    Provides 1-second latency price updates
    """
    
    def __init__(self):
        self.prices: Dict[str, float] = {}
        self.ws: Optional[websocket.WebSocketApp] = None
        self.connected = False
        self.thread = None
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # CoinDCX WebSocket format
            if 'p' in data and 's' in data:  # price and symbol
                symbol = data['s']
                price = float(data['p'])
                self.prices[symbol] = price
                
            elif 'market' in data and 'last_traded_price' in data:
                symbol = data['market']
                price = float(data['last_traded_price'])
                self.prices[symbol] = price
                
        except Exception as e:
            print(f"⚠️ WebSocket message error: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket Error: {error}")
        self.connected = False
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        print(f"🔌 WebSocket Closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        # Auto-reconnect after delay
        print(f"⏳ Reconnecting in {config.WS_RECONNECT_DELAY} seconds...")
        time.sleep(config.WS_RECONNECT_DELAY)
        self.start()
    
    def on_open(self, ws):
        """Handle WebSocket connection open"""
        print("✅ WebSocket Connected")
        self.connected = True
        
        # Subscribe to all trading pairs
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": config.PAIRS,
            "id": 1
        }
        
        ws.send(json.dumps(subscribe_msg))
        print(f"📡 Subscribed to {len(config.PAIRS)} pairs")
    
    def start(self):
        """Start WebSocket connection in background thread"""
        if self.thread and self.thread.is_alive():
            print("⚠️ WebSocket already running")
            return
        
        def run_websocket():
            """Run WebSocket in thread"""
            websocket.enableTrace(False)
            
            # CoinDCX public WebSocket URL
            ws_url = "wss://stream.coindcx.com/socket.io/?EIO=3&transport=websocket"
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Run forever with auto-reconnect
            self.ws.run_forever(
                ping_interval=30,
                ping_timeout=10
            )
        
        self.thread = threading.Thread(target=run_websocket, daemon=True)
        self.thread.start()
        
        # Wait for connection
        time.sleep(2)
        print(f"🚀 WebSocket feed started (Thread: {self.thread.name})")
    
    def stop(self):
        """Stop WebSocket connection"""
        if self.ws:
            self.ws.close()
        self.connected = False
        print("🛑 WebSocket stopped")
    
    def get_price(self, pair: str) -> Optional[float]:
        """
        Get current price for a pair
        
        Args:
            pair: Trading pair (e.g., 'F-BTC_INR')
        
        Returns:
            Current price or None if not available
        """
        return self.prices.get(pair)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices"""
        return self.prices.copy()
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.connected
    
    def wait_for_connection(self, timeout: int = 10) -> bool:
        """
        Wait for WebSocket to connect
        
        Args:
            timeout: Maximum wait time in seconds
        
        Returns:
            True if connected, False if timeout
        """
        elapsed = 0
        while not self.connected and elapsed < timeout:
            time.sleep(0.5)
            elapsed += 0.5
        
        return self.connected
    
    def get_status(self) -> Dict:
        """Get WebSocket connection status"""
        return {
            'connected': self.connected,
            'pairs_tracking': len(self.prices),
            'prices': list(self.prices.keys())
        }


# Singleton instance
ws_feed = WebSocketFeed()