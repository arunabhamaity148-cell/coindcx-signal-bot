# ================================================================
# chart_image.py - TradingView Chart Image Generator
# ================================================================

import aiohttp
import urllib.parse
import logging

log = logging.getLogger("chart_image")


class ChartGenerator:
    """Generate TradingView chart images for signals"""
    
    # TradingView intervals
    INTERVALS = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "4h": "240",
        "1d": "D"
    }
    
    @staticmethod
    def get_tradingview_url(
        symbol: str,
        interval: str = "15m",
        indicators: list = None
    ) -> str:
        """
        Generate TradingView chart URL
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Chart interval (1m, 5m, 15m, 1h, 4h, 1d)
            indicators: List of indicators to show
            
        Returns:
            Direct chart URL
        """
        
        if indicators is None:
            indicators = ["MA@tv-basicstudies", "RSI@tv-basicstudies"]
        
        tv_symbol = f"BINANCE:{symbol}"
        tv_interval = ChartGenerator.INTERVALS.get(interval, "15")
        
        # Build indicator string
        studies = ",".join(indicators)
        
        # TradingView chart URL
        base_url = "https://www.tradingview.com/chart/"
        
        params = {
            "symbol": tv_symbol,
            "interval": tv_interval,
            "theme": "dark"
        }
        
        query_string = urllib.parse.urlencode(params)
        chart_url = f"{base_url}?{query_string}"
        
        return chart_url
    
    @staticmethod
    async def get_chart_screenshot_url(
        symbol: str,
        interval: str = "15m"
    ) -> str:
        """
        Generate chart screenshot URL using QuickChart API
        
        Args:
            symbol: Trading pair
            interval: Chart interval
            
        Returns:
            Screenshot URL (can be embedded in Telegram)
        """
        
        tv_symbol = f"BINANCE:{symbol}"
        tv_interval = ChartGenerator.INTERVALS.get(interval, "15")
        
        # QuickChart.io TradingView screenshot API
        url = (
            f"https://quickchart.io/tradingview/"
            f"?symbol={urllib.parse.quote(tv_symbol)}"
            f"&interval={tv_interval}"
            f"&theme=dark"
            f"&width=800"
            f"&height=400"
        )
        
        return url
    
    @staticmethod
    async def verify_chart_url(url: str) -> bool:
        """Verify if chart URL is accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(url, timeout=5) as resp:
                    return resp.status == 200
        except Exception as e:
            log.warning(f"Chart URL verification failed: {e}")
            return False


# Convenience functions
async def generate_chart_image(
    symbol: str,
    interval: str = "15m"
) -> str:
    """
    Quick function to generate chart image URL
    
    Usage:
        chart_url = await generate_chart_image("BTCUSDT", "15m")
    """
    return ChartGenerator.get_chart_screenshot_url(symbol, interval)


async def get_multi_timeframe_charts(symbol: str) -> dict:
    """
    Get chart URLs for multiple timeframes
    
    Returns:
        {
            "1m": url,
            "15m": url,
            "1h": url,
            "4h": url
        }
    """
    timeframes = ["1m", "15m", "1h", "4h"]
    
    charts = {}
    for tf in timeframes:
        charts[tf] = await generate_chart_image(symbol, tf)
    
    return charts