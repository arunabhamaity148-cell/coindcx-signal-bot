import aiohttp
import base64
import urllib.parse


async def generate_chart_image(symbol: str, interval: str = "1"):
    """
    Generate TradingView chart screenshot using the public chart API.
    Works on Railway / Render / VPS (no browser needed).
    """

    tv_symbol = f"BINANCE:{symbol}"

    encoded = urllib.parse.quote(tv_symbol)

    url = (
        f"https://quickchart.io/chart/render/snapshot?"
        f"symbol={encoded}&interval={interval}&theme=dark&studies=ema50,ema20"
    )

    return url