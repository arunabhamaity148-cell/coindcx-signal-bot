# TEST main.py — শুধুই loop test করার জন্য
import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

async def scanner():
    print("🚀 TEST Scanner Started...")
    async with aiohttp.ClientSession() as session:
        while True:
            print("---- NEW SCAN ROUND ----")
            for symbol in WATCHLIST:
                print("Checking:", symbol)
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(scanner())