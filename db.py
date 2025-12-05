import asyncpg
import os

DB_URL = os.getenv("DB_URL")

async def init_db():
    return await asyncpg.connect(DB_URL)


async def save_signal(conn, sig):
    await conn.execute(
        """
        INSERT INTO signals (symbol, side, score, confidence, last_price, created_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        """,
        sig["symbol"],
        sig["side"],
        sig["score"],
        sig.get("confidence", 0),
        sig["last"]
    )