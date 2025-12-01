# AI-Only Crypto Futures Signal Bot (Hybrid Indicators + Caching + SQLite)

Minimal, production-ready 2-file core with extras:
- main.py  — driver, live price fetch (ccxt), OpenAI scoring, Telegram send, caching, sqlite logging
- helpers.py — indicators, prompt builder, TP/SL, small cache

## Setup
1. Copy `.env.example` → `.env` and fill credentials (BOT_TOKEN, CHAT_ID, OPENAI_API_KEY).
2. Install dependencies: