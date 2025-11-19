# main.py
# Orchestrator for ArunBot
# - Uses helpers.py (sync analyze_symbol_sync + async CooldownManager)
# - Copy-paste into repo and run: python main.py
# - Ensure helpers.py, requirements installed and .env / Railway secrets set.

import os
import sys
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from helpers import (
    get_exchange,
    analyze_symbol_sync,
    format_signal_message,
    send_telegram_message,
    init_cooldown_manager,
    cooldown_key_for,
    preview_signal_log,
    SCAN_BATCH_SIZE,
    LOOP_SLEEP_SECONDS,
    MAX_EMITS_PER_LOOP,
    NOTIFY_ONLY,
)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("main")

# Thread pool for blocking calls (ccxt sync fetch + telegram send)
EXECUTOR = ThreadPoolExecutor(max_workers=6)

# Load coins.csv
def load_coins(path: str = "coins.csv") -> List[str]:
    coins = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.lower().startswith("symbol"):
                    continue
                coins.append(s)
    except Exception as e:
        LOG.warning("Failed to read coins.csv: %s", e)
    return coins

async def send_telegram_async(msg: str, preview: bool = False) -> bool:
    # send_telegram_message is synchronous in helpers; run in executor
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, send_telegram_message, msg, preview)

async def analyze_and_maybe_emit(ex, coin: str, cd_mgr, emitted_counter: dict):
    """
    - Run the synchronous analyzer in threadpool
    - Check cooldown via cd_mgr (async)
    - If ok, set cooldown and emit (log + telegram)
    """
    loop = asyncio.get_running_loop()
    sig = await loop.run_in_executor(EXECUTOR, analyze_symbol_sync, ex, coin)
    if not sig:
        return False

    pair = sig.get("pair") or coin
    mode = sig.get("mode", "MID")
    key = cooldown_key_for(sig, mode)

    # Check cooldown: if cooled, skip
    is_blocked = await cd_mgr.is_cooled(key)
    if is_blocked:
        LOG.info("Cooldown active for %s (%s) — skipping", pair, mode)
        return False

    # Try set cooldown (atomic with Redis if available)
    ok = await cd_mgr.set_cooldown(key, mode)
    if not ok:
        LOG.info("Failed to set cooldown for %s — skipping", pair)
        return False

    # Emit signal: preview in logs + Telegram
    preview_signal_log(sig)
    msg = format_signal_message(sig)
    # send telegram (async wrapper)
    await send_telegram_async(msg, preview=NOTIFY_ONLY)
    LOG.info("EMIT -> %s | mode=%s | score=%.1f", pair, mode, sig.get("score", 0.0))

    emitted_counter["count"] += 1
    return True

async def worker_loop(coins: List[str]):
    if not coins:
        LOG.error("No coins loaded. Ensure coins.csv present.")
        return

    LOG.info("Starting main loop (scan batch size=%s, max emits per loop=%s)", SCAN_BATCH_SIZE, MAX_EMITS_PER_LOOP)
    ex = None
    try:
        ex = get_exchange(api_keys=False)
    except Exception as e:
        LOG.warning("Exchange init failed, continuing — will attempt fetches later: %s", e)
        ex = None

    # init cooldown manager
    cd_mgr = await init_cooldown_manager()

    start_idx = 0
    loop_idx = 0

    # main loop
    while True:
        loop_idx += 1
        emitted_counter = {"count": 0}
        LOG.info("Main loop #%d — scanning batch start_idx=%d", loop_idx, start_idx)

        # refresh exchange instance periodically to avoid stale connections
        if ex is None:
            try:
                ex = get_exchange(api_keys=False)
            except Exception as e:
                LOG.warning("Exchange init failed in loop: %s", e)
                ex = None

        batch = coins[start_idx: start_idx + SCAN_BATCH_SIZE]
        if not batch:
            # wrap around
            start_idx = 0
            batch = coins[0:SCAN_BATCH_SIZE]

        # analyze coins in batch concurrently but limit concurrency to executor size
        tasks = []
        for coin in batch:
            if emitted_counter["count"] >= MAX_EMITS_PER_LOOP:
                LOG.info("Max emits reached this loop (%d)", MAX_EMITS_PER_LOOP)
                break
            # schedule analyze+emit
            task = asyncio.create_task(analyze_and_maybe_emit(ex, coin, cd_mgr, emitted_counter))
            tasks.append(task)
            # small gap to avoid burst
            await asyncio.sleep(0.05)

        # wait for tasks to finish or time out
        if tasks:
            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                LOG.exception("Error during gather: %s", e)

        # advance start index
        start_idx = (start_idx + SCAN_BATCH_SIZE) % max(1, len(coins))
        LOG.info("Loop #%d complete — scanned=%d emitted=%d — sleeping %ss", loop_idx, len(batch), emitted_counter["count"], LOOP_SLEEP_SECONDS)
        await asyncio.sleep(LOOP_SLEEP_SECONDS)

async def main():
    # load coins
    coins = load_coins("coins.csv")
    LOG.info("Loaded %d coins from coins.csv", len(coins))

    # sanity: if few coins and we expect many, warn
    if len(coins) < 10:
        LOG.warning("Less than 10 coins loaded; consider adding more to coins.csv")

    # run worker loop
    try:
        await worker_loop(coins)
    except asyncio.CancelledError:
        LOG.info("Main cancelled")
    except Exception as e:
        LOG.exception("Fatal error in main: %s", e)
    finally:
        LOG.info("Shutting down executor")
        EXECUTOR.shutdown(wait=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOG.info("Interrupted by user — exiting")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
