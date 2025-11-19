# main.py — Orchestrator for SAFE setup (Option A)
# Copy -> save as main.py

import os, time, logging, asyncio
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

EXECUTOR = ThreadPoolExecutor(max_workers=6)

def load_coins(path: str = "coins.csv") -> List[str]:
    coins = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                if s.lower().startswith("symbol"): continue
                coins.append(s)
    except Exception as e:
        LOG.warning("Failed to read coins.csv: %s", e)
    return coins

async def send_telegram_async(msg: str, preview: bool = False) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, send_telegram_message, msg, preview)

async def analyze_and_maybe_emit(ex, coin: str, cd_mgr, emitted_counter: dict):
    loop = asyncio.get_running_loop()
    sig = await loop.run_in_executor(EXECUTOR, analyze_symbol_sync, ex, coin)
    if not sig:
        return False

    key = cooldown_key_for(sig)
    # check cooldown
    if await cd_mgr.is_cooled(key):
        LOG.info("Cooldown active %s — skip", key); return False

    # set cooldown
    ok = await cd_mgr.set_cooldown(key, sig.get("mode","MID"))
    if not ok:
        LOG.info("Could not set cooldown for %s — skip", key); return False

    # emit
    preview_signal_log(sig)
    msg = format_signal_message(sig)
    await send_telegram_async(msg, preview=NOTIFY_ONLY)
    LOG.info("EMIT -> %s | mode=%s | score=%.1f", sig.get("pair"), sig.get("mode"), sig.get("score"))
    emitted_counter["count"] += 1
    return True

async def worker_loop(coins):
    if not coins:
        LOG.error("No coins loaded"); return
    LOG.info("Starting worker loop — batch=%s max_emits=%s", SCAN_BATCH_SIZE, MAX_EMITS_PER_LOOP)
    ex = None
    try:
        ex = get_exchange(api_keys=False)
    except Exception as e:
        LOG.warning("Exchange init failed: %s", e)

    cd_mgr = await init_cooldown_manager()

    start_idx = 0
    loop_idx = 0
    while True:
        loop_idx += 1
        emitted_counter = {"count": 0}
        LOG.info("Loop %d start idx=%d", loop_idx, start_idx)

        if ex is None:
            try: ex = get_exchange(api_keys=False)
            except Exception as e: LOG.warning("Exchange init in-loop failed: %s", e); ex = None

        batch = coins[start_idx:start_idx+SCAN_BATCH_SIZE]
        if not batch:
            start_idx = 0; batch = coins[0:SCAN_BATCH_SIZE]

        tasks = []
        for coin in batch:
            if emitted_counter["count"] >= MAX_EMITS_PER_LOOP:
                LOG.info("Max emits reached this loop"); break
            task = asyncio.create_task(analyze_and_maybe_emit(ex, coin, cd_mgr, emitted_counter))
            tasks.append(task)
            await asyncio.sleep(0.05)

        if tasks:
            try: await asyncio.gather(*tasks)
            except Exception as e: LOG.exception("Gather error: %s", e)

        start_idx = (start_idx + SCAN_BATCH_SIZE) % max(1, len(coins))
        LOG.info("Loop %d done scanned=%d emitted=%d sleeping=%s", loop_idx, len(batch), emitted_counter["count"], LOOP_SLEEP_SECONDS)
        await asyncio.sleep(LOOP_SLEEP_SECONDS)

async def main():
    coins = load_coins("coins.csv")
    LOG.info("Loaded %d coins", len(coins))
    await worker_loop(coins)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOG.info("Interrupted — exiting")