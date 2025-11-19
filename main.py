# main.py
# Final runner for signal scanner (uses helpers.py)
# - Single-file entrypoint that uses your helpers.py functions
# - Reads coins.csv, loops in batches, honors cooldowns and NOTIFY_ONLY
# - Designed to work with the helpers.py you provided
# Save as main.py and run (python main.py)

from __future__ import annotations
import os
import sys
import time
import csv
import signal
import asyncio
import logging
from typing import List, Optional

# ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# import helpers (your helpers.py must be in same folder)
from helpers import (
    get_exchange,
    analyze_symbol_sync,
    init_cooldown_manager,
    cooldown_key_for,
    preview_signal_log,
    send_telegram_message,
)

# basic logger
LOG = logging.getLogger("main")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(os.getenv("MAIN_LOG_LEVEL", "INFO"))

# config from env (mirrors helpers)
SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "1"))
COIN_CSV = os.getenv("COINS_CSV_PATH", "coins.csv")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")

# Graceful shutdown flag
_run = True
def _handle_exit(signum, frame):
    global _run
    LOG.info("Shutdown signal received (%s). Stopping main loop...", signum)
    _run = False

signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)

# helper: read coins.csv (first column header 'symbol', supports both SYMBOL or SYMBOL/USDT)
def load_coins_from_csv(path: str) -> List[str]:
    coins: List[str] = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            # if header looks like symbol, continue; otherwise treat first row as data
            # we accept simple lists like:
            # symbol
            # BTC
            # ETH
            # or full pairs like BTC/USDT
            if header:
                if len(header) == 1 and header[0].strip().lower() == "symbol":
                    # header consumed, iterate remaining
                    pass
                else:
                    # treat header as a coin entry if it doesn't look like header
                    coins.append(header[0].strip())
            for row in rdr:
                if not row: continue
                val = row[0].strip()
                if not val: continue
                # normalize: allow either 'BTC' or 'BTC/USDT'
                if "/" not in val:
                    val = f"{val}/{QUOTE_ASSET}"
                coins.append(val)
    except FileNotFoundError:
        LOG.error("coins.csv not found at %s — create coins.csv with a `symbol` column", path)
    except Exception as e:
        LOG.exception("Failed to read coins.csv: %s", e)
    # dedupe preserving order
    seen = set()
    out = []
    for c in coins:
        up = c.upper()
        if up in seen: continue
        seen.add(up)
        out.append(up)
    return out

async def main_loop():
    LOG.info("Starting main loop")
    # init exchange (no api keys by default)
    try:
        ex = get_exchange(api_keys=False)
    except Exception as e:
        LOG.exception("Failed to create exchange instance: %s", e)
        return

    # init cooldown manager (from helpers)
    cd = await init_cooldown_manager()

    # load coins
    coins = load_coins_from_csv(COIN_CSV)
    if not coins:
        LOG.error("No coins loaded. Exiting.")
        return
    LOG.info("Loaded %d coins from %s", len(coins), COIN_CSV)

    loop_idx = 0
    while _run:
        loop_idx += 1
        LOG.info("Main loop #%d — scanning up to %d coins (start_idx=%d)", loop_idx, SCAN_BATCH_SIZE, (loop_idx-1)*SCAN_BATCH_SIZE)
        emitted = 0
        start_idx = ((loop_idx-1) * SCAN_BATCH_SIZE) % max(1, len(coins))
        # create a view of coins for this loop (wrap-around)
        batch = []
        idx = start_idx
        while len(batch) < SCAN_BATCH_SIZE and len(batch) < len(coins):
            batch.append(coins[idx % len(coins)])
            idx += 1

        for pair in batch:
            if emitted >= MAX_EMITS_PER_LOOP:
                LOG.debug("Max emits for this loop reached (%d). Breaking.", MAX_EMITS_PER_LOOP)
                break
            if not _run:
                break
            try:
                # quick cooldown check - if cooled, skip; we want to emit only if not cooled
                key_mid = cooldown_key_for(pair, "MID")
                key_quick = cooldown_key_for(pair, "QUICK")
                key_trend = cooldown_key_for(pair, "TREND")
                # If any of the modes are still cooling, we still allow analysis but skip emission for that mode as needed.
                # Run analysis (sync)
                sig = analyze_symbol_sync(ex, pair)
                if not sig:
                    LOG.debug("No signal for %s", pair)
                    continue

                # check per-pair cooldown for this sig.mode
                cd_key = cooldown_key_for(sig)
                is_cooled = await cd.is_cooled(cd_key)
                if is_cooled:
                    LOG.debug("%s skipped: cooled (%s)", sig.get("pair"), cd_key)
                    continue

                # preview in logs
                preview_signal_log(sig)

                # try set cooldown (atomic-ish)
                set_ok = await cd.set_cooldown(cd_key, sig.get("mode", "MID"))
                if not set_ok:
                    LOG.debug("%s cooldown set failed/other actor set it", sig.get("pair"))
                    continue

                # format and send message
                msg = preview_signal_message(sig := sig) if False else None  # slip - for compatibility below
                # use helpers.format_signal_message by calling send_telegram_message expects formatted text
                from helpers import format_signal_message
                text = format_signal_message(sig)
                sent = send_telegram_message(text, preview=True)  # preview True will only log unless NOTIFY_ONLY env off
                if sent:
                    emitted += 1
                    LOG.info("Emitted signal for %s mode=%s score=%.1f", sig.get("pair"), sig.get("mode"), sig.get("score"))
                else:
                    LOG.warning("Failed to send signal for %s", sig.get("pair"))
            except Exception as e:
                LOG.exception("Error while analyzing %s: %s", pair, e)
                # continue scanning others

        LOG.info("Loop %d done: scanned=%d emitted=%d sleeping=%.1fs", loop_idx, len(batch), emitted, LOOP_SLEEP_SECONDS)
        # sleep but allow quick exit
        slept = 0.0
        while slept < LOOP_SLEEP_SECONDS and _run:
            await asyncio.sleep(0.5)
            slept += 0.5

    LOG.info("Main loop exiting — persisting cooldown map if applicable and cleaning up.")
    # no explicit cleanup required; helpers cooldown persists to local JSON if redis not used

def preview_signal_message(sig):
    # small wrapper to avoid circular import earlier — uses helpers.format_signal_message
    try:
        from helpers import format_signal_message
        return format_signal_message(sig)
    except Exception:
        return str(sig)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        LOG.info("KeyboardInterrupt received — shutting down.")
    except Exception as e:
        LOG.exception("Fatal error in main: %s", e)
        raise