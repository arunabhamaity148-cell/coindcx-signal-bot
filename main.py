# main.py
# Final runner that uses helpers.py (sync). Adds detailed loop logging:
# - loop number, start_idx, scanned, emitted, sleeping seconds
# - handles get_exchange / get_ex import flexibly
# - uses CooldownManager (init_cooldown_manager) from helpers
#
# Save next to helpers.py and run: python main.py

from __future__ import annotations
import os
import sys
import time
import math
import logging
import csv
from typing import List

# --- logging setup
LOG = logging.getLogger("main")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(os.getenv("MAIN_LOG_LEVEL", "INFO"))

# --- import helpers with graceful fallback for different names
try:
    # prefer the expected API
    from helpers import (
        init_cooldown_manager,
        analyze_symbol_sync,
        format_signal_message,
        send_telegram_message,
        preview_signal_log,
        cooldown_key_for,
        SCAN_BATCH_SIZE,
        LOOP_SLEEP_SECONDS,
        MAX_EMITS_PER_LOOP,
    )
    # attempt to import exchange factory (some helpers versions call get_exchange or get_ex)
    try:
        from helpers import get_exchange as _get_exchange  # type: ignore
    except Exception:
        try:
            from helpers import get_ex as _get_exchange  # type: ignore
        except Exception:
            _get_exchange = None
except Exception as e:
    LOG.exception("Failed to import helpers module: %s", e)
    raise

if _get_exchange is None:
    LOG.warning("helpers.get_exchange / get_ex not found — will create exchange directly if needed.")


def load_coins_csv(path: str = "coins.csv") -> List[str]:
    coins = []
    if not os.path.exists(path):
        LOG.warning("coins.csv not found at %s", path)
        return coins
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: 
                    continue
                # first column may be header 'symbol'
                symbol = str(row[0]).strip()
                if not symbol or symbol.lower() == "symbol":
                    continue
                coins.append(symbol.upper())
    except Exception as e:
        LOG.exception("Failed to read coins.csv: %s", e)
    LOG.info("Loaded %d coins from %s", len(coins), path)
    return coins


def create_exchange_instance():
    if _get_exchange:
        try:
            ex = _get_exchange(api_keys=False)
            LOG.info("Exchange created via helpers factory.")
            return ex
        except Exception as e:
            LOG.warning("helpers exchange factory failed: %s", e)
    # fallback: attempt to import ccxt and create a default binance exchange
    try:
        import ccxt
        ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        try:
            ex.load_markets()
        except Exception:
            LOG.debug("exchange.load_markets ignored/fail")
        LOG.info("Exchange created via fallback ccxt.binance().")
        return ex
    except Exception as e:
        LOG.exception("Unable to create exchange: %s", e)
        raise RuntimeError("Exchange creation failed") from e


def main_loop():
    coins = load_coins_csv()
    if not coins:
        LOG.error("No coins to scan. Place symbols in coins.csv (one per line). Exiting.")
        return

    ex = create_exchange_instance()

    # init cooldown manager
    cd = None
    try:
        cd = init_cooldown_manager()
        # init_cooldown_manager returns a coroutine or object depending on helpers implementation,
        # call it if coroutine-like
        if callable(getattr(cd, "__await__", None)):
            # it's a coroutine function, await-like -> run on loop
            import asyncio
            cd = asyncio.run(cd)
        LOG.info("Cooldown manager initialized.")
    except Exception as e:
        LOG.exception("Cooldown manager initialization failed: %s", e)
        cd = None

    loop_count = 0
    batch = int(SCAN_BATCH_SIZE or 20)
    sleep_s = float(LOOP_SLEEP_SECONDS or 5.0)
    max_emits = int(MAX_EMITS_PER_LOOP or 1)

    LOG.info("Starting main loop: batch=%d sleep=%.1fs max_emits=%d", batch, sleep_s, max_emits)

    try:
        while True:
            loop_count += 1
            emitted_total = 0
            scanned_total = 0

            # process in batches
            for start_idx in range(0, len(coins), batch):
                loop_start_time = time.time()
                batch_coins = coins[start_idx:start_idx + batch]
                LOG.info("Loop %d | start_idx=%d | scanning %d coins (batch)", loop_count, start_idx, len(batch_coins))

                emitted_in_batch = 0
                for symbol in batch_coins:
                    scanned_total += 1
                    try:
                        sig = analyze_symbol_sync(ex, symbol)
                    except Exception as e:
                        LOG.exception("Error analyzing %s: %s", symbol, e)
                        sig = None

                    if sig is None:
                        continue

                    # cooldown key & check
                    ckey = cooldown_key_for(sig.get("pair", symbol), sig.get("mode", None))
                    cooled = False
                    try:
                        if cd:
                            # cd.is_cooled may be coroutine or function
                            is_cooled = cd.is_cooled(ckey)
                            if callable(getattr(is_cooled, "__await__", None)):
                                import asyncio
                                is_cooled = asyncio.run(is_cooled)
                            cooled = is_cooled
                        else:
                            cooled = False
                    except Exception as e:
                        LOG.warning("Cooldown check failed for %s: %s", ckey, e)
                        cooled = False

                    if cooled:
                        LOG.debug("Skipping %s (%s) — cooled.", sig.get("pair"), sig.get("mode"))
                        continue

                    # emit signal
                    try:
                        preview_signal_log(sig)
                        msg = format_signal_message(sig)
                        sent = send_telegram_message(msg, preview=True)  # preview True by default; helpers handles NOTIFY_ONLY
                        if sent:
                            # set cooldown (async or sync)
                            if cd:
                                try:
                                    res = cd.set_cooldown(ckey, sig.get("mode", "MID"))
                                    if callable(getattr(res, "__await__", None)):
                                        import asyncio
                                        res = asyncio.run(res)
                                except Exception as e:
                                    LOG.warning("Failed to set cooldown for %s: %s", ckey, e)
                            emitted_total += 1
                            emitted_in_batch += 1
                        else:
                            LOG.warning("Telegram send failed for %s", sig.get("pair"))
                    except Exception as e:
                        LOG.exception("Failed to emit signal for %s: %s", symbol, e)

                    # respect max emits per loop (global)
                    if emitted_total >= max_emits:
                        LOG.info("Max emits reached for this loop (max_emits=%d). Breaking batch.", max_emits)
                        break

                # batch done
                elapsed = time.time() - loop_start_time
                LOG.info(
                    "Loop %d | batch start_idx=%d done — scanned=%d emitted=%d elapsed=%.2fs sleeping=%.2fs",
                    loop_count, start_idx, scanned_total, emitted_total, elapsed, sleep_s
                )

                if emitted_total >= max_emits:
                    break

                # sleep between batches
                time.sleep(sleep_s)

            # full loop done
            LOG.info("Loop %d complete — total_scanned=%d total_emitted=%d sleeping=%.2fs", loop_count, scanned_total, emitted_total, sleep_s)
            # short pause before next full cycle
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        LOG.info("Shutdown requested by user (KeyboardInterrupt). Exiting cleanly.")
    except Exception as e:
        LOG.exception("Unexpected error in main loop: %s", e)
    finally:
        LOG.info("Main stopped.")


if __name__ == "__main__":
    main_loop()