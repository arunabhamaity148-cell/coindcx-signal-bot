# bootstrap_train.py
# ---------------------------------------------------
# AUTO MODEL CHECKER + QUICK MODEL GENERATOR + SERVER STARTER
# ---------------------------------------------------
# Railway start হলে:
# 1) দেখে নেবে model আছে কিনা
# 2) না থাকলে create_quick_models.py চালিয়ে 3টা model তৈরি করবে
# 3) সব ঠিক থাকলে uvicorn main.py চালু করবে
# ---------------------------------------------------

import os
import sys
import time
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("bootstrap")

MODEL_FILES = ["gb_model.pkl", "rf_model.pkl", "lr_model.pkl"]

# Commands
TRAIN_CMD_QUICK = [sys.executable, "create_quick_models.py"]   # quick synthetic trainer
TRAIN_CMD_FULL = [sys.executable, "train_ml.py"]               # optional full trainer
UVICORN_CMD = [
    "uvicorn", "main:app",
    "--host", "0.0.0.0",
    "--port", str(int(os.getenv("PORT", 8080))),
    "--log-level", "info"
]


# ---------------------------------------------------
# Utility: Check if models already exist
# ---------------------------------------------------
def models_exist():
    ok = True
    for f in MODEL_FILES:
        if not os.path.exists(f):
            log.info(f"Model missing: {f}")
            ok = False
    if ok:
        log.info("✓ All model files present.")
    return ok


# ---------------------------------------------------
# Utility: Run command with logs
# ---------------------------------------------------
def run_cmd(cmd, timeout=None):
    log.info(f"Running: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        for line in iter(proc.stdout.readline, b""):
            if not line:
                break
            try:
                log.info(line.decode().rstrip())
            except:
                log.info(str(line))

        proc.wait(timeout=timeout)

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        log.info(f"Finished: {' '.join(cmd)}")

    except subprocess.TimeoutExpired:
        proc.kill()
        log.error(f"Timeout: {' '.join(cmd)}")
        raise

    except Exception as e:
        log.error(f"Command failed: {cmd} -> {e}")
        raise


# ---------------------------------------------------
# MAIN LOGIC: ensure models exist
# ---------------------------------------------------
def ensure_models():
    # যদি আগেই থাকে → ok
    if models_exist():
        return True

    # Quick model generator চালাও
    try:
        log.info("⚡ Running quick model generator (create_quick_models.py)...")
        run_cmd(TRAIN_CMD_QUICK, timeout=600)
    except Exception as e:
        log.error(f"Quick model error: {e}")
        return False

    # আবার check
    if models_exist():
        log.info("✓ Quick models created successfully.")
        return True

    # fallback to full training (optional)
    try:
        log.info("⚠ Quick model fail — Running full train_ml.py...")
        run_cmd(TRAIN_CMD_FULL, timeout=3600)
    except Exception as e:
        log.error(f"Full model training failed: {e}")
        return False

    return models_exist()


# ---------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------
def main():
    log.info("🚀 Bootstrap starting...")

    # Optional delay (Railway-এর Redis/Websocket setup-এর জন্য)
    delay = int(os.getenv("BOOTSTRAP_DELAY_SEC", "3"))
    if delay > 0:
        time.sleep(delay)

    # Step 1: ensure models exist
    ok = ensure_models()
    if not ok:
        log.error("❌ Bootstrap failed (models missing). Stopping container.")
        sys.exit(1)

    # Step 2: Start uvicorn server
    log.info("🚀 Starting uvicorn server...")
    os.execvp(UVICORN_CMD[0], UVICORN_CMD)


if __name__ == "__main__":
    main()