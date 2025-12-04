# bootstrap_train.py
# -----------------
# এই ফাইলটি Railway (বা অন্য কোনো container) এ চালালে:
# 1) চেক করে যদি trained models (.pkl) না থাকে -> download_data.py চালাবে এবং train_ml.py চালাবে
# 2) মডেল তৈরি হওয়ার পরে uvicorn দিয়ে main.py চালাবে (FastAPI app)
#
# তোমাকে কেবল এই ফাইলটা GitHub-এ যোগ করে Dockerfile-এর CMD লাইনে
# `python bootstrap_train.py` বসাতে হবে।
#
# ব্যবহার: container startup এ এটি auto চলবে। প্রথমবার মডেল ট্রেন করতে
# কিছু সময় লাগতে পারে। পরেরবার মডেল থাকলে training স্কিপ করবে।

import os
import sys
import time
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("bootstrap")

# মডেল ফাইল নামগুলো (helpers.load_ensemble এ যেগুলো লোড করে)
MODEL_FILES = ["gb_model.pkl", "rf_model.pkl", "lr_model.pkl"]

# কোন কমান্ডগুলো চালাতে হবে (project root ধরছি)
DOWNLOAD_CMD = [sys.executable, "download_data.py"]
TRAIN_CMD = [sys.executable, "train_ml.py"]
UVICORN_CMD = [
    "uvicorn", "main:app",
    "--host", "0.0.0.0",
    "--port", str(int(os.getenv("PORT", 8080))),
    "--log-level", "info"
]

def models_exist():
    """Check if all model files are present in project root."""
    for f in MODEL_FILES:
        if not os.path.exists(f):
            log.info(f"Model missing: {f}")
            return False
    log.info("All model files present.")
    return True

def run_cmd(cmd, timeout=None, env=None):
    """Run a shell command and stream output to logs. Raises on failure."""
    log.info(f"Running: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )
        # stream stdout
        if proc.stdout:
            for line in iter(proc.stdout.readline, b''):
                if not line:
                    break
                try:
                    log.info(line.decode().rstrip())
                except:
                    log.info(line.rstrip())
        proc.wait(timeout=timeout)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        log.info(f"Command finished: {' '.join(cmd)}")
    except subprocess.TimeoutExpired:
        proc.kill()
        log.error(f"Command timeout: {' '.join(cmd)}")
        raise
    except Exception as e:
        log.error(f"Command failed: {' '.join(cmd)} -> {e}")
        raise

def ensure_models():
    """Ensure models exist; if not, download data and train."""
    if models_exist():
        return True

    # 1) Download data
    try:
        log.info("Starting data download (download_data.py)...")
        run_cmd(DOWNLOAD_CMD, timeout=60*20)  # up to 20 minutes (adjust if needed)
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return False

    # 2) Train models
    try:
        log.info("Starting ML training (train_ml.py)...")
        run_cmd(TRAIN_CMD, timeout=60*60)  # up to 60 minutes (adjust if needed)
    except Exception as e:
        log.error(f"Model training failed: {e}")
        return False

    # 3) verify
    if models_exist():
        log.info("Models successfully created.")
        return True
    else:
        log.error("Models still missing after training.")
        return False

def main():
    log.info("Bootstrap starting...")

    # Optional short delay so that dependent services (like Redis) can start
    delay = int(os.getenv("BOOTSTRAP_DELAY_SEC", "3"))
    if delay > 0:
        log.info(f"Sleeping {delay}s to allow services to come up...")
        time.sleep(delay)

    ok = ensure_models()
    if not ok:
        log.error("Bootstrap failed (models missing). Exiting.")
        # exit with non-zero so Railway shows failure in logs/health
        sys.exit(1)

    # If models are present, start the FastAPI app via uvicorn (blocking call)
    log.info("Starting uvicorn server (main:app)...")
    # Use exec to replace the current process so container PID 1 is uvicorn
    try:
        os.execvp(UVICORN_CMD[0], UVICORN_CMD)
    except Exception as e:
        log.exception(f"Failed to exec uvicorn: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()