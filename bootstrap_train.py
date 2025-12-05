import os, sys, time, logging, subprocess
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("bootstrap")

MODEL_FILES = ["gb_model.pkl", "rf_model.pkl", "lr_model.pkl"]
UVICORN_CMD = ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(int(os.getenv("PORT", 8080))), "--log-level", "info"]

def models_exist():
    ok = all(os.path.exists(f) for f in MODEL_FILES)
    log.info("✓ All model files present." if ok else "⚠ Models missing.")
    return ok

def run(cmd, timeout=None):
    log.info(f"Run: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        log.info(line)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

def ensure_models():
    if models_exist():
        return True
    try:
        log.info("⚡ Quick model generator...")
        run([sys.executable, "create_quick_models.py"], timeout=120)
    except Exception as e:
        log.error(f"Quick model failed: {e}")
        return False
    return models_exist()

def main():
    log.info("🚀 Bootstrap starting...")
    time.sleep(int(os.getenv("BOOTSTRAP_DELAY_SEC", "3")))
    if not ensure_models():
        log.error("❌ Bootstrap failed.")
        sys.exit(1)
    log.info("🚀 Starting uvicorn...")
    os.execvp(UVICORN_CMD[0], UVICORN_CMD)

if __name__ == "__main__":
    main()
