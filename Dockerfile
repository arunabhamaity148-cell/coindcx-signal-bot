# FINAL Railway-ready Dockerfile (replace existing Dockerfile)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# system deps (minimal but enough for ta-lib, build and PostgreSQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl git ca-certificates libssl-dev libbz2-dev libffi-dev \
    liblzma-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
    zlib1g-dev libpq-dev gcc g++ make unzip && rm -rf /var/lib/apt/lists/*

# TA-Lib C library (needed by python ta-lib)
RUN set -eux; \
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz; \
    cd /tmp; tar -xzf ta-lib.tar.gz; cd ta-lib; ./configure --prefix=/usr && make && make install; \
    rm -rf /tmp/ta-lib*

# upgrade pip & tooling
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copy project
COPY . /app

# Install python deps (no venv, system pip). requirements.txt should NOT contain "python>=..."
RUN pip install --no-cache-dir -r requirements.txt

# TRAIN MODELS IF DATA EXISTS (glob all 12 CSVs)
# - uses shell glob expansion; if files present run training with all CSVs
# - set EPOCHS build-arg to control epochs
ARG EPOCHS=3
RUN bash -lc '\
    if ls data/BTCUSDT-15m-*.csv >/dev/null 2>&1; then \
      echo "Data CSVs found -> training models (this may take time)"; \
      python scripts/train_models.py --data data/BTCUSDT-15m-*.csv --epochs ${EPOCHS}; \
    else \
      echo "No data CSVs found in /app/data. Skipping model training."; \
    fi'

# Expose port if any (optional)
# EXPOSE 8080

# Default start command
CMD ["python", "main.py"]