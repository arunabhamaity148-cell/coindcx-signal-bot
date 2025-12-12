# ============================
# FINAL DOCKERFILE – RAILWAY
# ============================

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------
# System dependencies (TA-Lib compile + ML libs support)
# --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl git ca-certificates libssl-dev libbz2-dev \
    libffi-dev liblzma-dev libncurses5-dev libncursesw5-dev libreadline-dev \
    libsqlite3-dev zlib1g-dev libpq-dev gcc g++ make unzip \
    && rm -rf /var/lib/apt/lists/*

# -------- Install TA-Lib C library --------
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz \
 && cd /tmp && tar -xzf ta-lib.tar.gz && cd ta-lib \
 && ./configure --prefix=/usr && make && make install \
 && rm -rf /tmp/ta-lib /tmp/ta-lib.tar.gz

# -------- Python tooling --------
RUN python -m pip install --upgrade pip setuptools wheel

# --------------------------------------------------------
# Copy project
# --------------------------------------------------------
WORKDIR /app
COPY . /app

# --------------------------------------------------------
# Install Python dependencies
# --------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# TRAIN MODELS DURING BUILD
# --------------------------------------------------------
ARG EPOCHS=3

RUN bash -lc '\
    if ls data/BTCUSDT-15m-*.csv >/dev/null 2>&1; then \
      echo "📊 Training ML models using CSV files from /app/data ..."; \
      python scripts/train_models.py --data_dir data/ --epochs ${EPOCHS}; \
    else \
      echo "⚠️ No CSV files found in /app/data - skipping model training."; \
    fi'

# --------------------------------------------------------
# START BOT
# --------------------------------------------------------
CMD ["python", "main.py"]