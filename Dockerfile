# Dockerfile (final, railway-ready)
FROM python:3.11-slim

ARG EPOCHS=3
ARG TRAIN_IN_BUILD=false
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl git ca-certificates libssl-dev libbz2-dev libffi-dev \
    liblzma-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
    zlib1g-dev libpq-dev gcc g++ make unzip && rm -rf /var/lib/apt/lists/*

# TA-Lib C library (needed by python ta-lib) - optional; keep if using ta-lib
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz \
    && cd /tmp && tar -xzf ta-lib.tar.gz && cd ta-lib \
    && ./configure --prefix=/usr && make && make install \
    && rm -rf /tmp/ta-lib /tmp/ta-lib.tar.gz

# upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# copy project
COPY . /app

# install python deps (use --no-cache-dir to keep image smaller)
RUN pip install --no-cache-dir -r requirements.txt

# create models folder and data folder (if not present)
RUN mkdir -p /app/models /app/data

# Build-time: optional training (controlled by TRAIN_IN_BUILD arg)
# If TRAIN_IN_BUILD is "true" (string), training will run; otherwise skip.
RUN bash -lc ' \
    if [ "${TRAIN_IN_BUILD}" = "true" ]; then \
      if ls data/BTCUSDT-15m-*.csv >/dev/null 2>&1; then \
        echo "📊 Data CSVs present — training models during build (epochs=${EPOCHS})"; \
        python scripts/train_models.py --data_dir data/ --epochs ${EPOCHS}; \
      else \
        echo "⚠️ TRAIN_IN_BUILD=true but no CSVs found in /app/data — skipping training"; \
      fi \
    else \
      echo "ℹ️ TRAIN_IN_BUILD=false — skipping training during build"; \
    fi'

# expose port if your app uses one (optional)
# EXPOSE 8000

# default command
CMD ["python", "main.py"]