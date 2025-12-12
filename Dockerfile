# ----------  existing lines  ----------
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive

# install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl git ca-certificates libssl-dev libbz2-dev libffi-dev \
    liblzma-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
    zlib1g-dev libpq-dev gcc g++ make unzip && rm -rf /var/lib/apt/lists/*

# build ta-lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz \
    && cd /tmp && tar -xzf ta-lib.tar.gz && cd ta-lib \
    && ./configure --prefix=/usr && make && make install && rm -rf /tmp/ta-lib*

# python setup
RUN python -m pip install --upgrade pip setuptools wheel
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

# ----------  NEW: train models during build  ----------
RUN python scripts/train_models.py --data data/dummy.csv --epochs 3

# start bot
CMD ["python", "main.py"]
