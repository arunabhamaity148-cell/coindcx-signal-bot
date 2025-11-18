# use slim image but install build deps so pip can build packages from source if needed
FROM python:3.10-slim

# install system build deps (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy everything
COPY . /app

# upgrade pip & install requirements (no prefer-binary)
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# run
CMD ["python", "main.py"]