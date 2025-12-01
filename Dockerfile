# Use slim Python image for smaller size
FROM python:3.11-slim

# set workdir
WORKDIR /app

# make non-root user
RUN useradd --create-home --shell /bin/bash appuser

# install system deps required by some pip packages (ccxt, cryptography etc.)
# keep apt cache cleanup to reduce image size
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libssl-dev \
    libffi-dev \
    cargo \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# upgrade pip and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# copy application code
COPY . /app

# make sure env file is NOT in image for security (use .dockerignore)
# set permissions & switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# default env (can be overridden via --env-file or -e)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# expose nothing in particular (bot uses outbound HTTP)
# HEALTHCHECK optional - simple TCP not ideal; omitted by default

# start command
CMD ["python", "main.py"]