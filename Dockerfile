# ------------ BASE IMAGE ------------
FROM python:3.11-slim

# ------------ SYSTEM DEPENDENCIES ------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------ WORKDIR ------------
WORKDIR /app

# ------------ COPY PROJECT FILES ------------
COPY . /app

# ------------ INSTALL PYTHON DEPENDENCIES ------------
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# ------------ EXPOSE PORT ------------
EXPOSE 8000

# ------------ START COMMAND (main.py -> main:app) ------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]