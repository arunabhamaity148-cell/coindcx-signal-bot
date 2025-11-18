FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

CMD ["python", "main.py"]