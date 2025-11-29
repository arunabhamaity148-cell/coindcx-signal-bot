FROM python:3.11-slim

# set working folder
WORKDIR /app

# copy every file into container
COPY . .

# install python deps
RUN pip install --no-cache-dir -r requirements.txt

# start bot
CMD ["python", "bot.py"]
