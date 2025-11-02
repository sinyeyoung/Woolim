FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavfilter-dev \
    libswscale-dev libswresample-dev libavutil-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["bash","-lc","gunicorn app:app --bind 0.0.0.0:$PORT"]
