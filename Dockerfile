FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAPERCLUB_FACE_MODELS_DIR=/app/models \
    CAPERCLUB_FACE_DETECTION_MODEL=hog \
    CAPERCLUB_FACE_ENCODING_JITTERS=1 \
    CAPERCLUB_FACE_MATCH_THRESHOLD=0.44 \
    CAPERCLUB_FACE_MIN_SUPPORT=2 \
    CAPERCLUB_FACE_MIN_DETECTION_EDGE=960 \
    CAPERCLUB_FACE_MAX_DETECTION_EDGE=1600 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY . .


EXPOSE 8080

CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker main:app --workers 2 --timeout 120 --bind 0.0.0.0:${PORT:-8000}"]
