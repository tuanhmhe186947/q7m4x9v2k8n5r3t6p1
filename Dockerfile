FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIG_BEHAVIOR_PT_MODEL_PATH=/app/models/behavior/pig_behavior_sequence.pt \
    PIG_BEHAVIOR_DETECT_MODEL_PATH=/app/models/detector/pig_detector_yolo.pt \
    PIG_BEHAVIOR_VIDEO_PATH=/app/data/videos/pigs101219_full.mp4 \
    PIG_BEHAVIOR_API_HOST=0.0.0.0 \
    PIG_BEHAVIOR_API_PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /app/models/behavior /app/models/detector /app/data/videos /app/outputs

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[pt]"

EXPOSE 8000

CMD ["uvicorn", "pig_behavior.api:app", "--host", "0.0.0.0", "--port", "8000"]
