# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860

# minimal OS deps + OpenMP runtime for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer cache)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "torch>=2.2,<2.6" --extra-index-url https://download.pytorch.org/whl/cpu

# Copy app code
COPY summarizer.py app.py cli.py ./
COPY start.sh ./
RUN chmod +x start.sh

EXPOSE 7860

# Optional: set the default model via env (must have safetensors)
ENV SUMM_MODEL=facebook/bart-large-cnn

# Toggle runtime warmup (0 = off, 1 = on)
ENV PRELOAD_AT_START=0

# Start the app (with optional warmup)
CMD ["./start.sh"]
