# Custom Chatterbox TTS image with PyTorch 2.7+cu128 for Blackwell GPU (sm_120 / RTX 5070 Ti).
# The upstream travisvn/chatterbox-tts-api:latest ships PyTorch 2.6+cu124 which has no
# sm_120 kernel image and crashes with:
#   RuntimeError: CUDA error: no kernel image is available for execution on the device

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

WORKDIR /app

# Install PyTorch 2.7+ with CUDA 12.8 (first, so it isn't overwritten by chatterbox deps)
RUN pip install --upgrade pip && \
    pip install torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Install chatterbox-tts and the FastAPI wrapper deps
# (matches what travisvn/chatterbox-tts-api uses, minus torch)
RUN pip install \
        chatterbox-tts \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        librosa \
        soundfile \
        requests

# Copy the app source from the upstream image layer by layer
# We can't do a multi-stage from a non-distroless image cleanly, so we
# clone the upstream app code from GitHub instead.
RUN git clone --depth=1 https://github.com/travisvn/chatterbox-tts-api.git /tmp/upstream && \
    cp -r /tmp/upstream/app /app/app && \
    cp /tmp/upstream/main.py /app/main.py && \
    cp -r /tmp/upstream/voice-samples /app/voice-samples 2>/dev/null || true && \
    rm -rf /tmp/upstream

# Voice directory (mounted from host at runtime)
RUN mkdir -p /app/voices /app/models

ENV DEVICE=cuda \
    DEFAULT_MODEL=multilingual \
    PORT=8020 \
    MODEL_CACHE_DIR=/app/models \
    VOICE_SAMPLE_PATH=/app/voice-samples/voice-sample.mp3

EXPOSE 8020

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8020"]
