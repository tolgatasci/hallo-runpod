# syntax=docker/dockerfile:1.4
# Hallo - RunPod Serverless Container (GitHub Build Version)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx git wget \
    && rm -rf /var/lib/apt/lists/*

# Clone Hallo
RUN git clone https://github.com/fudan-generative-vision/hallo.git /app/hallo

# Install Hallo's official requirements
WORKDIR /app/hallo
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --ignore-installed -r requirements.txt

# Install RunPod handler dependencies + missing packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install runpod requests face_alignment

# Fix version conflicts
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --force-reinstall numpy==1.26.4 huggingface_hub==0.21.4

# Download ALL Hallo models from HuggingFace to correct path
# Models: hallo, stable-diffusion-v1-5, motion_module, face_analysis, wav2vec, audio_separator, sd-vae-ft-mse
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('fudan-generative-ai/hallo', local_dir='pretrained_models')"

# Download InsightFace models
RUN python -c "from insightface.app import FaceAnalysis; \
    app = FaceAnalysis(name='buffalo_l'); \
    app.prepare(ctx_id=-1)" || true

# Pre-download face_alignment models
RUN python -c "import face_alignment; \
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')" || true

# Verify setup
RUN python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
RUN ls -la /app/hallo/pretrained_models/

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
