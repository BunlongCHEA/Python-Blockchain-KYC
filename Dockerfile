# =============================================================
# Python KYC AI Service — Dockerfile
#
# Two build targets:
#   CPU (default):  docker build -t kyc-python .
#   GPU (CUDA):     docker build --build-arg USE_GPU=true -t kyc-python-gpu .
#
# basicsr is broken on PyPI (KeyError: '__version__'), so we:
#   1. Clone BasicSR from GitHub
#   2. Replace setup.py with a fixed minimal version
#   3. Write version.py with both __version__ and __gitsha__
#   4. Install from the patched local source
#
# Model weights (GFPGAN ~350MB, RealESRGAN ~64MB) are downloaded
# at build time and cached in the Docker layer.
# =============================================================

ARG USE_GPU=false

# ── Base image selection
# CPU: python:3.11-slim (~150MB)
# GPU: nvidia/cuda with Python (~4GB, includes CUDA runtime)
FROM python:3.11-slim AS base-cpu
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base-gpu-prep

# GPU base needs Python installed
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        && ln -sf /usr/bin/python3.11 /usr/bin/python \
        && ln -sf /usr/bin/pip3 /usr/bin/pip \
        && rm -rf /var/lib/apt/lists/*

FROM base-gpu-prep AS base-gpu

# ── Select base based on build arg
FROM base-${USE_GPU} AS base
# base-false = base-cpu (python:3.11-slim)
# base-true  = base-gpu (nvidia/cuda + python)

# ── System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgomp1 libgl1-mesa-glx \
        tesseract-ocr tesseract-ocr-khm \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies (layer cache)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# Step 1: Install everything EXCEPT face restoration packages

# ── Install everything EXCEPT face restoration packages
RUN grep -v -E "^(facexlib|gfpgan|realesrgan)" requirements.txt > requirements_base.txt \
    && pip install --no-cache-dir -r requirements_base.txt

# ── Fix and install basicsr (PyPI version is broken)
# Clone BasicSR, replace broken setup.py, install from source
RUN git clone --depth 1 https://github.com/XPixelGroup/BasicSR.git /tmp/BasicSR \
    && echo '__version__ = "1.4.2"' > /tmp/BasicSR/basicsr/version.py \
    && echo '__gitsha__ = "unknown"' >> /tmp/BasicSR/basicsr/version.py \
    && printf 'import os\n\
from setuptools import find_packages, setup\n\
\n\
def get_requirements():\n\
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")\n\
    if os.path.exists(req_path):\n\
        with open(req_path) as f:\n\
            return [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("-")]\n\
    return []\n\
\n\
setup(\n\
    name="basicsr",\n\
    version="1.4.2",\n\
    description="BasicSR patched build",\n\
    packages=find_packages(),\n\
    include_package_data=True,\n\
    python_requires=">=3.7",\n\
    install_requires=get_requirements(),\n\
)\n' > /tmp/BasicSR/setup.py \
    && rm -f /tmp/BasicSR/pyproject.toml \
    && pip install --no-cache-dir --no-build-isolation /tmp/BasicSR \
    && rm -rf /tmp/BasicSR

# ── Install face restoration packages
# RUN pip install --no-cache-dir facexlib>=0.2.5 gfpgan>=1.3.8 realesrgan>=0.3.0

# ── Now install face restoration packages (basicsr already available)
RUN grep -E "^(facexlib|gfpgan|realesrgan)" requirements.txt > requirements_face.txt \
    && pip install --no-cache-dir -r requirements_face.txt \
    && rm -f requirements_base.txt requirements_face.txt

# ── Download pre-trained model weights
# These are cached in the Docker layer so containers start instantly
RUN mkdir -p /app/models_pretrained \
    && wget -q --show-progress -O /app/models_pretrained/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
    && wget -q --show-progress -O /app/models_pretrained/RealESRGAN_x2plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    && echo "Models downloaded:" \
    && ls -lh /app/models_pretrained/

# ── Set model path environment variables
ENV GFPGAN_MODEL_PATH=/app/models_pretrained/GFPGANv1.4.pth
ENV REALESRGAN_MODEL_PATH=/app/models_pretrained/RealESRGAN_x2plus.pth

# ── Copy source code
COPY . .

EXPOSE 5001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]