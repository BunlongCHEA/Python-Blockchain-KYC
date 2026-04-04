# =============================================================
# Python KYC AI Service — Dockerfile
#
# Two build targets:
#   CPU (default):  docker build -t kyc-python .
#   GPU (CUDA):     docker build --target gpu -t kyc-python-gpu .
# =============================================================

# ═══════════════════════════════════════════════════════════════
# STAGE 1a: CPU base
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS base-cpu

# ═══════════════════════════════════════════════════════════════
# STAGE 1b: GPU base
# ═══════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# ═══════════════════════════════════════════════════════════════
# STAGE 2: shared-deps — system packages only (no Python yet)
#          Used as a template; actual pip install done per variant
# ═══════════════════════════════════════════════════════════════

# ── CPU: system deps ──────────────────────────────────────────
FROM base-cpu AS system-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgomp1 libgl1 \
        tesseract-ocr tesseract-ocr-khm \
        git wget \
    && rm -rf /var/lib/apt/lists/*

# ── GPU: system deps ──────────────────────────────────────────
FROM base-gpu AS system-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgomp1 libgl1 \
        tesseract-ocr tesseract-ocr-khm \
        git wget \
    && rm -rf /var/lib/apt/lists/*

# ═══════════════════════════════════════════════════════════════
# STAGE 3a: deps-cpu — Python packages on CPU base
# ═══════════════════════════════════════════════════════════════
FROM system-cpu AS deps-cpu

WORKDIR /app

# ── Copy requirements first (own layer for cache) ─────────────
COPY requirements.txt ./requirements.txt

# ── Step 1: base packages (everything except face restoration) ─
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && grep -v -E "^(facexlib|gfpgan|realesrgan|#)" requirements.txt \
       | grep -v "^$" \
       > /tmp/requirements_base.txt \
    && echo "=== Base requirements ===" \
    && cat /tmp/requirements_base.txt \
    && pip install --no-cache-dir -r /tmp/requirements_base.txt

# ── Step 2: fix and install basicsr from source ───────────────
RUN git clone --depth 1 https://github.com/XPixelGroup/BasicSR.git /tmp/BasicSR \
    && echo '__version__ = "1.4.2"' > /tmp/BasicSR/basicsr/version.py \
    && echo '__gitsha__ = "unknown"' >> /tmp/BasicSR/basicsr/version.py \
    && printf '%s\n' \
        'import os' \
        'from setuptools import find_packages, setup' \
        '' \
        'def get_requirements():' \
        '    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")' \
        '    if os.path.exists(req_path):' \
        '        with open(req_path) as f:' \
        '            return [l.strip() for l in f if l.strip() and not l.startswith("#") and not l.startswith("-")]' \
        '    return []' \
        '' \
        'setup(' \
        '    name="basicsr",' \
        '    version="1.4.2",' \
        '    description="BasicSR patched build",' \
        '    packages=find_packages(),' \
        '    include_package_data=True,' \
        '    python_requires=">=3.7",' \
        '    install_requires=get_requirements(),' \
        ')' \
       > /tmp/BasicSR/setup.py \
    && rm -f /tmp/BasicSR/pyproject.toml \
    && pip install --no-cache-dir --no-build-isolation /tmp/BasicSR \
    && rm -rf /tmp/BasicSR

# ── Step 3: face restoration packages (needs basicsr) ─────────
RUN grep -E "^(facexlib|gfpgan|realesrgan)" requirements.txt \
       > /tmp/requirements_face.txt \
    && echo "=== Face requirements ===" \
    && cat /tmp/requirements_face.txt \
    && pip install --no-cache-dir -r /tmp/requirements_face.txt

# ── Step 4: download model weights into layer cache ───────────
RUN mkdir -p /app/models_pretrained \
    && wget -q --show-progress \
        -O /app/models_pretrained/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
    && wget -q --show-progress \
        -O /app/models_pretrained/RealESRGAN_x2plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    && echo "=== Models downloaded ===" \
    && ls -lh /app/models_pretrained/

# ═══════════════════════════════════════════════════════════════
# STAGE 3b: deps-gpu — Python packages on GPU base
# ═══════════════════════════════════════════════════════════════
FROM system-gpu AS deps-gpu

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && grep -v -E "^(facexlib|gfpgan|realesrgan|#)" requirements.txt \
       | grep -v "^$" \
       > /tmp/requirements_base.txt \
    && echo "=== Base requirements ===" \
    && cat /tmp/requirements_base.txt \
    && pip install --no-cache-dir -r /tmp/requirements_base.txt

RUN git clone --depth 1 https://github.com/XPixelGroup/BasicSR.git /tmp/BasicSR \
    && echo '__version__ = "1.4.2"' > /tmp/BasicSR/basicsr/version.py \
    && echo '__gitsha__ = "unknown"' >> /tmp/BasicSR/basicsr/version.py \
    && printf '%s\n' \
        'import os' \
        'from setuptools import find_packages, setup' \
        '' \
        'def get_requirements():' \
        '    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")' \
        '    if os.path.exists(req_path):' \
        '        with open(req_path) as f:' \
        '            return [l.strip() for l in f if l.strip() and not l.startswith("#") and not l.startswith("-")]' \
        '    return []' \
        '' \
        'setup(' \
        '    name="basicsr",' \
        '    version="1.4.2",' \
        '    description="BasicSR patched build",' \
        '    packages=find_packages(),' \
        '    include_package_data=True,' \
        '    python_requires=">=3.7",' \
        '    install_requires=get_requirements(),' \
        ')' \
       > /tmp/BasicSR/setup.py \
    && rm -f /tmp/BasicSR/pyproject.toml \
    && pip install --no-cache-dir --no-build-isolation /tmp/BasicSR \
    && rm -rf /tmp/BasicSR

RUN grep -E "^(facexlib|gfpgan|realesrgan)" requirements.txt \
       > /tmp/requirements_face.txt \
    && echo "=== Face requirements ===" \
    && cat /tmp/requirements_face.txt \
    && pip install --no-cache-dir -r /tmp/requirements_face.txt

RUN mkdir -p /app/models_pretrained \
    && wget -q --show-progress \
        -O /app/models_pretrained/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
    && wget -q --show-progress \
        -O /app/models_pretrained/RealESRGAN_x2plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    && echo "=== Models downloaded ===" \
    && ls -lh /app/models_pretrained/

# ═══════════════════════════════════════════════════════════════
# STAGE 4a: Final CPU image  (default — docker build .)
# ═══════════════════════════════════════════════════════════════
FROM deps-cpu AS cpu

ENV GFPGAN_MODEL_PATH=/app/models_pretrained/GFPGANv1.4.pth
ENV REALESRGAN_MODEL_PATH=/app/models_pretrained/RealESRGAN_x2plus.pth
ENV USE_GPU=false
ENV DEEPFACE_HOME=/app/.deepface

# pre-create the folder with correct ownership during build
RUN mkdir -p /app/.deepface && chmod 777 /app/.deepface

WORKDIR /app
COPY . .

EXPOSE 5001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]

# ═══════════════════════════════════════════════════════════════
# STAGE 4b: Final GPU image  (docker build --target gpu .)
# ═══════════════════════════════════════════════════════════════
FROM deps-gpu AS gpu

ENV GFPGAN_MODEL_PATH=/app/models_pretrained/GFPGANv1.4.pth
ENV REALESRGAN_MODEL_PATH=/app/models_pretrained/RealESRGAN_x2plus.pth
ENV USE_GPU=true
ENV DEEPFACE_HOME=/app/.deepface

# pre-create the folder with correct ownership during build
RUN mkdir -p /app/.deepface && chmod 777 /app/.deepface

WORKDIR /app
COPY . .

EXPOSE 5001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]