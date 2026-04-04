# =============================================================
# Python KYC AI Service — Dockerfile
#
# Two build targets:
#   CPU (default):  docker build -t kyc-python .
#   GPU (CUDA):     docker build --build-arg USE_GPU=true -t kyc-python-gpu .
#                   docker build --target gpu -t kyc-python-gpu .
#
# The FROM base-${USE_GPU} pattern is NOT supported by BuildKit
# when the arg value resolves to a stage name dynamically.
# Solution: use explicit named stages + a shared deps stage,
# then select final stage via --target or a build arg at the
# final COPY step.
# =============================================================

# ═══════════════════════════════════════════════════════════════
# STAGE 1a: CPU base
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS base-cpu

# ═══════════════════════════════════════════════════════════════
# STAGE 1b: GPU base — install Python on CUDA image
# ═══════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        && ln -sf /usr/bin/python3.11 /usr/bin/python \
        && ln -sf /usr/bin/pip3 /usr/bin/pip \
        && rm -rf /var/lib/apt/lists/*

# ═══════════════════════════════════════════════════════════════
# STAGE 2: deps-cpu — system deps + Python packages on CPU base
# ═══════════════════════════════════════════════════════════════
FROM base-cpu AS deps-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgomp1 libgl1-mesa-glx \
        tesseract-ocr tesseract-ocr-khm \
        git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN grep -v -E "^(facexlib|gfpgan|realesrgan)" requirements.txt > requirements_base.txt \
    && pip install --no-cache-dir -r requirements_base.txt

RUN git clone --depth 1 https://github.com/XPixelGroup/BasicSR.git /tmp/BasicSR \
    && echo '__version__ = "1.4.2"' > /tmp/BasicSR/basicsr/version.py \
    && echo '__gitsha__ = "unknown"' >> /tmp/BasicSR/basicsr/version.py \
    && printf 'import os\nfrom setuptools import find_packages, setup\n\ndef get_requirements():\n    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")\n    if os.path.exists(req_path):\n        with open(req_path) as f:\n            return [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("-")]\n    return []\n\nsetup(\n    name="basicsr",\n    version="1.4.2",\n    description="BasicSR patched build",\n    packages=find_packages(),\n    include_package_data=True,\n    python_requires=">=3.7",\n    install_requires=get_requirements(),\n)\n' > /tmp/BasicSR/setup.py \
    && rm -f /tmp/BasicSR/pyproject.toml \
    && pip install --no-cache-dir --no-build-isolation /tmp/BasicSR \
    && rm -rf /tmp/BasicSR

RUN grep -E "^(facexlib|gfpgan|realesrgan)" requirements.txt > requirements_face.txt \
    && pip install --no-cache-dir -r requirements_face.txt \
    && rm -f requirements_base.txt requirements_face.txt

RUN mkdir -p /app/models_pretrained \
    && wget -q -O /app/models_pretrained/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
    && wget -q -O /app/models_pretrained/RealESRGAN_x2plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    && echo "Models downloaded:" && ls -lh /app/models_pretrained/

# ═══════════════════════════════════════════════════════════════
# STAGE 3: deps-gpu — same as deps-cpu but on GPU base
# ═════════════════════════════════════════════════��═════════════
FROM base-gpu AS deps-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgomp1 libgl1-mesa-glx \
        tesseract-ocr tesseract-ocr-khm \
        git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN grep -v -E "^(facexlib|gfpgan|realesrgan)" requirements.txt > requirements_base.txt \
    && pip install --no-cache-dir -r requirements_base.txt

RUN git clone --depth 1 https://github.com/XPixelGroup/BasicSR.git /tmp/BasicSR \
    && echo '__version__ = "1.4.2"' > /tmp/BasicSR/basicsr/version.py \
    && echo '__gitsha__ = "unknown"' >> /tmp/BasicSR/basicsr/version.py \
    && printf 'import os\nfrom setuptools import find_packages, setup\n\ndef get_requirements():\n    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")\n    if os.path.exists(req_path):\n        with open(req_path) as f:\n            return [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("-")]\n    return []\n\nsetup(\n    name="basicsr",\n    version="1.4.2",\n    description="BasicSR patched build",\n    packages=find_packages(),\n    include_package_data=True,\n    python_requires=">=3.7",\n    install_requires=get_requirements(),\n)\n' > /tmp/BasicSR/setup.py \
    && rm -f /tmp/BasicSR/pyproject.toml \
    && pip install --no-cache-dir --no-build-isolation /tmp/BasicSR \
    && rm -rf /tmp/BasicSR

RUN grep -E "^(facexlib|gfpgan|realesrgan)" requirements.txt > requirements_face.txt \
    && pip install --no-cache-dir -r requirements_face.txt \
    && rm -f requirements_base.txt requirements_face.txt

RUN mkdir -p /app/models_pretrained \
    && wget -q -O /app/models_pretrained/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
    && wget -q -O /app/models_pretrained/RealESRGAN_x2plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    && echo "Models downloaded:" && ls -lh /app/models_pretrained/

# ═══════════════════════════════════════════════════════════════
# STAGE 4a: Final CPU image  (default — docker build .)
# ═══════════════════════════════════════════════════════════════
FROM deps-cpu AS cpu

ENV GFPGAN_MODEL_PATH=/app/models_pretrained/GFPGANv1.4.pth
ENV REALESRGAN_MODEL_PATH=/app/models_pretrained/RealESRGAN_x2plus.pth
ENV USE_GPU=false

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

WORKDIR /app
COPY . .

EXPOSE 5001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]