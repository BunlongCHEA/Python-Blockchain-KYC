"""
Centralised settings loaded from environment variables / .env file.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# EasyOCR language code mapping:
# ISO 639-1 code → EasyOCR code
# 'km' (Khmer/Cambodia ISO code) is NOT valid in EasyOCR.
# EasyOCR uses 'kh' for Khmer script.
# _LANG_REMAP = {
#     "km": "kh",   # Khmer: ISO 639-1 → EasyOCR code
# }

# def _parse_ocr_languages(raw: str) -> list[str]:
#     """
#     Parse comma-separated language codes, strip whitespace,
#     and remap any ISO codes that differ from EasyOCR's internal codes.
#     """
#     langs = [l.strip() for l in raw.split(",") if l.strip()]
#     remapped = [_LANG_REMAP.get(l, l) for l in langs]
#     return remapped


def _detect_gpu() -> bool:
    """
    Check if CUDA GPU is actually available.
    Returns True only if PyTorch can see a CUDA device.
    """
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logger.info("[GPU] CUDA available: %s (%.1f GB VRAM)", gpu_name, vram_gb)
        else:
            logger.info("[GPU] CUDA not available - using CPU")
        return available
    except ImportError:
        logger.info("[GPU] PyTorch not installed with CUDA - using CPU")
        return False
    except Exception as exc:
        logger.warning("[GPU] Detection failed: %s - using CPU", exc)
        return False

# def _detect_gpu() -> bool:
#     """
#     Check if CUDA GPU is actually available via PyTorch.
#     """
#     try:
#         import torch
#         available = torch.cuda.is_available()
#         if available:
#             gpu_name = torch.cuda.get_device_name(0)
#             # total_memory works across all PyTorch versions
#             props = torch.cuda.get_device_properties(0)
#             vram_bytes = getattr(props, "total_memory", 0) or getattr(props, "total_mem", 0)
#             vram_gb = vram_bytes / (1024**3) if vram_bytes else 0
#             logger.info("[GPU] CUDA available: %s (%.1f GB VRAM)", gpu_name, vram_gb)
#         else:
#             logger.info("[GPU] CUDA not available - using CPU")
#         return available
#     except ImportError:
#         logger.info("[GPU] PyTorch not installed with CUDA - using CPU")
#         return False
#     except Exception as exc:
#         logger.warning("[GPU] Detection failed: %s - using CPU", exc)
#         return False


class Settings:
    # ── Server ────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 5001))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ── PostgreSQL ────────────────────────────────────────
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", 5433))
    POSTGRES_DB: str   = os.getenv("POSTGRES_DB",   "kyc_blockchain")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "12345")
    
    # ── GPU Acceleration ──────────────────────────────────
    # Set USE_GPU=true in .env to enable CUDA acceleration.
    # Auto-detects: if USE_GPU=true but no CUDA found, falls back to CPU.
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"

    # Actual GPU availability (checked at startup)
    # USE_GPU=true in .env is the user's request;
    # GPU_AVAILABLE is whether CUDA actually works.
    GPU_AVAILABLE: bool = False  # set in __init__

    # ── DeepFace ──────────────────────────────────────────
    FACE_MODEL: str    = os.getenv("FACE_MODEL",    "ArcFace")
    FACE_DETECTOR: str = os.getenv("FACE_DETECTOR", "retinaface")
    FACE_THRESHOLD: float = float(os.getenv("FACE_THRESHOLD", 0.68))

    # ── OCR ───────────────────────────────────────────────
    # NOTE: EasyOCR uses 'kh' for Khmer — NOT 'km'.
    # _parse_ocr_languages() auto-remaps 'km' -> 'kh' as a safety net.
    # OCR_LANGUAGES: list[str] = _parse_ocr_languages(
    #     os.getenv("OCR_LANGUAGES", "kh,en")
    # )
    
    # Tesseract lang string: "khm+eng" supports Khmer + Latin in one pass.
    # Set OCR_LANGUAGES env var to override, e.g. "khm+eng+fra".
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "khm+eng")
    OCR_GPU: bool = os.getenv("OCR_GPU", "false").lower() == "true"

    # ── Scoring thresholds ────────────────────────────────
    SCORE_VERIFIED: float    = float(os.getenv("SCORE_VERIFIED",    80.0))
    SCORE_NEEDS_REVIEW: float = float(os.getenv("SCORE_NEEDS_REVIEW", 50.0))

    def __init__(self):
        if self.USE_GPU:
            self.GPU_AVAILABLE = _detect_gpu()
            if not self.GPU_AVAILABLE:
                logger.warning("[GPU] USE_GPU=true but no CUDA found - falling back to CPU")
        else:
            self.GPU_AVAILABLE = False
            logger.info("[GPU] USE_GPU=false - running on CPU")


settings = Settings()