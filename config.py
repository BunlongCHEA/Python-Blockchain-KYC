"""
Centralised settings loaded from environment variables / .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

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


settings = Settings()