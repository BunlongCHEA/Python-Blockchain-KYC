"""
Centralised settings loaded from environment variables / .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()


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
    OCR_LANGUAGES: list[str] = os.getenv("OCR_LANGUAGES", "km,en").split(",")
    OCR_GPU: bool = os.getenv("OCR_GPU", "false").lower() == "true"

    # ── Scoring thresholds ────────────────────────────────
    SCORE_VERIFIED: float    = float(os.getenv("SCORE_VERIFIED",    80.0))
    SCORE_NEEDS_REVIEW: float = float(os.getenv("SCORE_NEEDS_REVIEW", 50.0))


settings = Settings()