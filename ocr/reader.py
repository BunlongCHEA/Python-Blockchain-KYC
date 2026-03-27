import os
import base64
import json
import logging
# import tempfile

# from typing import Optional
# import numpy as np
# import easyocr

# import numpy as np
# import pytesseract
# from PIL import Image
# from pytesseract import Output

# from google.api_core.client_options import ClientOptions
# from google.cloud import documentai
from google.cloud import vision
from google.oauth2 import service_account

from config import settings

logger = logging.getLogger(__name__)


# Config from .env
# _PROJECT_ID   = os.getenv("GOOGLE_PROJECT_ID", "").strip()
# _LOCATION     = os.getenv("GOOGLE_DOCAI_LOCATION", "us").strip()
# _PROCESSOR_ID = os.getenv("GOOGLE_DOCAI_PROCESSOR_ID", "").strip()
_CREDENTIALS_B64  = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64", "").strip()
_CREDENTIALS_FILE  = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

# Language hints — Khmer first, then English (Latin fallback for MRZ/labels)
_LANGUAGE_HINTS = ["km", "en"]

# MRZ-only hints — pure ASCII/Latin for clean MRZ extraction
_MRZ_LANGUAGE_HINTS = ["en"]


# Google Credential for both Vision API and Document AI:
def _get_credentials():
    """
    Load credentials — priority:
    1. GOOGLE_APPLICATION_CREDENTIALS_BASE64  (base64 JSON string)
    2. GOOGLE_APPLICATION_CREDENTIALS         (file path)
    3. Application Default Credentials        (gcloud login / GCE metadata)
    """
    if _CREDENTIALS_B64:
        try:
            json_bytes   = base64.b64decode(_CREDENTIALS_B64)
            info         = json.loads(json_bytes)
            credentials  = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            logger.info("Google credentials loaded from base64 env var")
            return credentials
        except Exception as exc:
            logger.error("Failed to load credentials from base64: %s", exc)
            raise

    if _CREDENTIALS_FILE:
        credentials = service_account.Credentials.from_service_account_file(
            _CREDENTIALS_FILE,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        logger.info("Google credentials loaded from file: %s", _CREDENTIALS_FILE)
        return credentials

    # Fall back to ADC (works on GCE / Cloud Run automatically)
    logger.info("Using Application Default Credentials")
    return None


def _get_client() -> vision.ImageAnnotatorClient:
    credentials = _get_credentials()
    if credentials:
        return vision.ImageAnnotatorClient(credentials=credentials)
    return vision.ImageAnnotatorClient()  # ADC


def _run_vision_ocr(
    image_bytes: bytes,
    language_hints: list[str],
    label: str = "full",
) -> tuple[list[str], float]:
    """
    Shared Vision API call — used by both run_ocr() and run_ocr_mrz().

    Args:
        image_bytes    : raw JPEG/PNG bytes
        language_hints : e.g. ["km","en"] for full card, ["en"] for MRZ
        label          : logging label ("full" or "mrz")

    Returns:
        (texts, avg_confidence)
    """
    try:
        client = _get_client()

        image   = vision.Image(content=image_bytes)
        context = vision.ImageContext(language_hints=language_hints)

        # DOCUMENT_TEXT_DETECTION is better than TEXT_DETECTION for ID cards:
        # - preserves reading order
        # - returns per-word confidence scores
        # - handles dense text layouts (ID cards)
        response = client.document_text_detection(
            image=image,
            image_context=context,
        )

        if response.error.message:
            logger.error("Vision API error [%s]: %s", label, response.error.message)
            return [], 0.0

        annotation = response.full_text_annotation
        if not annotation:
            logger.warning("Vision API returned no text annotation [%s]", label)
            return [], 0.0

        texts       = []
        confidences = []

        # Extract paragraph-level blocks (best granularity for ID card fields)
        for page in annotation.pages:
            for block in page.blocks:
                block_text  = ""
                block_confs = []

                for paragraph in block.paragraphs:
                    para_text = ""
                    for word in paragraph.words:
                        word_text = "".join(s.text for s in word.symbols)
                        para_text += word_text + " "
                        if word.confidence > 0:
                            block_confs.append(word.confidence)
                    block_text += para_text.strip() + "\n"

                block_text = block_text.strip()
                if block_text:
                    texts.append(block_text)
                    if block_confs:
                        confidences.append(sum(block_confs) / len(block_confs))

        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        logger.info("Cloud Vision OCR [%s]: %d blocks, avg_conf=%.4f", label, len(texts), avg_conf)
        return texts, avg_conf

    except Exception as exc:
        logger.error("Google Cloud Vision OCR failed [%s]: %s", label, exc)
        return [], 0.0


def run_ocr(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[list[str], float]:
    """
    Send FULL image bytes to Google Cloud Vision.
    Language hints: Khmer (km) + English (en) — captures Khmer text + labels.

    Returns:
        texts      : list[str]  — one string per text block/paragraph
        confidence : float      — average confidence (0.0–1.0)
    """
    return _run_vision_ocr(image_bytes, _LANGUAGE_HINTS, label="full")


def run_ocr_mrz(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[list[str], float]:
    """
    Send CROPPED MRZ image bytes to Google Cloud Vision.
    Language hints: English only — MRZ is pure ASCII (A-Z, 0-9, <).

    Using English-only hints prevents the OCR engine from misinterpreting
    MRZ chevrons (<<<) as Khmer characters, which produces garbled fields.

    Returns:
        texts      : list[str]  — one string per text block/paragraph
        confidence : float      — average confidence (0.0–1.0)
    """
    return _run_vision_ocr(image_bytes, _MRZ_LANGUAGE_HINTS, label="mrz")