"""
EasyOCR singleton loader.
Provides run_ocr() which returns (list[str], mean_confidence).
"""
import logging
from typing import Optional

import numpy as np
import easyocr

from config import settings

logger = logging.getLogger(__name__)

_reader: Optional[easyocr.Reader] = None


def get_reader() -> easyocr.Reader:
    """Return (or lazily initialise) the shared EasyOCR reader."""
    global _reader
    if _reader is None:
        logger.info(
            "Loading EasyOCR — languages: %s, GPU: %s",
            settings.OCR_LANGUAGES,
            settings.OCR_GPU,
        )
        _reader = easyocr.Reader(settings.OCR_LANGUAGES, gpu=settings.OCR_GPU)
    return _reader


def run_ocr(img: np.ndarray) -> tuple[list[str], float]:
    """
    Run EasyOCR on a preprocessed image.

    Args:
        img: grayscale or BGR ndarray (already preprocessed).

    Returns:
        (texts, mean_confidence)
        texts            – list of recognised text strings
        mean_confidence  – average per-word confidence (0-1)
    """
    reader = get_reader()
    results = reader.readtext(img, detail=1)
    texts = [r[1] for r in results]
    confs = [r[2] for r in results]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return texts, avg_conf