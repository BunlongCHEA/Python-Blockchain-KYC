"""
OCR reader using Pytesseract (Tesseract engine) — Khmer + English.

Drop-in replacement for ocr/reader.py:
  run_ocr()     → full image, lang=khm+eng  (Khmer text + labels)
  run_ocr_mrz() → MRZ crop,   lang=eng      (ASCII-only: A-Z, 0-9, <)

Returns the same (list[str], float) signature as the Cloud Vision reader.

Prerequisites:
  sudo apt-get install tesseract-ocr tesseract-ocr-khm   # engine + Khmer data
  pip install pytesseract Pillow
"""
import io
import logging
from collections import defaultdict

import pytesseract
from PIL import Image
from pytesseract import Output

logger = logging.getLogger(__name__)

# extractor_id  = "tesseract"
EXTRACTOR_ID = "pytesseract"

# ── Language / config presets ─────────────────────────────────────────────────

# Full card: Khmer first, Latin fallback for labels/MRZ
_LANG_FULL = "khm+eng"

# MRZ strip: pure ASCII — prevents Khmer misreads on <<< chevrons
_LANG_MRZ  = "eng"

# OEM 3 = LSTM neural engine  |  PSM 6 = assume a uniform block of text
_CONFIG    = "--oem 3 --psm 6"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Convert raw JPEG/PNG bytes → PIL Image in RGB mode."""
    img = Image.open(io.BytesIO(image_bytes))
    return img.convert("RGB")


def _run_tesseract_ocr(
    image_bytes: bytes,
    lang: str,
    label: str = "full",
) -> tuple[list[str], float]:
    """
    Shared Tesseract OCR call.

    Args:
        image_bytes : raw JPEG/PNG bytes
        lang        : Tesseract language string, e.g. "khm+eng" or "eng"
        label       : logging label ("full" or "mrz")

    Returns:
        (texts, avg_confidence)
          texts          : list[str] — one string per block (mirrors Vision reader)
          avg_confidence : float 0.0-1.0
    """
    try:
        pil_img = _image_from_bytes(image_bytes)

        data = pytesseract.image_to_data(
            pil_img,
            lang=lang,
            config=_CONFIG,
            output_type=Output.DICT,
        )

        # Group words by block_num — mirrors Vision API's paragraph-level blocks
        blocks: dict[int, list[str]] = defaultdict(list)
        block_confs: dict[int, list[float]] = defaultdict(list)

        n = len(data["text"])
        for i in range(n):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])
            block_num = data["block_num"][i]

            if word and conf > 0:          # conf == -1 means no recognition
                blocks[block_num].append(word)
                block_confs[block_num].append(conf / 100.0)  # normalise → 0-1

        texts:       list[str] = []
        confidences: list[float] = []

        for block_num in sorted(blocks):
            block_text = " ".join(blocks[block_num])
            if block_text:
                texts.append(block_text)
                avg_block_conf = sum(block_confs[block_num]) / len(block_confs[block_num])
                confidences.append(avg_block_conf)

        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        logger.info(
            "Tesseract OCR [%s] lang=%s: %d blocks, avg_conf=%.4f",
            label, lang, len(texts), avg_conf,
        )
        return texts, avg_conf

    except Exception as exc:
        logger.error("Tesseract OCR failed [%s]: %s", label, exc)
        return [], 0.0


# ── Public API (same signature as ocr/reader.py) ──────────────────────────────

def run_ocr(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[list[str], float]:
    """
    Run OCR on the FULL image using Khmer + English language data.
    Captures Khmer script body text AND Latin labels/dates.

    Returns:
        texts      : list[str]  — one string per text block
        confidence : float      — average confidence (0.0-1.0)
    """
    return _run_tesseract_ocr(image_bytes, lang=_LANG_FULL, label="full")


def run_ocr_mrz(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[list[str], float]:
    """
    Run OCR on the CROPPED MRZ strip using English-only language data.
    English-only prevents Khmer misreads on MRZ chevrons (<<<).

    Returns:
        texts      : list[str]  — one string per text block
        confidence : float      — average confidence (0.0-1.0)
    """
    return _run_tesseract_ocr(image_bytes, lang=_LANG_MRZ, label="mrz")