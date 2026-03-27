import os
import base64
import json
import logging
# import tempfile

# import numpy as np
# import pytesseract
# from PIL import Image
# from pytesseract import Output

from config import settings

logger = logging.getLogger(__name__)

# _tess_cmd = os.getenv("TESSERACT_CMD", "").strip()
# if _tess_cmd:
#     pytesseract.pytesseract.tesseract_cmd = _tess_cmd
#     logger.debug("Tesseract binary overridden via TESSERACT_CMD: %s", _tess_cmd)


# def run_ocr(img: np.ndarray) -> tuple[list[str], float]:
#     """
#     Run Tesseract OCR on a preprocessed image.

#     Args:
#         img: grayscale or BGR ndarray (already preprocessed by
#              utils.image.preprocess_for_ocr).

#     Returns:
#         (texts, mean_confidence)
#         texts            - list of non-empty recognised word strings
#         mean_confidence  - average per-word confidence in [0, 1]
#     """
#     # Convert ndarray → PIL Image (Tesseract works best with PIL)
#     pil_img = Image.fromarray(img)

#     lang = settings.OCR_LANGUAGES          # e.g. "khm+eng"
#     # PSM 3  = fully automatic page segmentation (default)
#     # OEM 3  = default engine (LSTM + legacy)
#     config = "--psm 3 --oem 3"

#     logger.debug("Running Tesseract — lang: %s, config: %s", lang, config)

#     data = pytesseract.image_to_data(
#         pil_img,
#         lang=lang,
#         config=config,
#         output_type=Output.DICT,
#     )

#     texts: list[str] = []
#     confs: list[float] = []

#     for word, conf in zip(data["text"], data["conf"]):
#         word = word.strip()
#         # Tesseract reports conf == -1 for non-word rows (block/line markers)
#         if word and int(conf) >= 0:
#             texts.append(word)
#             confs.append(int(conf) / 100.0)   # normalise 0-100 → 0-1

#     avg_conf = float(np.mean(confs)) if confs else 0.0

#     logger.debug(
#         "Tesseract result — %d words, avg_conf=%.3f", len(texts), avg_conf
#     )
#     return texts, avg_conf


# ***************************************************
# EasyOCR Code

# _reader: Optional[easyocr.Reader] = None

# def get_reader() -> easyocr.Reader:
#     """Return (or lazily initialise) the shared EasyOCR reader."""
#     global _reader
#     if _reader is None:
#         logger.info(
#             "Loading EasyOCR — languages: %s, GPU: %s",
#             settings.OCR_LANGUAGES,
#             settings.OCR_GPU,
#         )
#         _reader = easyocr.Reader(settings.OCR_LANGUAGES, gpu=settings.OCR_GPU)
#     return _reader


# def run_ocr(img: np.ndarray) -> tuple[list[str], float]:
#     """
#     Run EasyOCR on a preprocessed image.

#     Args:
#         img: grayscale or BGR ndarray (already preprocessed).

#     Returns:
#         (texts, mean_confidence)
#         texts            – list of recognised text strings
#         mean_confidence  – average per-word confidence (0-1)
#     """
#     reader = get_reader()
#     results = reader.readtext(img, detail=1)
#     texts = [r[1] for r in results]
#     confs = [r[2] for r in results]
#     avg_conf = float(np.mean(confs)) if confs else 0.0
#     return texts, avg_conf