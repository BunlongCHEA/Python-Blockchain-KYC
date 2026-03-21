"""
EasyOCR singleton loader.
Provides run_ocr() which returns (list[str], mean_confidence).
"""
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

def run_ocr(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[list[str], float]:
    """
    Send image bytes to Google Cloud Vision TEXT_DETECTION.
    Language hints: Khmer (km) + English (en) for MRZ/labels.

    Returns:
        texts      : list[str]  — one string per text block/paragraph
        confidence : float      — average confidence (0.0–1.0)
    """
    try:
        client = _get_client()

        image   = vision.Image(content=image_bytes)
        context = vision.ImageContext(language_hints=_LANGUAGE_HINTS)

        # DOCUMENT_TEXT_DETECTION is better than TEXT_DETECTION for ID cards:
        # - preserves reading order
        # - returns per-word confidence scores
        # - handles dense text layouts (ID cards)
        response = client.document_text_detection(
            image=image,
            image_context=context,
        )

        if response.error.message:
            logger.error("Vision API error: %s", response.error.message)
            return [], 0.0

        annotation = response.full_text_annotation
        if not annotation:
            logger.warning("Vision API returned no text annotation")
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
        logger.info("Cloud Vision OCR: %d blocks, avg_conf=%.4f", len(texts), avg_conf)
        return texts, avg_conf

    except Exception as exc:
        logger.error("Google Cloud Vision OCR failed: %s", exc)
        return [], 0.0


# ***************************************************
# Google Document AI OCR

# def _get_client() -> documentai.DocumentProcessorServiceClient:
#     opts        = ClientOptions(api_endpoint=f"{_LOCATION}-documentai.googleapis.com")
#     credentials = _get_credentials()

#     if credentials:
#         return documentai.DocumentProcessorServiceClient(
#             credentials=credentials,
#             client_options=opts,
#         )
#     # ADC path
#     return documentai.DocumentProcessorServiceClient(client_options=opts)

# def run_ocr(image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[list[str], float]:
#     """
#     Send image bytes to Google Document AI OCR processor.

#     Returns:
#         texts      : list[str]  — one string per detected text block
#         confidence : float      — average confidence across all blocks (0.0-1.0)

#     Keeps the same (texts, confidence) return signature as the old
#     pytesseract run_ocr so scan.py / verify.py need minimal changes.
#     """
#     if not all([_PROJECT_ID, _LOCATION, _PROCESSOR_ID]):
#         logger.error("Missing GOOGLE_PROJECT_ID / GOOGLE_DOCAI_LOCATION / GOOGLE_DOCAI_PROCESSOR_ID")
#         return [], 0.0

#     try:
#         # → asia-southeast1-documentai.googleapis.com
#         # → projects/6****8/locations/asia-southeast1/processors/<document-ai-procid>
#         client         = _get_client()
#         processor_name = client.processor_path(_PROJECT_ID, _LOCATION, _PROCESSOR_ID)

#         request = documentai.ProcessRequest(
#             name=processor_name,
#             raw_document=documentai.RawDocument(
#                 content=image_bytes,
#                 mime_type=mime_type,
#             ),
#         )

#         result   = client.process_document(request=request)
#         document = result.document

#         texts       = []
#         confidences = []

#         for page in document.pages:
#             for block in page.blocks:
#                 segs = block.layout.text_anchor.text_segments
#                 conf = block.layout.confidence
#                 text = "".join(
#                     document.text[s.start_index: s.end_index] for s in segs
#                 ).strip()
#                 if text:
#                     texts.append(text)
#                     confidences.append(conf)

#         avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
#         logger.info("Google Doc AI OCR: %d blocks, avg_conf=%.4f", len(texts), avg_conf)
#         return texts, avg_conf

#     except Exception as exc:
#         logger.error("Google Document AI OCR failed: %s", exc)
#         return [], 0.0


# ***************************************************
# Tesseract Code

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