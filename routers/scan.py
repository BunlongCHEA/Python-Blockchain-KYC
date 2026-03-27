"""
/api/kyc/scan  –  Document OCR endpoints.
Accepts base64 JSON body or multipart file upload.
"""
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import base64
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models import OCRScanResult, ScanRequest
from ocr.reader import run_ocr, run_ocr_mrz
from ocr.extractor_id import extract_cambodian_id_fields
from ocr.extractor_passport import extract_passport_fields
from utils.image import decode_base64_image, detect_mrz_zone, preprocess_for_ocr

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kyc/scan", tags=["OCR / Scan"])


# ── Field validation (shared logic with verify.py) ────────────────────────────

REQUIRED_FIELDS: List[str] = ["first_name", "last_name", "date_of_birth", "id_number"]

VALIDATABLE_FIELDS: List[str] = [
    "first_name", "last_name", "date_of_birth",
    "expiry_date", "id_number", "sex", "nationality",
]

_RE_KHMER = re.compile(r"[\u1780-\u17FF]")
_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_DIGIT = re.compile(r"\d")
_RE_DATE  = re.compile(
    r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
)


def _is_field_valid(key: str, value: str) -> Tuple[bool, str]:
    """Return (is_valid, reason) for a single extracted field."""
    if not value or not value.strip():
        return False, "empty"

    val = value.strip()

    if _RE_LATIN.search(val) and _RE_KHMER.search(val):
        return False, f"mixed-script garbling: '{val}'"

    if key == "nationality":
        if _RE_DIGIT.search(val):
            return False, f"contains digits: '{val}'"
        if len(val) < 2:
            return False, f"too short: '{val}'"

    if key in ("date_of_birth", "expiry_date"):
        if not _RE_DATE.search(val):
            return False, f"not a recognisable date: '{val}'"

    if key == "sex":
        if val.upper() not in ("M", "F"):
            return False, f"invalid sex value: '{val}'"

    if key == "id_number":
        cleaned = re.sub(r"[\s\-]", "", val)
        if len(cleaned) < 6:
            return False, f"too short ({len(cleaned)} chars): '{val}'"
        if not re.match(r"^[A-Za-z0-9]+$", cleaned):
            return False, f"non-alphanumeric chars: '{val}'"

    if key in ("first_name", "last_name"):
        letters_only = re.sub(r"[^A-Za-z\u1780-\u17FF]", "", val)
        if len(letters_only) < 2:
            return False, f"too few letters: '{val}'"

    return True, "ok"


def _validate_fields(fields: Dict[str, Any]) -> Tuple[bool, Dict[str, str], List[str]]:
    """Validate extracted fields. Returns (all_required_ok, invalid_fields, missing_required)."""
    invalid_fields: Dict[str, str] = {}
    missing_required: List[str] = []

    for key in VALIDATABLE_FIELDS:
        raw_value = str(fields.get(key, "") or "")
        ok, reason = _is_field_valid(key, raw_value)
        if not ok:
            invalid_fields[key] = reason
            if key in REQUIRED_FIELDS:
                missing_required.append(key)
            logger.warning("[SCAN-VALIDATE] Field '%s' FAILED — %s", key, reason)
        else:
            logger.info("[SCAN-VALIDATE] Field '%s' OK — '%s'", key, raw_value)

    return len(missing_required) == 0, invalid_fields, missing_required


# ── Dual-zone OCR processing ─────────────────────────────────────────────────

def _process(img_bytes: bytes, document_type: str) -> OCRScanResult:
    """
    Dual-zone OCR: full image + cropped MRZ → merge → validate.

    Args:
        img_bytes     : raw JPEG/PNG bytes
        document_type : "national_id" or "passport"

    Returns:
        OCRScanResult with full_text, mrz_text, merged fields, and validation.
    """
    # ── Pass 1: Full image (Khmer + English) ──────────────────────────────────
    full_texts, full_conf = run_ocr(img_bytes)
    logger.info("[SCAN] Pass 1 (full): %d blocks, conf=%.4f", len(full_texts), full_conf)

    full_fields = (
        extract_passport_fields(full_texts)
        if document_type == "passport"
        else extract_cambodian_id_fields(full_texts)
    )

    # ── Pass 2: Cropped MRZ (English only) ────────────────────────────────────
    mrz_texts: list = []
    mrz_conf: float = 0.0
    mrz_fields: dict = {}

    # Decode bytes → numpy for OpenCV MRZ detection
    arr    = np.frombuffer(img_bytes, dtype=np.uint8)
    id_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if id_img is not None:
        mrz_crop = detect_mrz_zone(id_img)
        if mrz_crop is not None and mrz_crop.size > 0:
            success, mrz_buf = cv2.imencode(".jpg", mrz_crop)
            if success:
                mrz_bytes = mrz_buf.tobytes()
                mrz_texts, mrz_conf = run_ocr_mrz(mrz_bytes)
                logger.info("[SCAN] Pass 2 (MRZ crop): %d blocks, conf=%.4f",
                            len(mrz_texts), mrz_conf)

                mrz_fields = (
                    extract_passport_fields(mrz_texts)
                    if document_type == "passport"
                    else extract_cambodian_id_fields(mrz_texts)
                )
        else:
            logger.warning("[SCAN] MRZ zone detection failed — using full-image only")
    else:
        logger.error("[SCAN] Could not decode image bytes to numpy array")

    # ── Merge: MRZ wins for structured fields ─────────────────────────────────
    all_keys = ("id_number", "first_name", "last_name", "date_of_birth",
                "expiry_date", "sex", "nationality")

    merged: dict = {}
    field_sources: dict = {}
    for key in all_keys:
        mrz_val  = str(mrz_fields.get(key, "") or "").strip()
        full_val = str(full_fields.get(key, "") or "").strip()

        if mrz_val:
            merged[key] = mrz_val
            field_sources[key] = "mrz_crop"
        elif full_val:
            merged[key] = full_val
            field_sources[key] = "full_image"
        else:
            merged[key] = ""
            field_sources[key] = "not_found"

    best_conf = max(full_conf, mrz_conf) if mrz_conf > 0 else full_conf

    # ── Validate extracted fields ─────────────────────────────────────────────
    all_ok, invalid_fields, missing_required = _validate_fields(merged)

    return OCRScanResult(
        raw_text=full_texts,
        extracted_fields=merged,
        confidence=round(best_conf, 4),
        document_type=document_type,
        # Dual-zone additions
        mrz_raw_text=mrz_texts,
        mrz_confidence=round(mrz_conf, 4),
        field_sources=field_sources,
        ocr_strategy="dual_zone",
        # Validation additions
        fields_valid=all_ok,
        invalid_fields=invalid_fields,
        missing_required=missing_required,
    )


# ── POST /api/kyc/scan  (JSON base64) ─────────────────────────────────────────
@router.post("", response_model=OCRScanResult, summary="Scan document — base64 JSON")
def scan_document(body: ScanRequest):
    """
    Accepts a base64-encoded image of an ID card or passport and returns
    OCR-extracted fields (Khmer / English) using dual-zone strategy.

    Response includes:
    - `extracted_fields` — merged from full image + MRZ crop
    - `field_sources` — which zone each field came from
    - `fields_valid` / `invalid_fields` / `missing_required` — validation
    """
    try:
        # Decode base64 → raw bytes
        img = base64.b64decode(body.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _process(img, body.document_type)


# ── POST /api/kyc/scan/upload  (multipart) ────────────────────────────────────
@router.post(
    "/upload",
    response_model=OCRScanResult,
    summary="Scan document — multipart file upload",
)
async def scan_document_upload(
    file: UploadFile          = File(..., description="ID card or passport image file"),
    document_type: str        = Form("national_id", description="'national_id' or 'passport'"),
):
    """
    Accepts an image file (JPEG / PNG) via multipart/form-data and returns
    OCR-extracted fields using dual-zone strategy.
    """
    img = await file.read()
    if not img:
        raise HTTPException(status_code=400, detail="Cannot read uploaded image file")

    return _process(img, document_type)