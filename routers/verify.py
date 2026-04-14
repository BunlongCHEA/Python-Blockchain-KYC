"""
/api/kyc/verify  –  Full KYC pipeline:
  OCR → Face comparison (optional) → DB field match → scored verdict.
"""
import base64
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import settings
from database import match_fields_with_db
from face.verify import verify_faces
from models import KYCVerifyRequest, KYCVerifyResponse
from ocr.extractor_id import extract_cambodian_id_fields
from ocr.extractor_passport import extract_passport_fields
from ocr.reader import run_ocr, run_ocr_mrz
from utils.image import decode_base64_image, preprocess_for_ocr, detect_mrz_zone
from utils.scoring import compute_overall_score

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kyc/verify", tags=["KYC Verification"])

# ── Extracted-field validation ────────────────────────────────────────────────

# Fields that MUST be non-empty and well-formed — pipeline fails without them
REQUIRED_FIELDS: List[str] = ["first_name", "last_name", "date_of_birth", "id_number"]

# All fields we inspect (required + nice-to-have)
VALIDATABLE_FIELDS: List[str] = [
    "first_name", "last_name", "date_of_birth",
    "expiry_date", "id_number", "sex", "nationality",
]

# Regex: at least one Khmer character (Unicode block U+1780–U+17FF)
_RE_KHMER = re.compile(r"[\u1780-\u17FF]")
# Regex: at least one Basic-Latin letter
_RE_LATIN = re.compile(r"[A-Za-z]")
# Regex: any digit
_RE_DIGIT = re.compile(r"\d")
# Regex: common date-like patterns (DD/MM/YYYY, YYYY-MM-DD, DD.MM.YYYY, etc.)
_RE_DATE = re.compile(
    r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"   # 12/04/1999, 12-04-1999
    r"|\d{4}-\d{2}-\d{2}"                         # 1999-04-12
)


def _is_field_valid(key: str, value: str) -> Tuple[bool, str]:
    """
    Return (is_valid, reason) for a single extracted field.

    Checks:
      1. Non-empty
      2. No mixed-script garbling  (Latin + Khmer in one token → camera misread)
      3. Nationality must be alphabetic (reject "18K")
      4. Date fields must look like a real date
      5. Sex must be a single M or F
      6. id_number must be alphanumeric and ≥ 6 chars
    """
    if not value or not value.strip():
        return False, "empty"

    val = value.strip()

    # ── Mixed-script detection (e.g. "AAAកក") ──
    if _RE_LATIN.search(val) and _RE_KHMER.search(val):
        return False, f"mixed-script garbling: '{val}'"

    # ── Per-field rules ──
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
        # Must be at least 2 meaningful characters (letters only)
        letters_only = re.sub(r"[^A-Za-z\u1780-\u17FF]", "", val)
        if len(letters_only) < 2:
            return False, f"too few letters: '{val}'"

    return True, "ok"


def _validate_extracted_fields(
    fields: Dict[str, Any],
    document_type: str,
) -> Tuple[bool, Dict[str, str], List[str]]:
    """
    Validate every field in VALIDATABLE_FIELDS.

    Returns:
        all_required_ok : True when every REQUIRED field passed
        invalid_fields  : {field_name: reason} for every failed field
        missing_required: list of required field names that failed
    """
    invalid_fields: Dict[str, str] = {}
    missing_required: List[str] = []

    for key in VALIDATABLE_FIELDS:
        raw_value = str(fields.get(key, "") or "")
        ok, reason = _is_field_valid(key, raw_value)

        if not ok:
            invalid_fields[key] = reason
            if key in REQUIRED_FIELDS:
                missing_required.append(key)
            logger.warning("[VALIDATE] Field '%s' FAILED — %s", key, reason)
        else:
            logger.info("[VALIDATE] Field '%s' OK — '%s'", key, raw_value)

    all_required_ok = len(missing_required) == 0
    return all_required_ok, invalid_fields, missing_required


# ── Dual-zone OCR helper ─────────────────────────────────────────────────────

def _dual_zone_ocr(
    id_img: np.ndarray,
    id_img_bytes: bytes,
    document_type: str,
) -> Tuple[list, float, dict, list, float]:
    """
    Run OCR twice — full image + cropped MRZ — and merge extracted fields.

    Returns:
        full_texts    : raw text blocks from full-image pass
        full_conf     : avg confidence from full-image pass
        merged_fields : extracted fields (MRZ overrides full where available)
        mrz_texts     : raw text blocks from MRZ-only pass (for debugging)
        mrz_conf      : avg confidence from MRZ-only pass
    """
    # ── Pass 1: Full image (Khmer + English) ──────────────────────────────────
    full_texts, full_conf = run_ocr(id_img_bytes)
    logger.info("[DUAL-OCR] Pass 1 (full): %d blocks, conf=%.4f", len(full_texts), full_conf)

    full_fields = (
        extract_passport_fields(full_texts)
        if document_type == "passport"
        else extract_cambodian_id_fields(full_texts)
    )

    # ── Pass 2: Cropped MRZ (English only) ────────────────────────────────────
    mrz_texts: list = []
    mrz_conf: float = 0.0
    mrz_fields: dict = {}

    mrz_crop = detect_mrz_zone(id_img)
    if mrz_crop is not None and mrz_crop.size > 0:
        # Encode the cropped MRZ region back to bytes for Vision API
        success, mrz_buf = cv2.imencode(".jpg", mrz_crop)
        if success:
            mrz_bytes = mrz_buf.tobytes()
            mrz_texts, mrz_conf = run_ocr_mrz(mrz_bytes)
            logger.info("[DUAL-OCR] Pass 2 (MRZ crop): %d blocks, conf=%.4f",
                        len(mrz_texts), mrz_conf)

            mrz_fields = (
                extract_passport_fields(mrz_texts)
                if document_type == "passport"
                else extract_cambodian_id_fields(mrz_texts)
            )
    else:
        logger.warning("[DUAL-OCR] MRZ zone detection failed — using full-image only")

    # ── Merge: MRZ wins for structured fields, full wins for context ──────────
    all_keys = ("id_number", "first_name", "last_name", "date_of_birth",
                "expiry_date", "sex", "nationality")

    merged: dict = {}
    for key in all_keys:
        mrz_val  = str(mrz_fields.get(key, "") or "").strip()
        full_val = str(full_fields.get(key, "") or "").strip()

        # MRZ value takes priority (cleaner ASCII source)
        if mrz_val:
            merged[key] = mrz_val
            logger.info("[DUAL-MERGE] %s = '%s' (from MRZ crop)", key, mrz_val)
        elif full_val:
            merged[key] = full_val
            logger.info("[DUAL-MERGE] %s = '%s' (from full image)", key, full_val)
        else:
            merged[key] = ""
            logger.info("[DUAL-MERGE] %s = '' (no match in either pass)", key)

    # Use the higher confidence of the two passes
    best_conf = max(full_conf, mrz_conf) if mrz_conf > 0 else full_conf

    return full_texts, best_conf, merged, mrz_texts, mrz_conf


# ── Internal pipeline ─────────────────────────────────────────────────────────

def _run_pipeline(
    customer_id: str,
    document_type: str,
    id_img_bytes:  Optional[bytes],
    id_img: Optional[np.ndarray],
    selfie_img: Optional[np.ndarray],
) -> KYCVerifyResponse:
    """Core KYC pipeline: dual-zone OCR → field validation → face compare → DB match → scoring."""
    now = datetime.now(timezone.utc).isoformat()

    ocr_result:  Optional[dict] = None
    face_result: Optional[dict] = None
    db_match:    dict           = {"db_found": False, "match_score": 0.0}
    ocr_conf:    float          = 0.0
    face_sim:    float          = 0.0
    fields:      dict           = {}

    # ── Step 1: Dual-Zone OCR ─────────────────────────────
    if id_img is not None:
        try:
            full_texts, ocr_conf, fields, mrz_texts, mrz_conf = _dual_zone_ocr(
                id_img, id_img_bytes, document_type,
            )

            ocr_result = {
                "raw_text":         full_texts,
                "extracted_fields": fields,
                "confidence":       round(ocr_conf, 4),
                "document_type":    document_type,
                # Dual-zone debug info
                "mrz_raw_text":     mrz_texts,
                "mrz_confidence":   round(mrz_conf, 4),
                "ocr_strategy":     "dual_zone",
            }

        except Exception as exc:
            logger.error("OCR step failed: %s", exc)
            ocr_result = {"error": str(exc)}

    # ── Step 1b: Validate extracted fields ────────────────
    if ocr_result and "error" not in ocr_result:
        all_ok, invalid_fields, missing_required = _validate_extracted_fields(
            fields, document_type,
        )

        ocr_result["fields_valid"]     = all_ok
        ocr_result["invalid_fields"]   = invalid_fields
        ocr_result["missing_required"] = missing_required

        if not all_ok:
            logger.warning(
                "[PIPELINE] OCR_INCOMPLETE for %s — missing required: %s | invalid: %s",
                customer_id, missing_required, invalid_fields,
            )
            return KYCVerifyResponse(
                customer_id=customer_id,
                document_verified=False,
                face_matched=False,
                ocr_result=ocr_result,
                face_result=None,
                field_match={"db_found": False, "match_score": 0.0},
                overall_score=0.0,
                status="OCR_INCOMPLETE",
                reason=(
                    f"Required fields not extracted correctly: "
                    f"{', '.join(missing_required)}. "
                    f"Please re-scan with better image quality / lighting."
                ),
                timestamp=now,
                score_breakdown={
                    "ocr_confidence":   round(ocr_conf, 4),
                    "mrz_confidence":   round(mrz_conf if 'mrz_conf' in dir() else 0.0, 4),
                    "fields_extracted": len(fields) - len(invalid_fields),
                    "fields_total":     len(VALIDATABLE_FIELDS),
                    "invalid_fields":   invalid_fields,
                    "missing_required": missing_required,
                    "ocr_strategy":     "dual_zone",
                },
            )

    # ── Step 1c: DB field match (only if fields are valid) ─
    if ocr_result and "error" not in ocr_result and customer_id:
        db_match = match_fields_with_db(customer_id, fields)

    # ── Step 2: Face comparison ───────────────────────────
    if selfie_img is not None and id_img is not None:
        try:
            face_result = verify_faces(id_img, selfie_img)
            face_sim    = face_result.get("similarity_score", 0.0)
        except Exception as exc:
            logger.error("Face step failed: %s", exc)
            face_result = {"error": str(exc)}

    # ── Step 3: Scoring ───────────────────────────────────
    score, status, reason = compute_overall_score(
        ocr_confidence=ocr_conf,
        face_similarity=face_sim,
        db_match_score=db_match.get("match_score", 0.0),
        has_selfie=selfie_img is not None,
    )

    # ── Score breakdown (visible in API response) ─────────────────────────────
    has_selfie = selfie_img is not None
    if has_selfie:
        breakdown = {
            "formula":              "db_weighted + face_weighted + ocr_weighted",
            "ocr_confidence":       round(ocr_conf, 4),
            "ocr_weighted":         round(ocr_conf * 0.35 * 100, 2),
            "face_similarity":      round(face_sim, 4),
            "face_weighted":        round((face_sim / 100) * 0.40 * 100, 2),
            "db_match_score":       round(db_match.get("match_score", 0.0), 4),
            "db_weighted":          round(db_match.get("match_score", 0.0) * 0.25 * 100, 2),
            "db_matched_fields":    db_match.get("matched_fields", {}),
            "db_found":             db_match.get("db_found", False),
            "overall_score":        score,
            "threshold_verified":   settings.SCORE_VERIFIED,
            "threshold_review":     settings.SCORE_NEEDS_REVIEW,
            "ocr_strategy":         "dual_zone",
        }
    else:
        breakdown = {
            "formula":              "db_weighted + ocr_weighted (no face_weighted since no selfie)",
            "ocr_confidence":       round(ocr_conf, 4),
            "ocr_weighted":         round(ocr_conf * 0.60 * 100, 2),
            "face_similarity":      None,
            "face_weighted":        None,
            "db_match_score":       round(db_match.get("match_score", 0.0), 4),
            "db_weighted":          round(db_match.get("match_score", 0.0) * 0.40 * 100, 2),
            "db_matched_fields":    db_match.get("matched_fields", {}),
            "db_found":             db_match.get("db_found", False),
            "overall_score":        score,
            "threshold_verified":   settings.SCORE_VERIFIED,
            "threshold_review":     settings.SCORE_NEEDS_REVIEW,
            "ocr_strategy":         "dual_zone",
        }
        
    # ── Build verified_by: [ocr_engine, face_model, preprocessing] ──────────
    ocr_engine   = "google_vision"                                    # reader.py always uses Cloud Vision
    face_model   = face_result.get("model", settings.FACE_MODEL) if face_result else settings.FACE_MODEL
    preprocessing = face_result.get("preprocessing", "raw") if face_result else "raw"
    verified_by  = f"{ocr_engine}|{face_model}|{preprocessing}"      # e.g. "google_vision|ArcFace|gfpgan_restored"

    # ── Build document_hash: SHA-256 of raw ID image bytes ──────────────────
    document_hash = ""
    if id_img_bytes:
        document_hash = hashlib.sha256(id_img_bytes).hexdigest()      # 64-char hex, fits varchar(64)

    # ── verification_date: Unix timestamp (bigint) ───────────────────────────
    verification_date = int(datetime.now(timezone.utc).timestamp())

    return KYCVerifyResponse(
        customer_id=customer_id,
        document_verified=ocr_conf >= 0.5,
        face_matched=bool(face_result.get("verified", False)) if face_result else False,
        ocr_result=ocr_result,
        face_result=face_result,
        field_match=db_match,
        overall_score=score,
        status=status,
        reason=reason,
        timestamp=now,
        score_breakdown=breakdown,
        verified_by=verified_by,
        document_hash=document_hash,
        verification_date=verification_date,
    )


# ── POST /api/kyc/verify  (JSON base64) ───────────────────────────────────────
@router.post(
    "",
    response_model=KYCVerifyResponse,
    summary="Full KYC verify — base64 JSON",
)
def verify_kyc(body: KYCVerifyRequest):
    """
    Full KYC pipeline via JSON body.
    - id_image_base64 is required (ID card or passport)
    - selfie_image_base64 is optional — enables face comparison
    """
    id_img_bytes: Optional[bytes]    = None
    id_img:     Optional[np.ndarray] = None
    selfie_img: Optional[np.ndarray] = None

    if body.id_image_base64:
        try:
            # id_img = decode_base64_image(body.id_image_base64)
            
            id_img_bytes = base64.b64decode(body.id_image_base64)
            arr          = np.frombuffer(id_img_bytes, dtype=np.uint8)
            id_img       = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"id_image_base64: {exc}")

    if body.selfie_image_base64:
        try:
            selfie_img = decode_base64_image(body.selfie_image_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"selfie_image_base64: {exc}")

    return _run_pipeline(body.customer_id, body.document_type, id_img_bytes, id_img, selfie_img)


# ── POST /api/kyc/verify/upload  (multipart) ──────────────────────────────────
@router.post(
    "/upload",
    response_model=KYCVerifyResponse,
    summary="Full KYC verify — multipart file upload",
)
async def verify_kyc_upload(
    customer_id: str            = Form(...),
    document_type: str          = Form("national_id"),
    id_image: UploadFile        = File(..., description="ID card or passport image"),
    selfie_image: Optional[UploadFile] = File(None, description="Selfie image (optional)"),
):
    """
    Full KYC pipeline accepting file uploads.
    - id_image is required
    - selfie_image is optional — enables face comparison
    """
    # Decode ID image
    # id_data = await id_image.read()
    # ID image — need both bytes (OCR) and cv2 array (face)
    id_img_bytes = await id_image.read()
    id_arr  = np.frombuffer(id_img_bytes, dtype=np.uint8)
    id_img  = cv2.imdecode(id_arr, cv2.IMREAD_COLOR)
    if id_img is None:
        raise HTTPException(status_code=400, detail="Cannot decode id_image file")

    # Decode optional selfie
    selfie_img: Optional[np.ndarray] = None
    if selfie_image:
        selfie_data = await selfie_image.read()
        selfie_arr  = np.frombuffer(selfie_data, dtype=np.uint8)
        selfie_img  = cv2.imdecode(selfie_arr, cv2.IMREAD_COLOR)
        if selfie_img is None:
            raise HTTPException(status_code=400, detail="Cannot decode selfie_image file")

    return _run_pipeline(customer_id, document_type, id_img_bytes, id_img, selfie_img)