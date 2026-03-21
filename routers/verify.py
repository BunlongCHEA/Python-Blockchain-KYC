"""
/api/kyc/verify  –  Full KYC pipeline:
  OCR → Face comparison (optional) → DB field match → scored verdict.
"""
import base64
import logging
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import settings
from database import match_fields_with_db
from face.verify import verify_faces
from models import KYCVerifyRequest, KYCVerifyResponse
from ocr.extractor_id import extract_cambodian_id_fields
from ocr.extractor_passport import extract_passport_fields
from ocr.reader import run_ocr
from utils.image import decode_base64_image, preprocess_for_ocr
from utils.scoring import compute_overall_score

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kyc/verify", tags=["KYC Verification"])


# ── Internal pipeline ─────────────────────────────────────────────────────────

def _run_pipeline(
    customer_id: str,
    document_type: str,
    id_img_bytes:  Optional[bytes], 
    id_img: Optional[np.ndarray],
    selfie_img: Optional[np.ndarray],
) -> KYCVerifyResponse:
    """Core KYC pipeline: OCR → face compare → DB match → scoring."""
    now = datetime.now(timezone.utc).isoformat()

    ocr_result:  Optional[dict] = None
    face_result: Optional[dict] = None
    db_match:    dict           = {"db_found": False, "match_score": 0.0}
    ocr_conf:    float          = 0.0
    face_sim:    float          = 0.0

    # ── Step 1: OCR ──────────────────────────────────────
    if id_img is not None:
        try:
            # processed = preprocess_for_ocr(id_img)
            # texts, ocr_conf = run_ocr(processed)
            
            texts, ocr_conf = run_ocr(id_img_bytes) # bytes, not numpy

            fields = (
                extract_passport_fields(texts)
                if document_type == "passport"
                else extract_cambodian_id_fields(texts)
            )

            ocr_result = {
                "raw_text":         texts,
                "extracted_fields": fields,
                "confidence":       round(ocr_conf, 4),
                "document_type":    document_type,
            }

            # ── Step 1b: DB field match ───────────────
            if customer_id:
                db_match = match_fields_with_db(customer_id, fields)

        except Exception as exc:
            logger.error("OCR step failed: %s", exc)
            ocr_result = {"error": str(exc)}

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
    id_data = await id_image.read()
    id_arr  = np.frombuffer(id_data, dtype=np.uint8)
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

    return _run_pipeline(customer_id, document_type, id_img, selfie_img)