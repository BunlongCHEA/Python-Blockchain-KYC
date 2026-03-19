"""
/api/kyc/scan  –  Document OCR endpoints.
Accepts base64 JSON body or multipart file upload.
"""
import io
import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models import OCRResult, ScanRequest
from ocr.reader import run_ocr
from ocr.extractor_id import extract_cambodian_id_fields
from ocr.extractor_passport import extract_passport_fields
from utils.image import decode_base64_image, preprocess_for_ocr

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kyc/scan", tags=["OCR / Scan"])


def _process(img: np.ndarray, document_type: str) -> OCRResult:
    processed          = preprocess_for_ocr(img)
    texts, confidence  = run_ocr(processed)

    if document_type == "passport":
        fields = extract_passport_fields(texts)
    else:
        fields = extract_cambodian_id_fields(texts)

    return OCRResult(
        raw_text=texts,
        extracted_fields=fields,
        confidence=round(confidence, 4),
        document_type=document_type,
    )


# ── POST /api/kyc/scan  (JSON base64) ─────────────────────────────────────────
@router.post("", response_model=OCRResult, summary="Scan document — base64 JSON")
def scan_document(body: ScanRequest):
    """
    Accepts a base64-encoded image of an ID card or passport and returns
    OCR-extracted fields (Khmer / English).
    """
    try:
        img = decode_base64_image(body.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _process(img, body.document_type)


# ── POST /api/kyc/scan/upload  (multipart) ────────────────────────────────────
@router.post(
    "/upload",
    response_model=OCRResult,
    summary="Scan document — multipart file upload",
)
async def scan_document_upload(
    file: UploadFile          = File(..., description="ID card or passport image file"),
    document_type: str        = Form("national_id", description="'national_id' or 'passport'"),
):
    """
    Accepts an image file (JPEG / PNG) via multipart/form-data and returns
    OCR-extracted fields.
    """
    data = await file.read()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode uploaded image file")

    return _process(img, document_type)