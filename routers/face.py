"""
/api/kyc/face  –  Face comparison endpoints.
"""
import logging

from fastapi import APIRouter, HTTPException

from models import FaceCompareRequest, FaceResult
from face.verify import verify_faces
from utils.image import decode_base64_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kyc/face", tags=["Face Verification"])


# ── POST /api/kyc/face/compare ────────────────────────────────────────────────
@router.post(
    "/compare",
    response_model=FaceResult,
    summary="Compare ID photo vs selfie",
)
def compare_faces(body: FaceCompareRequest):
    """
    Compare the face extracted from an ID / Passport photo against a
    live selfie image using DeepFace (ArcFace model).

    The pipeline runs two attempts:
      1. Enhanced — upscale ID photo + denoise + CLAHE contrast + normalize brightness
      2. Raw     — original images with no preprocessing

    The best result (lowest distance) is returned.
    The `preprocessing` field in the response shows which attempt won.
    """
    try:
        id_img     = decode_base64_image(body.id_image_base64)
        selfie_img = decode_base64_image(body.selfie_image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = verify_faces(id_img, selfie_img)

    return FaceResult(
        verified=result["verified"],
        distance=result["distance"],
        threshold=result["threshold"],
        model=result["model"],
        similarity_score=result["similarity_score"],
        preprocessing=result.get("preprocessing"),
        device=result.get("device"),
        error=result.get("error"),
    )