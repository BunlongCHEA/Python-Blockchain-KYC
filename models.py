"""
Pydantic request & response models shared across all routers.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Requests

class ScanRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image (with or without data-URI prefix)")
    document_type: str = Field("national_id", description="'national_id' or 'passport'")


class FaceCompareRequest(BaseModel):
    id_image_base64: str     = Field(..., description="Base64 image of ID/Passport photo page")
    selfie_image_base64: str = Field(..., description="Base64 selfie image")


class KYCVerifyRequest(BaseModel):
    customer_id: str            = Field(..., description="Existing KYC customer ID")
    id_image_base64: Optional[str]     = Field(None, description="Base64 ID/Passport image")
    selfie_image_base64: Optional[str] = Field(None, description="Base64 selfie image")
    document_type: str          = Field("national_id", description="'national_id' or 'passport'")


# Responses

class OCRResult(BaseModel):
    raw_text: List[str]
    extracted_fields: Dict[str, Any]
    confidence: float
    document_type: str
    
    
class OCRScanResult(BaseModel):
    """
    Extended OCR result with dual-zone (full + MRZ crop) info and field validation.
    Used by /api/kyc/scan endpoints.
    """
    # ── Same as OCRResult (full-image pass) ──
    raw_text: List[str] = Field(
        ..., description="Raw text blocks from full-image OCR pass (Khmer + English)"
    )
    extracted_fields: Dict[str, Any] = Field(
        ..., description="Merged fields — MRZ crop values override full-image values"
    )
    confidence: float = Field(
        ..., description="Best confidence from either pass (0.0–1.0)"
    )
    document_type: str

    # ── Dual-zone additions ──
    mrz_raw_text: List[str] = Field(
        default=[], description="Raw text blocks from MRZ-only OCR pass (English)"
    )
    mrz_confidence: float = Field(
        default=0.0, description="Confidence from MRZ-only pass (0.0–1.0)"
    )
    field_sources: Dict[str, str] = Field(
        default={},
        description="Which OCR pass each field came from: 'mrz_crop' | 'full_image' | 'not_found'"
    )
    ocr_strategy: str = Field(
        default="dual_zone", description="OCR strategy used: 'dual_zone' | 'single'"
    )

    # ── Validation additions ──
    fields_valid: bool = Field(
        ..., description="True if ALL required fields (first_name, last_name, date_of_birth, id_number) are valid"
    )
    invalid_fields: Dict[str, str] = Field(
        default={}, description="Map of field_name → failure reason for every invalid field"
    )
    missing_required: List[str] = Field(
        default=[], description="List of required field names that failed validation"
    )


class FaceResult(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str
    similarity_score: float
    error: Optional[str] = None


class KYCVerifyResponse(BaseModel):
    customer_id: str
    document_verified: bool
    face_matched: bool
    ocr_result: Optional[Dict[str, Any]]
    face_result: Optional[Dict[str, Any]]
    field_match: Optional[Dict[str, Any]]
    overall_score: float
    status: str           # VERIFIED | REJECTED | NEEDS_REVIEW | OCR_INCOMPLETE
    reason: str
    timestamp: str
    # Score breakdown for debugging
    score_breakdown: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-component scores used to compute overall_score"
    )