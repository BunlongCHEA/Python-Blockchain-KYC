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
    status: str           # VERIFIED | REJECTED | NEEDS_REVIEW
    reason: str
    timestamp: str