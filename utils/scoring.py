"""
KYC overall score computation and verdict mapping.
"""
from config import settings


def compute_overall_score(
    ocr_confidence: float,
    face_similarity: float,
    db_match_score: float,
    has_selfie: bool,
) -> tuple[float, str, str]:
    """
    Compute a 0-100 overall KYC score and return (score, status, reason).

    Weights:
    - With selfie:   OCR 30 % | Face 40 % | DB 30 %
    - Without selfie: OCR 50 % | DB 50 %

    Status thresholds (configurable via env):
    - >= SCORE_VERIFIED (default 80)  → VERIFIED
    - >= SCORE_NEEDS_REVIEW (default 50) → NEEDS_REVIEW
    - else → REJECTED
    """
    if has_selfie:
        raw = (
            ocr_confidence          * 0.30
            + (face_similarity / 100) * 0.40
            + db_match_score          * 0.30
        )
    else:
        raw = ocr_confidence * 0.50 + db_match_score * 0.50

    score = round(min(raw * 100, 100.0), 2)

    if score >= settings.SCORE_VERIFIED:
        return score, "VERIFIED", "All checks passed"
    elif score >= settings.SCORE_NEEDS_REVIEW:
        return score, "NEEDS_REVIEW", "Score is borderline — manual review required"
    else:
        return score, "REJECTED", "Verification score too low"