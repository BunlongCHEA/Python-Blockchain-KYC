"""
KYC overall score computation and verdict mapping.
"""
from config import settings


def compute_overall_score(
    ocr_confidence: float,      # 0.0 – 1.0  (from Vision API)
    face_similarity: float,     # 0.0 – 100.0 (from DeepFace)
    db_match_score: float,      # 0.0 – 1.0  (from DB field matching)
    has_selfie: bool,
) -> tuple[float, str, str]:
    """
    Compute a 0-100 overall KYC score and return (score, status, reason).
    
    Normalisation:
      ocr_confidence  is 0-1   → multiply by 100 to get 0-100 points
      face_similarity is 0-100 → use directly
      db_match_score  is 0-1   → multiply by 100 to get 0-100 points

    Weights WITH selfie   : OCR 35% | Face 40% | DB 25%
    Weights WITHOUT selfie: OCR 60% | DB 40%

    face_similarity is already 0-100 from DeepFace — divide by 100 to normalise.

    Status thresholds (set in .env):
      SCORE_VERIFIED      default 75  (was 80 — relaxed since DB no longer checks encrypted ID)
      SCORE_NEEDS_REVIEW  default 50
    """
    # if has_selfie:
    #     raw = (
    #         ocr_confidence            * 0.35   # OCR quality
    #         + (face_similarity / 100) * 0.40   # face match
    #         + db_match_score          * 0.25   # name + DOB match
    #     )
    # else:
    #     raw = (
    #         ocr_confidence * 0.60
    #         + db_match_score * 0.40
    #     )

    # score = round(min(raw * 100, 100.0), 2)

    if has_selfie:
        score = round(min(
            ocr_confidence  * 100 * 0.35
            + face_similarity     * 0.40
            + db_match_score * 100 * 0.25,
            100.0,
        ), 2)
    else:
        score = round(min(
            ocr_confidence  * 100 * 0.60
            + db_match_score * 100 * 0.40,
            100.0,
        ), 2)

    if score >= settings.SCORE_VERIFIED:
        return score, "VERIFIED", "All checks passed"
    elif score >= settings.SCORE_NEEDS_REVIEW:
        return score, "NEEDS_REVIEW", "Score is borderline — manual review required"
    else:
        return score, "REJECTED", "Verification score too low"