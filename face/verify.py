"""
Face verification using DeepFace (ArcFace model + RetinaFace detector).
"""
import logging
import os
from typing import Any, Dict

import numpy as np
from deepface import DeepFace

from config import settings
from utils.image import save_temp_image

logger = logging.getLogger(__name__)


def verify_faces(id_img: np.ndarray, selfie_img: np.ndarray) -> Dict[str, Any]:
    """
    Compare the face in an ID/Passport photo against a selfie image.

    Args:
        id_img:     OpenCV BGR ndarray of the document photo.
        selfie_img: OpenCV BGR ndarray of the selfie.

    Returns:
        Dict with keys: verified, distance, threshold, model,
                        similarity_score, [error]
    """
    id_path     = save_temp_image(id_img)
    selfie_path = save_temp_image(selfie_img)

    try:
        result = DeepFace.verify(
            img1_path=id_path,
            img2_path=selfie_path,
            model_name=settings.FACE_MODEL,
            detector_backend=settings.FACE_DETECTOR,
            enforce_detection=False,
        )

        distance  = float(result.get("distance",  1.0))
        threshold = float(result.get("threshold", settings.FACE_THRESHOLD))
        # # Similarity: 100 % when distance == 0, 0 % at threshold
        # similarity = max(0.0, round((1.0 - distance / threshold) * 100, 2))
        
        # Use DeepFace's built-in confidence score (0-100) if available.
        # DeepFace >= 0.0.93 returns "confidence" from a pre-trained logistic
        # regression model that properly maps distance → probability.
        confidence = result.get("confidence")

        if confidence is not None:
            similarity = round(float(confidence), 2)
        else:
            # Fallback: convert cosine distance to similarity percentage.
            # Cosine distance range for ArcFace is 0.0 (identical) to ~1.0 (opposite).
            # Formula: similarity = (1 - distance) * 100
            #   distance=0.0   → 100%  (identical)
            #   distance=0.68  →  32%  (at threshold)
            #   distance=1.0   →   0%  (completely different)
            similarity = max(0.0, round((1.0 - distance) * 100, 2))

        logger.info(
            "[FACE] model=%s distance=%.6f threshold=%.4f verified=%s similarity=%.2f",
            settings.FACE_MODEL, distance, threshold,
            result.get("verified"), similarity,
        )

        return {
            "verified":         bool(result.get("verified", False)),
            "distance":         round(distance, 6),
            "threshold":        threshold,
            "model":            settings.FACE_MODEL,
            "similarity_score": similarity,
        }

    except Exception as exc:
        logger.error("DeepFace verification error: %s", exc)
        return {
            "verified":         False,
            "distance":         1.0,
            "threshold":        settings.FACE_THRESHOLD,
            "model":            settings.FACE_MODEL,
            "similarity_score": 0.0,
            "error":            str(exc),
        }
    finally:
        _unlink(id_path)
        _unlink(selfie_path)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass