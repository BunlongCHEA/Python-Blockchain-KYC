"""
Face verification using DeepFace (ArcFace model + RetinaFace detector).
"""
import logging
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace

from config import settings
from utils.image import save_temp_image

logger = logging.getLogger(__name__)

# Minimum face dimension — if the image is smaller, upscale it.
# ArcFace input is 112×112; we want at least 2× that for the detector.
_MIN_FACE_DIM = 250


# ── Image enhancement helpers ─────────────────────────────────────────────────

def _upscale_if_small(img: np.ndarray, min_dim: int = _MIN_FACE_DIM) -> np.ndarray:
    """
    Upscale an image if its shorter side is below min_dim.
    ID card photos are often tiny (~80×100px) — ArcFace needs more pixels.
    """
    h, w = img.shape[:2]
    short_side = min(h, w)
    if short_side >= min_dim:
        return img

    scale = min_dim / short_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    logger.info("[FACE-PREP] Upscaled from %dx%d → %dx%d (scale=%.2f)", w, h, new_w, new_h, scale)
    return upscaled


def _enhance_id_photo(img: np.ndarray) -> np.ndarray:
    """
    Enhance a printed/laminated ID card photo for better face recognition.

    Steps:
      1. Denoise   — removes print grain and camera noise
      2. CLAHE     — adaptive contrast enhancement (recovers washed-out faces)
      3. Sharpen   — recovers edges lost to printing + re-photographing
    """
    # Work on a copy
    enhanced = img.copy()

    # Step 1: Denoise (preserves edges better than Gaussian blur)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, h=6, hColor=6,
                                                templateWindowSize=7,
                                                searchWindowSize=21)

    # Step 2: CLAHE on L channel (contrast-limited adaptive histogram equalization)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Step 3: Light sharpen
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    logger.info("[FACE-PREP] ID photo enhanced (denoise + CLAHE + sharpen)")
    return enhanced


def _normalize_brightness(img: np.ndarray, target_mean: float = 127.0) -> np.ndarray:
    """
    Normalize image brightness so both ID photo and selfie have similar
    overall luminance — prevents the model from being confused by lighting
    differences between a studio ID photo and a natural-light selfie.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    current_mean = l_channel.mean()
    if current_mean > 0:
        scale = target_mean / current_mean
        l_channel = np.clip(l_channel * scale, 0, 255).astype(np.uint8)

    result = cv2.merge([l_channel, a_channel, b_channel])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    logger.info("[FACE-PREP] Brightness normalized: mean %.1f → %.1f", current_mean, target_mean)
    return result


# ── Core verification ─────────────────────────────────────────────────────────

def _run_deepface(
    id_img: np.ndarray,
    selfie_img: np.ndarray,
    label: str = "default",
) -> Dict[str, Any]:
    """
    Run a single DeepFace.verify() call and return the raw result dict.
    Handles temp file creation/cleanup.
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
            align=True,            # Ensure 5-point landmark alignment
        )
        logger.info("[FACE-%s] distance=%.6f threshold=%.4f verified=%s",
                    label, result.get("distance"), result.get("threshold"),
                    result.get("verified"))
        return result

    except Exception as exc:
        logger.error("[FACE-%s] DeepFace error: %s", label, exc)
        return {"distance": 1.0, "threshold": settings.FACE_THRESHOLD,
                "verified": False, "error": str(exc)}
    finally:
        _unlink(id_path)
        _unlink(selfie_path)


def _compute_similarity(result: dict) -> float:
    """Extract similarity score (0-100) from DeepFace result."""
    distance = float(result.get("distance", 1.0))

    # Use DeepFace's built-in confidence score (0-100) if available.
    # DeepFace >= 0.0.93 returns "confidence" from a pre-trained logistic
    # regression model that properly maps distance → probability.
    confidence = result.get("confidence")

    if confidence is not None:
        return round(float(confidence), 2)

    # Fallback: convert cosine distance to similarity percentage.
    # Cosine distance range for ArcFace is 0.0 (identical) to ~1.0 (opposite).
    return max(0.0, round((1.0 - distance) * 100, 2))


def verify_faces(id_img: np.ndarray, selfie_img: np.ndarray) -> Dict[str, Any]:
    """
    Compare the face in an ID/Passport photo against a selfie image.

    Strategy — two attempts to maximize match quality:
      Attempt 1: Upscaled + enhanced ID  vs  normalized selfie
      Attempt 2: Only if Attempt 1 fails — raw images (no preprocessing)

    The best (lowest distance) result is returned.

    Args:
        id_img:     OpenCV BGR ndarray of the document photo.
        selfie_img: OpenCV BGR ndarray of the selfie.

    Returns:
        Dict with keys: verified, distance, threshold, model,
                        similarity_score, preprocessing, [error]
    """
    attempts: list[Tuple[str, np.ndarray, np.ndarray]] = []

    # ── Attempt 1: Enhanced ID + normalized selfie (recommended) ──
    try:
        enhanced_id = _upscale_if_small(id_img)
        enhanced_id = _enhance_id_photo(enhanced_id)
        enhanced_id = _normalize_brightness(enhanced_id)

        norm_selfie = _normalize_brightness(selfie_img)

        attempts.append(("enhanced", enhanced_id, norm_selfie))
    except Exception as exc:
        logger.warning("[FACE] Enhancement failed, will use raw: %s", exc)

    # ── Attempt 2: Raw images (fallback — no preprocessing) ──
    attempts.append(("raw", id_img, selfie_img))

    # ── Run all attempts, pick the best ───────────────────────────────────────
    best_result: Optional[dict] = None
    best_distance: float = 999.0
    best_label: str = "raw"

    for label, img_id, img_selfie in attempts:
        result = _run_deepface(img_id, img_selfie, label=label)
        dist = float(result.get("distance", 999.0))

        if dist < best_distance:
            best_distance = dist
            best_result = result
            best_label = label

        # If already verified, no need to try more
        if result.get("verified", False):
            best_result = result
            best_label = label
            break

    # ── Build response ────────────────────────────────────────────────────────
    if best_result is None:
        best_result = {"distance": 1.0, "threshold": settings.FACE_THRESHOLD,
                       "verified": False}

    distance   = float(best_result.get("distance", 1.0))
    threshold  = float(best_result.get("threshold", settings.FACE_THRESHOLD))
    similarity = _compute_similarity(best_result)
    verified   = bool(best_result.get("verified", False))

    logger.info(
        "[FACE] BEST: attempt=%s model=%s distance=%.6f threshold=%.4f "
        "verified=%s similarity=%.2f",
        best_label, settings.FACE_MODEL, distance, threshold,
        verified, similarity,
    )

    response: Dict[str, Any] = {
        "verified":         verified,
        "distance":         round(distance, 6),
        "threshold":        threshold,
        "model":            settings.FACE_MODEL,
        "similarity_score": similarity,
        "preprocessing":    best_label,
    }

    if "error" in best_result:
        response["error"] = best_result["error"]

    return response


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# def verify_faces(id_img: np.ndarray, selfie_img: np.ndarray) -> Dict[str, Any]:
#     """
#     Compare the face in an ID/Passport photo against a selfie image.

#     Args:
#         id_img:     OpenCV BGR ndarray of the document photo.
#         selfie_img: OpenCV BGR ndarray of the selfie.

#     Returns:
#         Dict with keys: verified, distance, threshold, model,
#                         similarity_score, [error]
#     """
#     id_path     = save_temp_image(id_img)
#     selfie_path = save_temp_image(selfie_img)

#     try:
#         result = DeepFace.verify(
#             img1_path=id_path,
#             img2_path=selfie_path,
#             model_name=settings.FACE_MODEL,
#             detector_backend=settings.FACE_DETECTOR,
#             enforce_detection=False,
#         )

#         distance  = float(result.get("distance",  1.0))
#         threshold = float(result.get("threshold", settings.FACE_THRESHOLD))
#         # # Similarity: 100 % when distance == 0, 0 % at threshold
#         # similarity = max(0.0, round((1.0 - distance / threshold) * 100, 2))
        
#         # Use DeepFace's built-in confidence score (0-100) if available.
#         # DeepFace >= 0.0.93 returns "confidence" from a pre-trained logistic
#         # regression model that properly maps distance → probability.
#         confidence = result.get("confidence")

#         if confidence is not None:
#             similarity = round(float(confidence), 2)
#         else:
#             # Fallback: convert cosine distance to similarity percentage.
#             # Cosine distance range for ArcFace is 0.0 (identical) to ~1.0 (opposite).
#             # Formula: similarity = (1 - distance) * 100
#             #   distance=0.0   → 100%  (identical)
#             #   distance=0.68  →  32%  (at threshold)
#             #   distance=1.0   →   0%  (completely different)
#             similarity = max(0.0, round((1.0 - distance) * 100, 2))

#         logger.info(
#             "[FACE] model=%s distance=%.6f threshold=%.4f verified=%s similarity=%.2f",
#             settings.FACE_MODEL, distance, threshold,
#             result.get("verified"), similarity,
#         )

#         return {
#             "verified":         bool(result.get("verified", False)),
#             "distance":         round(distance, 6),
#             "threshold":        threshold,
#             "model":            settings.FACE_MODEL,
#             "similarity_score": similarity,
#         }

#     except Exception as exc:
#         logger.error("DeepFace verification error: %s", exc)
#         return {
#             "verified":         False,
#             "distance":         1.0,
#             "threshold":        settings.FACE_THRESHOLD,
#             "model":            settings.FACE_MODEL,
#             "similarity_score": 0.0,
#             "error":            str(exc),
#         }
#     finally:
#         _unlink(id_path)
#         _unlink(selfie_path)


# def _unlink(path: str) -> None:
#     try:
#         os.unlink(path)
#     except OSError:
#         pass