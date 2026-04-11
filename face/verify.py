"""
Face verification using DeepFace (ArcFace model + RetinaFace detector).

Enhancement pipeline for ID-card-vs-selfie:
  1. GFPGAN face restoration  — AI-reconstructs facial details from low-res ID photo
  2. Upscale small images     — ensures minimum pixel density for RetinaFace
  3. Denoise + CLAHE + sharpen — recovers detail from printed/scanned photos
  4. Brightness normalization  — aligns lighting between ID photo and selfie
  5. Multi-attempt strategy    — tries enhanced, restored, and raw images

NOTE: We do NOT pre-crop faces manually — RetinaFace (the detector_backend)
already does face detection + alignment internally better than any manual crop.
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


# ══════════════════════════════════════════════════════════════════════════════
# GPU Device Helper
# ══════════════════════════════════════════════════════════════════════════════

def _get_torch_device() -> str:
    """Return 'cuda' if GPU available and enabled, else 'cpu'."""
    if settings.GPU_AVAILABLE:
        return "cuda"
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 1: GFPGAN Face Restoration (best for faces)
# ══════════════════════════════════════════════════════════════════════════════

_gfpgan_restorer = None
_gfpgan_available = None  # None = not checked yet


def _get_gfpgan():
    """Lazy-load GFPGAN. Returns None if not installed."""
    global _gfpgan_restorer, _gfpgan_available

    if _gfpgan_available is False:
        return None
    if _gfpgan_restorer is not None:
        return _gfpgan_restorer

    try:
        from gfpgan import GFPGANer

        device = _get_torch_device()
        model_path = os.getenv("GFPGAN_MODEL_PATH", "GFPGANv1.4.pth")

        _gfpgan_restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,       # <-- GPU or CPU
        )
        _gfpgan_available = True
        logger.info("[GFPGAN] Loaded on %s", device.upper())
        return _gfpgan_restorer

    except ImportError:
        logger.warning("[GFPGAN] Not installed - skipping")
        _gfpgan_available = False
        return None
    except Exception as exc:
        logger.warning("[GFPGAN] Load failed: %s - skipping", exc)
        _gfpgan_available = False
        return None


def _restore_face_gfpgan(img: np.ndarray) -> Optional[np.ndarray]:
    """Use GFPGAN to restore facial details. Returns None if unavailable."""
    restorer = _get_gfpgan()
    if restorer is None:
        return None

    try:
        _, _, restored = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        if restored is not None:
            logger.info("[GFPGAN] Restored: %dx%d → %dx%d",
                        img.shape[1], img.shape[0],
                        restored.shape[1], restored.shape[0])
            return restored

        logger.warning("[GFPGAN] No face detected — skipping")
        return None
    except Exception as exc:
        logger.warning("[GFPGAN] Enhancement failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 2: Real-ESRGAN Super-Resolution (good general upscaling + face fix)
# ══════════════════════════════════════════════════════════════════════════════

_realesrgan_upsampler = None
_realesrgan_available = None


def _get_realesrgan():
    """Lazy-load Real-ESRGAN. Returns None if not installed."""
    global _realesrgan_upsampler, _realesrgan_available

    if _realesrgan_available is False:
        return None
    if _realesrgan_upsampler is not None:
        return _realesrgan_upsampler

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        device = _get_torch_device()
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)

        model_path = os.getenv("REALESRGAN_MODEL_PATH", "RealESRGAN_x2plus.pth")

        # half=True on GPU (faster), half=False on CPU (required)
        use_half = settings.GPU_AVAILABLE

        _realesrgan_upsampler = RealESRGANer(
            scale=2,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=use_half,       # <-- FP16 on GPU, FP32 on CPU
            device=device,       # <-- GPU or CPU
        )
        _realesrgan_available = True
        logger.info("[REALESRGAN] Loaded on %s (half=%s)", device.upper(), use_half)
        return _realesrgan_upsampler

    except ImportError:
        logger.warning("[REALESRGAN] Not installed - skipping")
        _realesrgan_available = False
        return None
    except Exception as exc:
        logger.warning("[REALESRGAN] Load failed: %s - skipping", exc)
        _realesrgan_available = False
        return None


def _upscale_realesrgan(img: np.ndarray) -> Optional[np.ndarray]:
    """Use Real-ESRGAN to upscale image with AI super-resolution."""
    upsampler = _get_realesrgan()
    if upsampler is None:
        return None

    try:
        output, _ = upsampler.enhance(img, outscale=2)
        if output is not None:
            logger.info("[REALESRGAN] Upscaled: %dx%d → %dx%d",
                        img.shape[1], img.shape[0],
                        output.shape[1], output.shape[0])
            return output

        logger.warning("[REALESRGAN] Enhancement returned None")
        return None
    except Exception as exc:
        logger.warning("[REALESRGAN] Enhancement failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 3: OpenCV Enhancement (always CPU - no GPU benefit)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Core Verification
# ══════════════════════════════════════════════════════════════════════════════

def _run_deepface(
    id_img: np.ndarray,
    selfie_img: np.ndarray,
    label: str = "default",
) -> Dict[str, Any]:
    """
    Run DeepFace.verify().

    GPU note: DeepFace automatically uses GPU if PyTorch/TensorFlow
    detects CUDA. No explicit device= parameter needed - it reads
    from the underlying framework (torch.cuda.is_available()).
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
        logger.info("[FACE-%s] distance=%.6f threshold=%.4f verified=%s (device=%s)",
                    label, result.get("distance"), result.get("threshold"),
                    result.get("verified"), _get_torch_device())
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

    Multi-attempt strategy (best lowest-distance wins):
      1. GFPGAN restored    — AI face reconstruction (if installed)
      2. Real-ESRGAN upscaled — AI super-resolution  (if installed)
      3. OpenCV enhanced     — upscale + CLAHE + denoise (always works)
      4. Raw                 — no preprocessing (baseline)

    Args:
        id_img:     OpenCV BGR ndarray of the document photo.
        selfie_img: OpenCV BGR ndarray of the selfie.

    Returns:
        Dict with keys: verified, distance, threshold, model,
                        similarity_score, preprocessing, [error]
    """
    attempts: List[Tuple[str, np.ndarray, np.ndarray]] = []

    # Normalize selfie once (shared across attempts that need it)
    try:
        norm_selfie = _normalize_brightness(selfie_img)
    except Exception:
        norm_selfie = selfie_img

    # ── Attempt 1: GFPGAN face restoration (best for faces) ──
    try:
        restored = _restore_face_gfpgan(id_img)
        if restored is not None:
            restored = _normalize_brightness(restored)
            attempts.append(("gfpgan_restored", restored, norm_selfie))
    except Exception as exc:
        logger.warning("[FACE] GFPGAN attempt prep failed: %s", exc)

    # ── Attempt 2: Real-ESRGAN super-resolution ──
    try:
        upscaled = _upscale_realesrgan(id_img)
        if upscaled is not None:
            upscaled = _enhance_id_photo(upscaled)
            upscaled = _normalize_brightness(upscaled)
            attempts.append(("realesrgan_enhanced", upscaled, norm_selfie))
    except Exception as exc:
        logger.warning("[FACE] Real-ESRGAN attempt prep failed: %s", exc)

    # ── Attempt 3: OpenCV enhanced (always available) ──
    # try:
    #     enhanced = _upscale_if_small(id_img)
    #     enhanced = _enhance_id_photo(enhanced)
    #     enhanced = _normalize_brightness(enhanced)
    #     attempts.append(("opencv_enhanced", enhanced, norm_selfie))
    # except Exception as exc:
    #     logger.warning("[FACE] OpenCV enhancement failed: %s", exc)

    # ── Attempt 4: Raw (always available) ──
    attempts.append(("raw", id_img, selfie_img))

    # ── Run all attempts, pick the best ──
    best_result: Optional[dict] = None
    best_distance: float = 999.0
    best_label: str = "raw"
    all_attempts: List[dict] = []

    for label, img_id, img_selfie in attempts:
        result = _run_deepface(img_id, img_selfie, label=label)
        dist = float(result.get("distance", 999.0))

        all_attempts.append({
            "strategy":   label,
            "distance":   round(dist, 6),
            "verified":   result.get("verified", False),
            "similarity": _compute_similarity(result),
        })

        logger.info("[FACE] Attempt '%s': distance=%.6f (best so far=%.6f)",
                    label, dist, best_distance)

        if dist < best_distance:
            best_distance = dist
            best_result = result
            best_label = label

        # Early stop if very confident
        if dist < 0.30:
            logger.info("[FACE] High confidence (distance < 0.30) — stopping early")
            break

    # ── Build response ──
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
        "device":           _get_torch_device(),
        "all_attempts":     all_attempts,
    }

    if "error" in best_result:
        response["error"] = best_result["error"]

    return response


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass