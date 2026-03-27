"""
Image encode / decode utilities shared across the service.
"""
import base64
import logging
import os
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def decode_base64_image(b64: str) -> np.ndarray:
    """
    Decode a base64 string (optionally with data-URI prefix) into an
    OpenCV BGR ndarray.

    Raises:
        ValueError: if the image cannot be decoded.
    """
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image from base64 data")
    return img


def encode_image_base64(img: np.ndarray) -> str:
    """Encode an OpenCV BGR ndarray back to a base64 string."""
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def save_temp_image(img: np.ndarray) -> str:
    """
    Write an ndarray to a temporary JPEG file.

    Returns:
        Absolute path to the temp file — caller is responsible for
        calling os.unlink() when finished.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, img)
    return tmp.name


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Apply grayscale conversion, upscaling, denoising and sharpening
    to improve OCR accuracy on identity documents.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale if the image is too small for reliable OCR
    h, w = gray.shape
    if w < 800:
        scale = 800.0 / w
        gray = cv2.resize(
            gray, (800, int(h * scale)), interpolation=cv2.INTER_CUBIC
        )

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    return gray


# ── MRZ Zone Detection ────────────────────────────────────────────────────────

def detect_mrz_zone(img: np.ndarray, padding: int = 10) -> Optional[np.ndarray]:
    """
    Detect and crop the MRZ (Machine Readable Zone) from an ID card image.

    Strategy:
      1. Grayscale → blackhat morphology (highlights dark text on light card)
      2. Scharr horizontal gradient (MRZ lines are strongly horizontal)
      3. Threshold → morphological close (merge characters into solid bars)
      4. Find contours → pick the bottom-most wide rectangle

    Cambodian National ID MRZ = TD1 format (3 lines × 30 chars),
    located at the bottom ~30-35% of the card.

    Args:
        img     : BGR ndarray of the full ID card
        padding : extra pixels around the detected MRZ box

    Returns:
        Cropped BGR ndarray of the MRZ zone, or None if detection fails.
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Step 1: Blackhat — reveals dark text on light background ──
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        # ── Step 2: Scharr gradient (horizontal emphasis) ──
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = grad_x.min(), grad_x.max()
        if max_val - min_val > 0:
            grad_x = ((grad_x - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            grad_x = grad_x.astype(np.uint8)

        # ── Step 3: Threshold + close gaps between characters ──
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        # Fill small holes
        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=2)

        # ── Step 4: Find contours → pick bottom-most wide rectangle ──
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter: MRZ region should be wide (>50% of card width) and in the
        # bottom 45% of the image
        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / float(ch) if ch > 0 else 0

            # MRZ is very wide and short — aspect ratio > 5
            # Must be at least 50% of card width
            # Must be in the bottom 45% of the card
            if cw > w * 0.50 and aspect > 5 and (y + ch) > h * 0.55:
                candidates.append((x, y, cw, ch))

        if not candidates:
            # Fallback: just take the bottom 35% of the card
            logger.info("[MRZ-CROP] No MRZ contour found — using bottom 35%% fallback")
            y_start = int(h * 0.65)
            cropped = img[max(0, y_start - padding):h, 0:w]
            return cropped

        # Pick the candidate closest to the bottom
        candidates.sort(key=lambda c: c[1] + c[3], reverse=True)
        bx, by, bw, bh = candidates[0]

        # Expand the box vertically to catch all 3 MRZ lines
        # TD1 = 3 lines, so expand by ~2x the detected height
        mrz_top = max(0, by - bh * 2 - padding)
        mrz_bot = min(h, by + bh + padding)
        mrz_left = max(0, bx - padding)
        mrz_right = min(w, bx + bw + padding)

        cropped = img[mrz_top:mrz_bot, mrz_left:mrz_right]

        logger.info(
            "[MRZ-CROP] Detected MRZ zone: x=%d y=%d w=%d h=%d → crop (%d,%d)-(%d,%d)",
            bx, by, bw, bh, mrz_left, mrz_top, mrz_right, mrz_bot,
        )
        return cropped

    except Exception as exc:
        logger.error("[MRZ-CROP] Detection failed: %s", exc)
        return None