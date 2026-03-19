"""
Image encode / decode utilities shared across the service.
"""
import base64
import os
import tempfile

import cv2
import numpy as np


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
    to improve EasyOCR accuracy on identity documents.
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