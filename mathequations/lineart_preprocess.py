"""Preprocessing helpers for pencil and black-line artwork."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .image_processing import normalize_white_background


def clean_lineart_image(image: np.ndarray, *, white_threshold: int = 246) -> np.ndarray:
    """Normalize paper-like background while preserving visible gray strokes."""
    clean = normalize_white_background(image, white_threshold=white_threshold)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    result = clean.copy()
    result[:, :, 0] = clahe
    result[:, :, 1] = clahe
    result[:, :, 2] = clahe
    result[gray >= white_threshold] = [255, 255, 255]
    return result


def _fixed_mask(gray: np.ndarray, *, threshold: int = 252) -> np.ndarray:
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


def _adaptive_mask(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        9,
    )


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    filtered = np.zeros_like(mask)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= 4 and max(width, height) >= 3:
            filtered[labels == label] = 255
    return filtered


def build_line_mask(image: np.ndarray, *, threshold_mode: str = "auto") -> np.ndarray:
    """Return a binary 0/255 mask for visible line-art strokes."""
    clean = clean_lineart_image(image)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    if threshold_mode == "fixed":
        return _clean_mask(_fixed_mask(gray))
    if threshold_mode == "adaptive":
        return _clean_mask(_adaptive_mask(gray))
    if threshold_mode != "auto":
        raise ValueError("threshold_mode must be one of: auto, fixed, adaptive")
    fixed = _fixed_mask(gray)
    adaptive = _adaptive_mask(gray)
    combined = cv2.bitwise_or(fixed, adaptive)
    return _clean_mask(combined)


def skeletonize_line_mask(mask: np.ndarray) -> np.ndarray:
    """Reduce a line mask to a 0/255 one-pixel skeleton."""
    binary = mask > 0
    skeleton = skeletonize(binary)
    return skeleton.astype(np.uint8) * 255
