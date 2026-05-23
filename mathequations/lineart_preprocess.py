"""Preprocessing helpers for pencil and black-line artwork."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola
from skimage.morphology import skeletonize

from .image_processing import normalize_white_background


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


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


def prepare_highres_lineart(
    image: np.ndarray,
    *,
    scale: int = 4,
    white_threshold: int = 246,
) -> np.ndarray:
    """Return a high-resolution normalized BGR image for centerline tracing."""
    if scale < 1:
        raise ValueError("scale must be at least 1")
    clean = normalize_white_background(_as_bgr(image), white_threshold=white_threshold)
    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    highres = cv2.resize(clean, None, fx=scale, fy=scale, interpolation=interpolation)
    gray = cv2.cvtColor(highres, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.bilateralFilter(clahe, d=5, sigmaColor=24, sigmaSpace=24)
    result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
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


def _filter_components(mask: np.ndarray, *, min_component_area: int) -> tuple[np.ndarray, int, int]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    removed = 0
    kept = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            filtered[labels == label] = 255
            kept += 1
        else:
            removed += 1
    return filtered, kept, removed


def _ximgproc_local_mask(gray: np.ndarray, *, threshold_mode: str) -> np.ndarray | None:
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is None or not hasattr(ximgproc, "niBlackThreshold"):
        return None
    block_size = max(15, min(gray.shape[:2]) // 8)
    if block_size % 2 == 0:
        block_size += 1
    method = getattr(ximgproc, "BINARIZATION_NIBLACK", 0)
    if threshold_mode == "sauvola":
        method = getattr(ximgproc, "BINARIZATION_SAUVOLA", method)
    try:
        return ximgproc.niBlackThreshold(
            gray,
            255,
            cv2.THRESH_BINARY_INV,
            block_size,
            0.18,
            binarizationMethod=method,
        )
    except (TypeError, cv2.error):
        return None


def build_local_line_mask(
    image: np.ndarray,
    *,
    threshold_mode: str = "sauvola",
    min_component_area: int = 2,
) -> np.ndarray:
    """Return a 0/255 local-threshold mask for Centerline V2."""
    if min_component_area < 1:
        raise ValueError("min_component_area must be at least 1")
    gray = cv2.cvtColor(_as_bgr(image), cv2.COLOR_BGR2GRAY)
    if threshold_mode in {"sauvola", "niblack"}:
        mask = _ximgproc_local_mask(gray, threshold_mode=threshold_mode)
        if mask is None:
            window = max(15, min(gray.shape[:2]) // 8)
            if window % 2 == 0:
                window += 1
            if threshold_mode == "sauvola":
                threshold = threshold_sauvola(gray, window_size=window, k=0.05)
            else:
                threshold = threshold_niblack(gray, window_size=window, k=-0.12)
            mask = np.where(gray < threshold, 255, 0).astype(np.uint8)
    elif threshold_mode == "adaptive":
        mask = _adaptive_mask(gray)
    elif threshold_mode == "fixed":
        mask = _fixed_mask(gray)
    else:
        raise ValueError("threshold_mode must be one of: sauvola, niblack, adaptive, fixed")
    filtered, _, _ = _filter_components(mask, min_component_area=min_component_area)
    return filtered


def line_mask_diagnostics(mask: np.ndarray) -> dict[str, float | int]:
    """Summarize foreground density and connected-component shape for diagnostics."""
    binary = (mask > 0).astype(np.uint8)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    component_sizes = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)]
    median_size = float(np.median(component_sizes)) if component_sizes else 0.0
    return {
        "foreground_density": float(cv2.countNonZero(binary) / binary.size),
        "component_count": len(component_sizes),
        "median_component_size": median_size,
        "removed_speckle_count": 0,
    }


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
