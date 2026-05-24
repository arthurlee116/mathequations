"""Color-specific mask extraction for the koala thick-lineart pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorMaskSpec:
    name: str
    hex_color: str


@dataclass(frozen=True)
class ExtractedColorMask:
    name: str
    hex_color: str
    mask: np.ndarray

    @property
    def pixel_count(self) -> int:
        return int(cv2.countNonZero(self.mask))


KOALA_COLOR_SPECS = {
    "black": ColorMaskSpec("black", "#000000"),
    "yellow": ColorMaskSpec("yellow", "#f5d529"),
    "pink": ColorMaskSpec("pink", "#d24b6a"),
}


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _filter_components(mask: np.ndarray, *, min_component_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            filtered[labels == label] = 255
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)


def _black_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(image)
    red_i = red.astype(np.int16)
    green_i = green.astype(np.int16)
    blue_i = blue.astype(np.int16)
    neutral_dark = (
        (gray < 135)
        & (np.abs(red_i - green_i) < 35)
        & (np.abs(red_i - blue_i) < 35)
    )
    very_dark = gray < 80
    return np.where(neutral_dark | very_dark, 255, 0).astype(np.uint8)


def _yellow_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (hue >= 15) & (hue <= 45) & (saturation > 40) & (value > 130)
    return np.where(mask, 255, 0).astype(np.uint8)


def _pink_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image)
    saturation = hsv[:, :, 1]
    red_i = red.astype(np.int16)
    green_i = green.astype(np.int16)
    blue_i = blue.astype(np.int16)
    red_dominant = (red_i > green_i + 20) & (red_i > blue_i + 10)
    mask = red_dominant & (saturation > 35) & (red > 110)
    return np.where(mask, 255, 0).astype(np.uint8)


def extract_koala_color_masks(
    image: np.ndarray,
    *,
    min_component_area: int = 12,
) -> dict[str, ExtractedColorMask]:
    """Return cleaned black/yellow/pink masks for the koala source image."""
    if min_component_area < 1:
        raise ValueError("min_component_area must be at least 1")
    bgr = _as_bgr(image)
    raw_masks = {
        "black": _black_mask(bgr),
        "yellow": _yellow_mask(bgr),
        "pink": _pink_mask(bgr),
    }
    extracted: dict[str, ExtractedColorMask] = {}
    for name, raw_mask in raw_masks.items():
        cleaned = _filter_components(raw_mask, min_component_area=min_component_area)
        spec = KOALA_COLOR_SPECS[name]
        extracted[name] = ExtractedColorMask(name, spec.hex_color, cleaned)
    return extracted


def require_color_masks(masks: dict[str, ExtractedColorMask]) -> None:
    """Fail when any required koala color mask has no foreground pixels."""
    missing = [
        name
        for name in KOALA_COLOR_SPECS
        if name not in masks or masks[name].pixel_count == 0
    ]
    if missing:
        raise ValueError(f"missing required color masks: {', '.join(missing)}")
