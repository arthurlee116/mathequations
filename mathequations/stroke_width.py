"""Stroke width estimation from binary masks."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

Point = tuple[float, float]


def stroke_distance_map(mask: np.ndarray) -> np.ndarray:
    """Return L2 distance to background for a 0/255 foreground mask."""
    binary = (mask > 0).astype(np.uint8)
    return cv2.distanceTransform(binary, cv2.DIST_L2, 5)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def sample_radii(
    points: list[Point],
    distance_map: np.ndarray,
    *,
    min_radius_pixels: float = 1.0,
    max_radius_pixels: float = 10.0,
) -> list[float]:
    """Sample clamped radius values at source-image pixel points."""
    if min_radius_pixels <= 0:
        raise ValueError("min_radius_pixels must be positive")
    if max_radius_pixels < min_radius_pixels:
        raise ValueError("max_radius_pixels must be at least min_radius_pixels")
    height, width = distance_map.shape[:2]
    values: list[float] = []
    for x_raw, y_raw in points:
        x = max(0, min(width - 1, int(round(x_raw))))
        y = max(0, min(height - 1, int(round(y_raw))))
        values.append(_clamp(float(distance_map[y, x]), min_radius_pixels, max_radius_pixels))
    return values


def summarize_radius(
    points: list[Point],
    distance_map: np.ndarray,
    *,
    min_radius_pixels: float = 1.0,
    max_radius_pixels: float = 10.0,
) -> float:
    """Return the median clamped radius for one stroke chain."""
    values = sample_radii(
        points,
        distance_map,
        min_radius_pixels=min_radius_pixels,
        max_radius_pixels=max_radius_pixels,
    )
    if not values:
        return min_radius_pixels
    return float(np.median(np.array(values, dtype=float)))


def stroke_radius_by_id(
    chains: list[Any],
    distance_map: np.ndarray,
    *,
    min_radius_pixels: float = 1.0,
    max_radius_pixels: float = 10.0,
) -> dict[int, float]:
    """Map each chain's stroke id to its median radius in source pixels."""
    result: dict[int, float] = {}
    for chain in chains:
        stroke_id = int(getattr(chain, "stroke_id"))
        points = list(getattr(chain, "points", []))
        result[stroke_id] = summarize_radius(
            points,
            distance_map,
            min_radius_pixels=min_radius_pixels,
            max_radius_pixels=max_radius_pixels,
        )
    return result
