from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _cartesian_to_pixel(
    point: dict[str, float],
    *,
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int]:
    u = point["x"] / scale + width / 2
    v = height / 2 - point["y"] / scale
    return (int(round(u)), int(round(v)))


def render_segments(
    segments: list[dict[str, Any]],
    path: Path,
    *,
    image_size: tuple[int, int],
    scale: float,
    line_color: tuple[int, int, int] = (0, 0, 0),
    line_thickness: int = 2,
) -> None:
    width, height = image_size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for segment in segments:
        start = _cartesian_to_pixel(segment["start"], width=width, height=height, scale=scale)
        end = _cartesian_to_pixel(segment["end"], width=width, height=height, scale=scale)
        cv2.line(canvas, start, end, line_color, line_thickness, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def render_filled_regions(
    region_contours: dict[int, list[np.ndarray]],
    label_to_gray: dict[int, int],
    line_segments: list[dict[str, Any]],
    path: Path,
    *,
    image_size: tuple[int, int],
    scale: float,
    base_gray_image: np.ndarray | None = None,
) -> None:
    width, height = image_size
    if base_gray_image is None:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        for label, contours in region_contours.items():
            gray = int(label_to_gray.get(label, 200))
            cv2.drawContours(canvas, contours, -1, (gray, gray, gray), thickness=-1)
    else:
        canvas = cv2.cvtColor(base_gray_image, cv2.COLOR_GRAY2BGR)

    for segment in line_segments:
        start = _cartesian_to_pixel(segment["start"], width=width, height=height, scale=scale)
        end = _cartesian_to_pixel(segment["end"], width=width, height=height, scale=scale)
        cv2.line(canvas, start, end, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
