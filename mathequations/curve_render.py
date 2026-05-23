"""Preview rendering for mixed line-art equation segments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

Point = tuple[float, float]


def _payload_to_point(payload: dict[str, float]) -> Point:
    return (float(payload["x"]), float(payload["y"]))


def sample_segment_points(segment: dict[str, Any], *, samples: int = 48) -> list[Point]:
    """Sample a mixed equation segment into Cartesian points."""
    kind = segment["type"]
    if kind == "linear":
        x_min = float(segment["restriction"]["min"])
        x_max = float(segment["restriction"]["max"])
        m = float(segment["m"])
        b = float(segment["b"])
        return [(float(x), float(m * x + b)) for x in np.linspace(x_min, x_max, samples)]
    if kind == "vertical":
        y_min = float(segment["restriction"]["min"])
        y_max = float(segment["restriction"]["max"])
        c = float(segment["c"])
        return [(c, float(y)) for y in np.linspace(y_min, y_max, samples)]
    if kind == "quadratic":
        x_min = float(segment["restriction"]["min"])
        x_max = float(segment["restriction"]["max"])
        a = float(segment["a"])
        b = float(segment["b"])
        c = float(segment["c"])
        return [(float(x), float(a * x * x + b * x + c)) for x in np.linspace(x_min, x_max, samples)]
    if kind == "bezier_cubic":
        p0, p1, p2, p3 = [_payload_to_point(point) for point in segment["control_points"]]
        result: list[Point] = []
        for t in np.linspace(0.0, 1.0, samples):
            x = (
                (1 - t) ** 3 * p0[0]
                + 3 * (1 - t) ** 2 * t * p1[0]
                + 3 * (1 - t) * t**2 * p2[0]
                + t**3 * p3[0]
            )
            y = (
                (1 - t) ** 3 * p0[1]
                + 3 * (1 - t) ** 2 * t * p1[1]
                + 3 * (1 - t) * t**2 * p2[1]
                + t**3 * p3[1]
            )
            result.append((float(x), float(y)))
        return result
    raise ValueError(f"Unsupported segment type: {kind}")


def _cartesian_to_pixel(point: Point, *, width: int, height: int, scale: float) -> tuple[int, int]:
    x, y = point
    u = x / scale + width / 2
    v = height / 2 - y / scale
    return (int(round(u)), int(round(v)))


def render_curve_segments(
    segments: list[dict[str, Any]],
    path: Path,
    *,
    image_size: tuple[int, int],
    scale: float,
    line_thickness: int = 1,
) -> None:
    """Render sampled mixed curve segments on a white preview canvas."""
    width, height = image_size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for segment in segments:
        points = sample_segment_points(segment)
        pixels = [_cartesian_to_pixel(point, width=width, height=height, scale=scale) for point in points]
        if len(pixels) >= 2:
            cv2.polylines(
                canvas,
                [np.array(pixels, dtype=np.int32).reshape((-1, 1, 2))],
                isClosed=False,
                color=(0, 0, 0),
                thickness=max(1, int(line_thickness)),
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(str(path), canvas)


def render_stroke_paths(
    strokes: list[Any],
    path: Path,
    *,
    image_size: tuple[int, int],
    line_thickness: int = 1,
) -> None:
    """Render traced pixel-space strokes before curve fitting."""
    width, height = image_size
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for stroke in strokes:
        pixels = np.array([(round(x), round(y)) for x, y in stroke.points], dtype=np.int32)
        if len(pixels) >= 2:
            cv2.polylines(
                canvas,
                [pixels.reshape((-1, 1, 2))],
                isClosed=bool(stroke.closed),
                color=(0, 0, 0),
                thickness=max(1, int(line_thickness)),
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(str(path), canvas)
