"""Render filled shapes from the JSON segment payload exported for functions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    """Convert ``#RRGGBB`` colors to OpenCV's BGR channel order."""
    if len(color) != 7 or not color.startswith("#"):
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    return (blue, green, red)


def _cartesian_to_pixel(
    point: dict[str, float],
    *,
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int]:
    """Map a Cartesian segment endpoint back into image pixels."""
    u = point["x"] / scale + width / 2
    v = height / 2 - point["y"] / scale
    return (int(round(u)), int(round(v)))


def _shape_order(payload: dict[str, Any]) -> list[str]:
    """Return shape IDs sorted by their exported z-index."""
    shapes = payload.get("shapes", [])
    ordered = sorted(
        shapes,
        key=lambda shape: int(shape.get("z_index", 0)),
    )
    return [str(shape["shape_id"]) for shape in ordered if "shape_id" in shape]


def _shape_fill_map(payload: dict[str, Any]) -> dict[str, str]:
    """Map each exported shape ID to its fill color."""
    result: dict[str, str] = {}
    for shape in payload.get("shapes", []):
        shape_id = shape.get("shape_id")
        fill = shape.get("fill")
        if shape_id is not None and fill is not None:
            result[str(shape_id)] = str(fill)
    return result


def render_function_segments_payload(payload: dict[str, Any], path: Path) -> None:
    """Fill reconstructed contours from the segment JSON payload."""
    metadata = payload["metadata"]
    width = int(metadata["image_width"])
    height = int(metadata["image_height"])
    scale = float(metadata["scale"])

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for segment in payload.get("segments", []):
        shape_id = str(segment.get("shape_id", segment.get("group", "shape")))
        contour_id = int(segment.get("contour_id", 0))
        grouped[(shape_id, contour_id)].append(segment)

    fill_by_shape = _shape_fill_map(payload)
    order = _shape_order(payload)
    missing_shape_ids = sorted({shape_id for shape_id, _ in grouped} - set(order))
    draw_order = order + missing_shape_ids

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for shape_id in draw_order:
        contours: list[np.ndarray] = []
        color = fill_by_shape.get(shape_id)
        for (group_shape_id, _), segments in grouped.items():
            if group_shape_id != shape_id:
                continue
            if len(segments) < 3:
                continue
            if color is None:
                color = str(segments[0].get("color", "#000000"))
            points = [
                _cartesian_to_pixel(segment["start"], width=width, height=height, scale=scale)
                for segment in segments
            ]
            contours.append(np.array(points, dtype=np.int32).reshape((-1, 1, 2)))

        if contours and color is not None:
            cv2.drawContours(canvas, contours, -1, _hex_to_bgr(color), thickness=-1)

    cv2.imwrite(str(path), canvas)


def render_function_segments_file(payload_path: Path, output_path: Path) -> None:
    """Render a segment JSON file directly to an image."""
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    render_function_segments_payload(payload, output_path)
