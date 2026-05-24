"""Expand fitted centerline segments into stacked offset expression segments."""

from __future__ import annotations

import math
from typing import Any

from .curve_equations import linear_segment, parametric_polyline_segment, vertical_segment
from .curve_render import sample_segment_points

Point = tuple[float, float]


def offset_indices(
    *,
    max_offsets: int,
    radius_pixels: float,
    offset_step_pixels: float,
) -> list[int]:
    """Return symmetric stack indices with total count bounded by max_offsets."""
    if max_offsets <= 0 or max_offsets % 2 == 0:
        raise ValueError("max_offsets must be an odd positive integer")
    if offset_step_pixels <= 0:
        raise ValueError("offset_step_pixels must be positive")
    half = max_offsets // 2
    usable = min(half, max(0, int(math.floor(radius_pixels / offset_step_pixels))))
    return list(range(-usable, usable + 1))


def _payload_point(payload: dict[str, float]) -> Point:
    return (float(payload["x"]), float(payload["y"]))


def _normal_for_points(start: Point, end: Point) -> Point:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return (0.0, 1.0)
    return (-dy / length, dx / length)


def _shift_point(point: Point, normal: Point, distance: float) -> Point:
    return (point[0] + normal[0] * distance, point[1] + normal[1] * distance)


def _polyline_normals(points: list[Point]) -> list[Point]:
    normals: list[Point] = []
    for index, point in enumerate(points):
        if index == 0:
            normal = _normal_for_points(point, points[1])
        elif index == len(points) - 1:
            normal = _normal_for_points(points[index - 1], point)
        else:
            normal = _normal_for_points(points[index - 1], points[index + 1])
        normals.append(normal)
    return normals


def _offset_polyline(points: list[Point], distance: float) -> list[Point]:
    normals = _polyline_normals(points)
    return [_shift_point(point, normal, distance) for point, normal in zip(points, normals)]


def _annotate(
    segment: dict[str, Any],
    *,
    source_segment: dict[str, Any],
    color_name: str,
    color: str,
    offset_index: int,
    offset_pixels: float,
    offset_distance: float,
    radius_pixels: float,
    scale: float,
) -> dict[str, Any]:
    segment["source_segment_id"] = int(source_segment["segment_id"])
    segment["offset_index"] = int(offset_index)
    segment["offset_pixels"] = float(offset_pixels)
    segment["offset_distance"] = float(offset_distance)
    segment["stroke_radius_pixels"] = float(radius_pixels)
    segment["stroke_radius"] = float(radius_pixels * scale)
    segment["color_name"] = color_name
    segment["color"] = color
    segment["source_type"] = str(source_segment["type"])
    return segment


def _source_points(source: dict[str, Any], normal: Point, distance: float) -> list[Point]:
    return [_shift_point(_payload_point(point), normal, distance) for point in source["source_points"]]


def _offset_linear(source: dict[str, Any], segment_id: int, distance: float) -> dict[str, Any]:
    start = _payload_point(source["start"])
    end = _payload_point(source["end"])
    normal = _normal_for_points(start, end)
    return linear_segment(
        segment_id,
        int(source["stroke_id"]),
        _shift_point(start, normal, distance),
        _shift_point(end, normal, distance),
        fit_error=float(source.get("fit_error", 0.0)),
        source_points=_source_points(source, normal, distance),
    )


def _offset_vertical(source: dict[str, Any], segment_id: int, distance: float) -> dict[str, Any]:
    start = _payload_point(source["start"])
    end = _payload_point(source["end"])
    shifted_start = (start[0] + distance, start[1])
    shifted_end = (end[0] + distance, end[1])
    return vertical_segment(
        segment_id,
        int(source["stroke_id"]),
        shifted_start,
        shifted_end,
        fit_error=float(source.get("fit_error", 0.0)),
        source_points=[shifted_start, shifted_end],
    )


def _offset_as_polyline(source: dict[str, Any], segment_id: int, distance: float) -> dict[str, Any]:
    sampled = sample_segment_points(source, samples=48)
    shifted = _offset_polyline(sampled, distance)
    return parametric_polyline_segment(
        segment_id,
        int(source["stroke_id"]),
        shifted,
        fit_error=float(source.get("fit_error", 0.0)),
        source_points=shifted,
    )


def expand_segment_offsets(
    source: dict[str, Any],
    *,
    radius_pixels: float,
    scale: float,
    offset_step_pixels: float,
    max_offsets: int,
    color_name: str,
    color: str,
    start_segment_id: int,
) -> list[dict[str, Any]]:
    """Expand one fitted segment into a symmetric offset stack."""
    expanded: list[dict[str, Any]] = []
    for offset_number, index in enumerate(
        offset_indices(
            max_offsets=max_offsets,
            radius_pixels=radius_pixels,
            offset_step_pixels=offset_step_pixels,
        )
    ):
        offset_pixels = float(index * offset_step_pixels)
        offset_distance = offset_pixels * scale
        segment_id = start_segment_id + offset_number
        if source["type"] == "linear":
            segment = _offset_linear(source, segment_id, offset_distance)
        elif source["type"] == "vertical":
            segment = _offset_vertical(source, segment_id, offset_distance)
        else:
            segment = _offset_as_polyline(source, segment_id, offset_distance)
        expanded.append(
            _annotate(
                segment,
                source_segment=source,
                color_name=color_name,
                color=color,
                offset_index=index,
                offset_pixels=offset_pixels,
                offset_distance=offset_distance,
                radius_pixels=radius_pixels,
                scale=scale,
            )
        )
    return expanded
