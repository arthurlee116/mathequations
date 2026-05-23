"""Fit traced line-art stroke paths into equation segments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .curve_equations import (
    bezier_cubic_segment,
    linear_segment,
    parametric_polyline_segment,
    quadratic_segment,
    vertical_segment,
)
from .skeleton_graph import StrokePath

Point = tuple[float, float]


def _is_x_monotonic(points: list[Point]) -> bool:
    xs = [point[0] for point in points]
    return all(a <= b for a, b in zip(xs, xs[1:])) or all(a >= b for a, b in zip(xs, xs[1:]))


def _has_quadratic_x_support(points: list[Point]) -> bool:
    return len({round(point[0], 9) for point in points}) >= 3


def _mean_error(points: list[Point], predicted: list[Point]) -> float:
    if not points:
        return 0.0
    return float(
        np.mean(
            [
                math.hypot(point[0] - estimate[0], point[1] - estimate[1])
                for point, estimate in zip(points, predicted)
            ]
        )
    )


def _linear_error(points: list[Point]) -> tuple[bool, float, float, float]:
    xs = np.array([point[0] for point in points], dtype=float)
    ys = np.array([point[1] for point in points], dtype=float)
    if float(xs.max() - xs.min()) < 1e-9:
        x_value = float(xs.mean())
        error = float(np.mean(np.abs(xs - x_value)))
        return True, 0.0, x_value, error
    m, b = np.polyfit(xs, ys, 1)
    predicted = m * xs + b
    error = float(np.mean(np.abs(ys - predicted)))
    return False, float(m), float(b), error


def _quadratic_error(points: list[Point]) -> tuple[tuple[float, float, float], float]:
    xs = np.array([point[0] for point in points], dtype=float)
    ys = np.array([point[1] for point in points], dtype=float)
    a, b, c = np.polyfit(xs, ys, 2)
    predicted = a * xs * xs + b * xs + c
    return (float(a), float(b), float(c)), float(np.mean(np.abs(ys - predicted)))


def _chord_parameters(points: list[Point]) -> np.ndarray:
    distances = [0.0]
    for a, b in zip(points, points[1:]):
        distances.append(distances[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = distances[-1]
    if total == 0:
        return np.linspace(0.0, 1.0, len(points))
    return np.array([distance / total for distance in distances], dtype=float)


def _fit_bezier(points: list[Point]) -> tuple[tuple[Point, Point, Point, Point], float]:
    p0 = points[0]
    p3 = points[-1]
    t = _chord_parameters(points)
    matrix = []
    rhs_x = []
    rhs_y = []
    for point, value in zip(points, t):
        b0 = (1 - value) ** 3
        b1 = 3 * (1 - value) ** 2 * value
        b2 = 3 * (1 - value) * value**2
        b3 = value**3
        matrix.append([b1, b2])
        rhs_x.append(point[0] - b0 * p0[0] - b3 * p3[0])
        rhs_y.append(point[1] - b0 * p0[1] - b3 * p3[1])
    controls_x, *_ = np.linalg.lstsq(np.array(matrix), np.array(rhs_x), rcond=None)
    controls_y, *_ = np.linalg.lstsq(np.array(matrix), np.array(rhs_y), rcond=None)
    p1 = (float(controls_x[0]), float(controls_y[0]))
    p2 = (float(controls_x[1]), float(controls_y[1]))
    predicted = []
    for value in t:
        x = (
            (1 - value) ** 3 * p0[0]
            + 3 * (1 - value) ** 2 * value * p1[0]
            + 3 * (1 - value) * value**2 * p2[0]
            + value**3 * p3[0]
        )
        y = (
            (1 - value) ** 3 * p0[1]
            + 3 * (1 - value) ** 2 * value * p1[1]
            + 3 * (1 - value) * value**2 * p2[1]
            + value**3 * p3[1]
        )
        predicted.append((float(x), float(y)))
    return (p0, p1, p2, p3), _mean_error(points, predicted)


def fit_points_to_segment(
    segment_id: int,
    stroke_id: int,
    points: list[Point],
    *,
    fit_mode: str,
) -> dict[str, Any]:
    """Fit one stroke piece into the simplest useful equation segment."""
    if len(points) < 2:
        raise ValueError("at least two points are required")
    start = points[0]
    end = points[-1]
    is_vertical, _m, _b_or_x, linear_error_value = _linear_error(points)
    endpoints_vertical = abs(end[0] - start[0]) <= 1e-9
    if is_vertical or linear_error_value <= 0.04 or fit_mode == "linear":
        if is_vertical or endpoints_vertical:
            return vertical_segment(segment_id, stroke_id, start, end, fit_error=linear_error_value, source_points=points)
        return linear_segment(segment_id, stroke_id, start, end, fit_error=linear_error_value, source_points=points)
    if (
        fit_mode in {"mixed", "quadratic"}
        and len(points) >= 3
        and _is_x_monotonic(points)
        and _has_quadratic_x_support(points)
    ):
        coefficients, quadratic_error_value = _quadratic_error(points)
        if quadratic_error_value <= min(0.04, linear_error_value * 0.55) or fit_mode == "quadratic":
            xs = [point[0] for point in points]
            return quadratic_segment(
                segment_id,
                stroke_id,
                coefficients=coefficients,
                x_range=(min(xs), max(xs)),
                endpoints=(start, end),
                fit_error=quadratic_error_value,
                source_points=points,
            )
    control_points, bezier_error_value = _fit_bezier(points)
    return bezier_cubic_segment(
        segment_id,
        stroke_id,
        control_points=control_points,
        fit_error=bezier_error_value,
        source_points=points,
    )


def _split_points(points: list[Point], piece_count: int) -> list[list[Point]]:
    if piece_count <= 1 or len(points) <= 2:
        return [points]
    indices = np.linspace(0, len(points) - 1, piece_count + 1)
    pieces: list[list[Point]] = []
    for start_raw, end_raw in zip(indices[:-1], indices[1:]):
        start = int(round(start_raw))
        end = int(round(end_raw))
        if end <= start:
            end = min(len(points) - 1, start + 1)
        piece = points[start : end + 1]
        if len(piece) >= 2:
            pieces.append(piece)
    return pieces


def _resample_by_arclength(points: list[Point], *, max_samples: int = 80) -> list[Point]:
    if len(points) <= max_samples:
        return points
    distances = [0.0]
    for start, end in zip(points, points[1:]):
        distances.append(distances[-1] + math.hypot(end[0] - start[0], end[1] - start[1]))
    total = distances[-1]
    if total <= 1e-9:
        return [points[0], points[-1]]
    samples = np.linspace(0.0, total, max_samples)
    result: list[Point] = []
    cursor = 0
    for sample in samples:
        while cursor < len(distances) - 2 and distances[cursor + 1] < sample:
            cursor += 1
        start_distance = distances[cursor]
        end_distance = distances[cursor + 1]
        start = points[cursor]
        end = points[cursor + 1]
        span = end_distance - start_distance
        if span <= 1e-9:
            result.append(start)
            continue
        t = (sample - start_distance) / span
        result.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
    return result


def _turn_angle_degrees(a: Point, b: Point, c: Point) -> float:
    left = (a[0] - b[0], a[1] - b[1])
    right = (c[0] - b[0], c[1] - b[1])
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 180.0
    dot = (left[0] * right[0] + left[1] * right[1]) / (left_norm * right_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _corner_indices(points: list[Point], *, angle_threshold_degrees: float = 125.0) -> list[int]:
    corners: list[int] = []
    for index in range(1, len(points) - 1):
        angle = _turn_angle_degrees(points[index - 1], points[index], points[index + 1])
        if angle < angle_threshold_degrees:
            if not corners or index - corners[-1] > 2:
                corners.append(index)
    return corners


def _split_at_indices(points: list[Point], split_indices: list[int]) -> list[list[Point]]:
    if not split_indices:
        return [points]
    pieces: list[list[Point]] = []
    start = 0
    for index in split_indices:
        piece = points[start : index + 1]
        if len(piece) >= 2:
            pieces.append(piece)
        start = index
    tail = points[start:]
    if len(tail) >= 2:
        pieces.append(tail)
    return pieces


def _polyline_error(points: list[Point], segment: dict[str, Any]) -> float:
    if segment["type"] == "parametric_polyline":
        return 0.0
    return float(segment.get("fit_error", 0.0))


def fit_stroke_chains(
    chains: list[Any],
    *,
    target: int,
    fit_mode: str,
) -> list[dict[str, Any]]:
    """Fit reconstructed centerline chains without fragmenting smooth strokes."""
    if target < 1:
        raise ValueError("target must be at least 1")
    useful = [chain for chain in chains if len(getattr(chain, "points", [])) >= 2]
    if not useful:
        return []

    segments: list[dict[str, Any]] = []
    next_segment_id = 1
    per_chain_budget = max(1, round(target / len(useful)))
    for chain_index, chain in enumerate(useful, start=1):
        stroke_id = int(getattr(chain, "stroke_id", chain_index))
        points = _resample_by_arclength(list(chain.points), max_samples=80)
        corners = _corner_indices(points)
        whole_segment = fit_points_to_segment(next_segment_id, stroke_id, points, fit_mode=fit_mode)

        if len(corners) >= max(4, per_chain_budget):
            segments.append(
                parametric_polyline_segment(
                    next_segment_id,
                    stroke_id,
                    points,
                    fit_error=_polyline_error(points, whole_segment),
                    source_points=list(chain.points),
                )
            )
            next_segment_id += 1
            continue

        if not corners and (
            whole_segment["type"] in {"linear", "vertical", "quadratic"}
            or float(whole_segment.get("fit_error", 0.0)) <= 0.6
        ):
            segments.append(whole_segment)
            next_segment_id += 1
            continue

        pieces = _split_at_indices(points, corners)
        if len(pieces) > max(2, per_chain_budget) and whole_segment["type"] == "bezier_cubic":
            segments.append(
                parametric_polyline_segment(
                    next_segment_id,
                    stroke_id,
                    points,
                    fit_error=float(whole_segment.get("fit_error", 0.0)),
                    source_points=list(chain.points),
                )
            )
            next_segment_id += 1
            continue
        for piece in pieces:
            if len(piece) < 2:
                continue
            segments.append(fit_points_to_segment(next_segment_id, stroke_id, piece, fit_mode=fit_mode))
            next_segment_id += 1
    return segments[: max(1, round(target * 1.15))]


def fit_stroke_paths(
    strokes: list[StrokePath],
    *,
    target: int,
    fit_mode: str,
) -> list[dict[str, Any]]:
    """Fit all strokes while distributing an approximate segment budget by length."""
    useful = [stroke for stroke in strokes if len(stroke.points) >= 2]
    if not useful:
        return []
    total_length = sum(max(1.0, stroke.length) for stroke in useful)
    segments: list[dict[str, Any]] = []
    next_segment_id = 1
    for stroke in useful:
        allocation = max(1, round(target * max(1.0, stroke.length) / total_length))
        allocation = min(allocation, max(1, len(stroke.points) // 3))
        for piece in _split_points(stroke.points, allocation):
            segments.append(
                fit_points_to_segment(
                    next_segment_id,
                    stroke.stroke_id,
                    piece,
                    fit_mode=fit_mode,
                )
            )
            next_segment_id += 1
    return segments[: max(1, round(target * 1.15))]
