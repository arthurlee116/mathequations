"""Equation records for line-art curve segments."""

from __future__ import annotations

from typing import Any

from .equations import format_number

Point = tuple[float, float]


def _point_payload(point: Point) -> dict[str, float]:
    return {"x": float(point[0]), "y": float(point[1])}


def linear_segment(
    segment_id: int,
    stroke_id: int,
    start: Point,
    end: Point,
    *,
    fit_error: float = 0.0,
    source_points: list[Point] | None = None,
) -> dict[str, Any]:
    x1, y1 = start
    x2, y2 = end
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    sign = "+" if b >= 0 else ""
    latex = (
        f"y={format_number(m)}x{sign}{format_number(b)}"
        f"\\left\\{{{format_number(x_min)}\\le x\\le {format_number(x_max)}\\right\\}}"
    )
    return {
        "segment_id": segment_id,
        "stroke_id": stroke_id,
        "type": "linear",
        "start": _point_payload(start),
        "end": _point_payload(end),
        "m": m,
        "b": b,
        "restriction": {"variable": "x", "min": x_min, "max": x_max},
        "equation": (
            f"y = {format_number(m)}x + {format_number(b)} "
            f"{{{format_number(x_min)} <= x <= {format_number(x_max)}}}"
        ),
        "latex": latex,
        "source_points": [_point_payload(point) for point in source_points or [start, end]],
        "fit_error": fit_error,
    }


def vertical_segment(
    segment_id: int,
    stroke_id: int,
    start: Point,
    end: Point,
    *,
    fit_error: float = 0.0,
    source_points: list[Point] | None = None,
) -> dict[str, Any]:
    x1, y1 = start
    _, y2 = end
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    latex = (
        f"x={format_number(x1)}"
        f"\\left\\{{{format_number(y_min)}\\le y\\le {format_number(y_max)}\\right\\}}"
    )
    return {
        "segment_id": segment_id,
        "stroke_id": stroke_id,
        "type": "vertical",
        "start": _point_payload(start),
        "end": _point_payload(end),
        "c": x1,
        "restriction": {"variable": "y", "min": y_min, "max": y_max},
        "equation": f"x = {format_number(x1)} {{{format_number(y_min)} <= y <= {format_number(y_max)}}}",
        "latex": latex,
        "source_points": [_point_payload(point) for point in source_points or [start, end]],
        "fit_error": fit_error,
    }


def quadratic_segment(
    segment_id: int,
    stroke_id: int,
    *,
    coefficients: tuple[float, float, float],
    x_range: tuple[float, float],
    endpoints: tuple[Point, Point],
    fit_error: float,
    source_points: list[Point] | None = None,
) -> dict[str, Any]:
    a, b, c = coefficients
    x_min, x_max = x_range
    start, end = endpoints
    b_sign = "+" if b >= 0 else ""
    c_sign = "+" if c >= 0 else ""
    latex = (
        f"y={format_number(a)}x^2{b_sign}{format_number(b)}x{c_sign}{format_number(c)}"
        f"\\left\\{{{format_number(x_min)}\\le x\\le {format_number(x_max)}\\right\\}}"
    )
    return {
        "segment_id": segment_id,
        "stroke_id": stroke_id,
        "type": "quadratic",
        "start": _point_payload(start),
        "end": _point_payload(end),
        "a": a,
        "b": b,
        "c": c,
        "restriction": {"variable": "x", "min": x_min, "max": x_max},
        "equation": (
            f"y = {format_number(a)}x^2 + {format_number(b)}x + {format_number(c)} "
            f"{{{format_number(x_min)} <= x <= {format_number(x_max)}}}"
        ),
        "latex": latex,
        "source_points": [_point_payload(point) for point in source_points or [start, end]],
        "fit_error": fit_error,
    }


def bezier_cubic_segment(
    segment_id: int,
    stroke_id: int,
    *,
    control_points: tuple[Point, Point, Point, Point],
    fit_error: float,
    source_points: list[Point] | None = None,
) -> dict[str, Any]:
    p0, p1, p2, p3 = control_points
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x_expr = (
        f"(1-t)^3({format_number(x0)})+3(1-t)^2t({format_number(x1)})+"
        f"3(1-t)t^2({format_number(x2)})+t^3({format_number(x3)})"
    )
    y_expr = (
        f"(1-t)^3({format_number(y0)})+3(1-t)^2t({format_number(y1)})+"
        f"3(1-t)t^2({format_number(y2)})+t^3({format_number(y3)})"
    )
    latex = f"({x_expr},{y_expr})\\left\\{{0\\le t\\le 1\\right\\}}"
    return {
        "segment_id": segment_id,
        "stroke_id": stroke_id,
        "type": "bezier_cubic",
        "start": _point_payload(p0),
        "end": _point_payload(p3),
        "control_points": [_point_payload(point) for point in control_points],
        "restriction": {"variable": "t", "min": 0.0, "max": 1.0},
        "equation": f"x(t) = {x_expr}; y(t) = {y_expr}; 0 <= t <= 1",
        "latex": latex,
        "source_points": [_point_payload(point) for point in source_points or [p0, p1, p2, p3]],
        "fit_error": fit_error,
    }


def segments_payload(
    segments: list[dict[str, Any]],
    *,
    strokes: list[dict[str, Any]],
    image_size: tuple[int, int],
    scale: float,
    target: int,
    fit_mode: str,
) -> dict[str, Any]:
    width, height = image_size
    return {
        "metadata": {
            "image_width": width,
            "image_height": height,
            "scale": scale,
            "target": target,
            "fit_mode": fit_mode,
            "equation_count": len(segments),
            "stroke_count": len(strokes),
        },
        "strokes": strokes,
        "segments": segments,
    }
