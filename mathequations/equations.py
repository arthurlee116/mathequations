from __future__ import annotations

from typing import Any


def format_number(value: float, places: int = 6) -> str:
    rounded = round(value, places)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:g}"


def equation_from_segment(
    segment_id: int,
    contour_id: int,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    vertical_tolerance: float = 1e-9,
) -> dict[str, Any]:
    x1, y1 = start
    x2, y2 = end
    base = {
        "segment_id": segment_id,
        "contour_id": contour_id,
        "group": f"contour_{contour_id}",
        "start": {"x": x1, "y": y1},
        "end": {"x": x2, "y": y2},
    }

    if abs(x2 - x1) <= vertical_tolerance:
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        c = x1
        return {
            **base,
            "type": "vertical",
            "c": c,
            "restriction": {"variable": "y", "min": y_min, "max": y_max},
            "equation": (
                f"x = {format_number(c)} "
                f"{{{format_number(y_min)} <= y <= {format_number(y_max)}}}"
            ),
        }

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    return {
        **base,
        "type": "linear",
        "m": m,
        "b": b,
        "restriction": {"variable": "x", "min": x_min, "max": x_max},
        "equation": (
            f"y = {format_number(m)}x + {format_number(b)} "
            f"{{{format_number(x_min)} <= x <= {format_number(x_max)}}}"
        ),
    }


def equations_from_contours(
    contours: list[list[tuple[float, float]]],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    segment_id = 1
    for contour_id, points in enumerate(contours):
        if len(points) < 2:
            continue
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            if start == end:
                continue
            segments.append(equation_from_segment(segment_id, contour_id, start, end))
            segment_id += 1
    return segments


def segments_to_jsonable(
    segments: list[dict[str, Any]],
    *,
    image_size: tuple[int, int],
    scale: float,
    target: int,
) -> dict[str, Any]:
    width, height = image_size
    return {
        "metadata": {
            "image_width": width,
            "image_height": height,
            "scale": scale,
            "target": target,
            "equation_count": len(segments),
        },
        "segments": segments,
    }
