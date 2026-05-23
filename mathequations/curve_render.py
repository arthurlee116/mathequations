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
    if kind == "parametric_polyline":
        polyline = [_payload_to_point(point) for point in segment["points"]]
        if len(polyline) <= 1:
            return polyline
        distances = [0.0]
        for start, end in zip(polyline, polyline[1:]):
            distances.append(distances[-1] + float(np.hypot(end[0] - start[0], end[1] - start[1])))
        total = distances[-1]
        if total <= 1e-9:
            return [polyline[0] for _ in range(samples)]
        result: list[Point] = []
        cursor = 0
        for sample in np.linspace(0.0, total, samples):
            while cursor < len(distances) - 2 and distances[cursor + 1] < sample:
                cursor += 1
            span = distances[cursor + 1] - distances[cursor]
            if span <= 1e-9:
                result.append(polyline[cursor])
                continue
            t = (sample - distances[cursor]) / span
            start = polyline[cursor]
            end = polyline[cursor + 1]
            result.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
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
    render_scale: int = 1,
) -> None:
    """Render sampled mixed curve segments on a white preview canvas."""
    if render_scale < 1:
        raise ValueError("render_scale must be at least 1")
    width, height = image_size
    canvas = np.full((height * render_scale, width * render_scale, 3), 255, dtype=np.uint8)
    for segment in segments:
        points = sample_segment_points(segment)
        pixels = [
            (
                _cartesian_to_pixel(point, width=width, height=height, scale=scale)[0] * render_scale,
                _cartesian_to_pixel(point, width=width, height=height, scale=scale)[1] * render_scale,
            )
            for point in points
        ]
        if len(pixels) >= 2:
            cv2.polylines(
                canvas,
                [np.array(pixels, dtype=np.int32).reshape((-1, 1, 2))],
                isClosed=False,
                color=(0, 0, 0),
                thickness=max(1, int(line_thickness * render_scale)),
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(str(path), canvas)


def render_curve_segment_previews(
    segments: list[dict[str, Any]],
    out_dir: Path,
    *,
    image_size: tuple[int, int],
    scale: float,
    render_scale: int = 4,
    line_thickness: int = 1,
) -> dict[str, Path]:
    """Write high-resolution and downsampled function previews."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = out_dir / "function_preview.png"
    highres_path = out_dir / "function_preview_highres.png"
    render_curve_segments(
        segments,
        highres_path,
        image_size=image_size,
        scale=scale,
        line_thickness=line_thickness,
        render_scale=render_scale,
    )
    highres = cv2.imread(str(highres_path), cv2.IMREAD_COLOR)
    if highres is None:
        raise ValueError(f"failed to read rendered preview: {highres_path}")
    width, height = image_size
    preview = cv2.resize(highres, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(preview_path), preview)
    return {"preview": preview_path, "highres": highres_path}


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


def render_stroke_chains(
    chains: list[Any],
    path: Path,
    *,
    image_size: tuple[int, int],
    line_thickness: int = 1,
    render_scale: int = 1,
) -> None:
    """Render reconstructed pixel-space stroke chains."""
    width, height = image_size
    canvas = np.full((height * render_scale, width * render_scale, 3), 255, dtype=np.uint8)
    for chain in chains:
        pixels = np.array(
            [(round(x * render_scale), round(y * render_scale)) for x, y in chain.points],
            dtype=np.int32,
        )
        if len(pixels) >= 2:
            cv2.polylines(
                canvas,
                [pixels.reshape((-1, 1, 2))],
                isClosed=bool(getattr(chain, "closed", False)),
                color=(0, 0, 0),
                thickness=max(1, int(line_thickness * render_scale)),
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(str(path), canvas)


def render_endpoint_overlay(
    endpoints: list[Any],
    bridges: list[Any],
    path: Path,
    *,
    image_size: tuple[int, int],
    line_thickness: int = 1,
    render_scale: int = 1,
) -> None:
    """Render endpoints and accepted bridges for diagnostics."""
    width, height = image_size
    canvas = np.full((height * render_scale, width * render_scale, 3), 255, dtype=np.uint8)
    point_by_id = {int(endpoint.endpoint_id): endpoint.point for endpoint in endpoints}
    for bridge in bridges:
        start = point_by_id.get(int(bridge.endpoint_a_id), bridge.start_point)
        end = point_by_id.get(int(bridge.endpoint_b_id), bridge.end_point)
        p0 = (round(start[0] * render_scale), round(start[1] * render_scale))
        p1 = (round(end[0] * render_scale), round(end[1] * render_scale))
        cv2.line(
            canvas,
            p0,
            p1,
            color=(80, 180, 80),
            thickness=max(1, int(line_thickness * render_scale)),
            lineType=cv2.LINE_AA,
        )
    radius = max(2, 2 * render_scale)
    for endpoint in endpoints:
        center = (round(endpoint.point[0] * render_scale), round(endpoint.point[1] * render_scale))
        cv2.circle(canvas, center, radius, (40, 80, 230), thickness=-1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
