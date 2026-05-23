"""Lucas-specific vector reconstruction before equation export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .equations import equation_from_segment, equations_from_contours, format_number
from .function_render import render_function_segments_payload
from .geometry import map_contours_to_cartesian
from .image_processing import foreground_bbox, foreground_mask, load_image, normalize_white_background
from .vector_model import Point, VectorArtwork, VectorShape
from .vector_render import render_vector_artwork, write_svg


@dataclass(frozen=True)
class LucasVectorPipelineResult:
    """Shape and segment counts produced by one vector pipeline run."""

    out_dir: Path
    shape_count: int
    segment_count: int


def _clean_mask(mask: np.ndarray, *, close_px: int = 5, open_px: int = 3) -> np.ndarray:
    """Normalize a mask to 0/255 and clean small gaps or speckles."""
    result = mask.astype(np.uint8)
    if result.max() <= 1:
        result = result * 255
    if close_px > 0:
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_CLOSE,
            np.ones((close_px, close_px), np.uint8),
            iterations=1,
        )
    if open_px > 0:
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_OPEN,
            np.ones((open_px, open_px), np.uint8),
            iterations=1,
        )
    return result


def _external_filled_mask(mask: np.ndarray) -> np.ndarray:
    """Fill the outer silhouette of a mask while discarding internal holes."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return filled


def _contour_points(contour: np.ndarray, *, epsilon_px: float) -> list[Point]:
    """Simplify one contour into vector points."""
    approx = cv2.approxPolyDP(contour, epsilon_px, True)
    return [(float(point[0][0]), float(point[0][1])) for point in approx]


def _shape_from_mask(
    mask: np.ndarray,
    *,
    shape_id: str,
    fill: str,
    z_index: int,
    min_area: float,
    epsilon_px: float,
    stroke: str | None = None,
    stroke_width: float = 0,
) -> VectorShape | None:
    """Create one vector shape from the usable contours in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    useful = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]
    useful.sort(key=cv2.contourArea, reverse=True)
    contour_points = [
        _contour_points(contour, epsilon_px=epsilon_px)
        for contour in useful
    ]
    contour_points = [points for points in contour_points if len(points) >= 3]
    if not contour_points:
        return None
    return VectorShape(
        shape_id=shape_id,
        contours=contour_points,
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
        z_index=z_index,
    )


def _superellipse_points(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    *,
    exponent: float = 3.2,
    point_count: int = 72,
) -> list[Point]:
    """Return rounded-rectangle-like points for stylized eye regions."""
    points: list[Point] = []
    for theta in np.linspace(0, 2 * np.pi, point_count, endpoint=False):
        cos_value = float(np.cos(theta))
        sin_value = float(np.sin(theta))
        x = center_x + radius_x * np.sign(cos_value) * (abs(cos_value) ** (2 / exponent))
        y = center_y + radius_y * np.sign(sin_value) * (abs(sin_value) ** (2 / exponent))
        points.append((round(x, 3), round(y, 3)))
    return points


def _split_purple_eye_shape(
    purple_mask: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[VectorShape | None, np.ndarray]:
    """Pull purple eye regions out of the broader purple mask."""
    contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []
    image_area = float(width * height)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        center_x = x + box_width / 2
        if area < max(120.0, image_area * 0.004):
            continue
        overlaps_eye_band = y < height * 0.62 and y + box_height > height * 0.25
        if not overlaps_eye_band:
            continue
        if box_width < width * 0.08 or box_height < height * 0.08:
            continue
        if width * 0.15 <= center_x <= width * 0.85:
            candidates.append((area, (x, y, box_width, box_height), contour))

    left_candidates = [candidate for candidate in candidates if candidate[1][0] + candidate[1][2] / 2 < width / 2]
    right_candidates = [candidate for candidate in candidates if candidate[1][0] + candidate[1][2] / 2 >= width / 2]
    if not left_candidates or not right_candidates:
        spanning = [
            candidate for candidate in candidates
            if candidate[1][0] < width / 2 < candidate[1][0] + candidate[1][2]
        ]
        if not spanning:
            return None, purple_mask
        # Some inputs merge both purple eyes into one blob; split that blob by
        # fitting two synthetic eye contours instead of trusting the mask edge.
        area, (x, y, box_width, box_height), contour = max(spanning, key=lambda candidate: candidate[0])
        center_y = y + box_height / 2
        radius_x = box_width * 0.22
        radius_y = box_height * 0.46
        contours_points = [
            _superellipse_points(x + box_width * 0.28, center_y, radius_x, radius_y),
            _superellipse_points(x + box_width * 0.72, center_y, radius_x, radius_y),
        ]
        eye_shape = VectorShape(
            shape_id="purple_eye_regions",
            contours=contours_points,
            fill="#6256c8",
            stroke=None,
            stroke_width=0,
            z_index=31,
        )
        remainder = purple_mask.copy()
        removal = np.zeros_like(purple_mask)
        cv2.drawContours(removal, [contour], -1, 255, thickness=-1)
        removal = cv2.dilate(removal, np.ones((7, 7), np.uint8), iterations=1)
        remainder[removal > 0] = 0
        return eye_shape, remainder

    left = max(left_candidates, key=lambda candidate: candidate[0])
    right = max(right_candidates, key=lambda candidate: candidate[0])
    selected = [left, right]
    boxes = [candidate[1] for candidate in selected]
    center_y = sum(y + box_height / 2 for _, y, _, box_height in boxes) / 2
    radius_x = sum(box_width for _, _, box_width, _ in boxes) / 4
    radius_y = sum(box_height for _, _, _, box_height in boxes) / 4
    radius_x *= 0.94
    radius_y *= 0.92

    contours_points = [
        _superellipse_points(x + box_width / 2, center_y, radius_x, radius_y)
        for x, _, box_width, _ in boxes
    ]
    eye_shape = VectorShape(
        shape_id="purple_eye_regions",
        contours=contours_points,
        fill="#6256c8",
        stroke=None,
        stroke_width=0,
        z_index=31,
    )

    remainder = purple_mask.copy()
    removal = np.zeros_like(purple_mask)
    cv2.drawContours(removal, [candidate[2] for candidate in selected], -1, 255, thickness=-1)
    removal = cv2.dilate(removal, np.ones((9, 9), np.uint8), iterations=1)
    remainder[removal > 0] = 0
    return eye_shape, remainder


def _bbox_from_points(points: list[Point]) -> tuple[float, float, float, float]:
    """Return the bounding box around a point list."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _eye_lid_points(eye_contour: list[Point]) -> list[Point]:
    """Build a black lower-lid contour inside one purple eye contour."""
    x_min, y_min, x_max, y_max = _bbox_from_points(eye_contour)
    width = x_max - x_min
    height = y_max - y_min
    y_top = y_min + height * 0.66
    y_bottom = y_min + height * 0.93
    left_x = x_min + width * 0.10
    right_x = x_max - width * 0.10
    points: list[Point] = [(round(left_x, 3), round(y_top, 3)), (round(right_x, 3), round(y_top, 3))]
    for theta in np.linspace(0, np.pi, 28):
        x = (left_x + right_x) / 2 + (right_x - left_x) / 2 * np.cos(theta)
        y = y_top + (y_bottom - y_top) * np.sin(theta)
        points.append((round(float(x), 3), round(float(y), 3)))
    return points


def _black_eye_lids_from_eyes(eye_shape: VectorShape | None) -> VectorShape | None:
    """Create eyelid shapes only when exactly two purple eyes were found."""
    if eye_shape is None:
        return None
    contours = [_eye_lid_points(contour) for contour in eye_shape.contours]
    if len(contours) != 2:
        return None
    return VectorShape(
        shape_id="black_eye_lids",
        contours=contours,
        fill="#050505",
        stroke=None,
        stroke_width=0,
        z_index=61,
    )


def _mask_without_shape(mask: np.ndarray, shape: VectorShape | None, *, dilate_px: int) -> np.ndarray:
    """Remove a known shape from a mask before extracting overlapping details."""
    if shape is None:
        return mask
    removal = np.zeros_like(mask)
    contour_arrays = [
        np.round(np.array(contour, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
        for contour in shape.contours
        if contour
    ]
    if contour_arrays:
        cv2.drawContours(removal, contour_arrays, -1, 255, thickness=-1)
        removal = cv2.dilate(removal, np.ones((dilate_px, dilate_px), np.uint8), iterations=1)
    result = mask.copy()
    result[removal > 0] = 0
    return result


def _mask_where(condition: np.ndarray, scope: np.ndarray, *, close_px: int, open_px: int) -> np.ndarray:
    """Apply a boolean color condition inside a mask scope, then clean it."""
    mask = np.zeros(scope.shape, dtype=np.uint8)
    mask[(condition) & (scope > 0)] = 255
    return _clean_mask(mask, close_px=close_px, open_px=open_px)


def build_lucas_vector_artwork(image: np.ndarray) -> VectorArtwork:
    """Reconstruct Lucas as named, layered vector shapes from the source image."""
    clean = normalize_white_background(image)
    height, width = clean.shape[:2]
    foreground = _clean_mask(foreground_mask(clean), close_px=7, open_px=3)
    silhouette = _external_filled_mask(foreground)
    inner_silhouette = cv2.erode(silhouette, np.ones((27, 27), np.uint8), iterations=1)

    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    y_coordinates = np.indices((height, width))[0]
    upper_character = np.zeros_like(silhouette)
    upper_character[(silhouette > 0) & (y_coordinates < height * 0.62)] = 255

    image_area = float(width * height)
    min_large = max(80.0, image_area * 0.0006)
    min_small = max(20.0, image_area * 0.00012)
    # The masks below are intentionally semantic: each threshold tries to
    # recover one visible character material or detail, then z-index handles
    # the paint order.
    purple_mask = _mask_where(
        (hue >= 112) & (hue <= 158) & (saturation >= 35) & (value >= 65),
        upper_character,
        close_px=11,
        open_px=5,
    )
    purple_eye_shape, purple_remainder = _split_purple_eye_shape(
        purple_mask,
        width=width,
        height=height,
    )
    black_scope = _mask_without_shape(inner_silhouette, purple_eye_shape, dilate_px=17)

    masks: list[tuple[str, np.ndarray, str, int, float, float]] = []
    masks.append(
        (
            "silhouette",
            silhouette,
            "#050505",
            0,
            max(120.0, image_area * 0.002),
            2.8,
        )
    )
    masks.append(
        (
            "green_body",
            _mask_where(
                (hue >= 35) & (hue <= 92) & (saturation >= 35) & (value >= 70),
                silhouette,
                close_px=11,
                open_px=5,
            ),
            "#69e6a1",
            10,
            min_large,
            4.2,
        )
    )
    masks.append(
        (
            "dark_blue_clothing",
            _mask_where(
                (hue >= 90) & (hue <= 132) & (saturation >= 45) & (value <= 120),
                silhouette,
                close_px=9,
                open_px=5,
            ),
            "#172157",
            20,
            min_large,
            3.4,
        )
    )
    masks.append(
        (
            "purple_regions",
            purple_remainder,
            "#6256c8",
            30,
            min_large,
            4.0,
        )
    )
    masks.append(
        (
            "metal_gray",
            _mask_where(
                (saturation <= 75) & (value >= 80) & (value <= 235),
                silhouette,
                close_px=7,
                open_px=3,
            ),
            "#b7c3cf",
            35,
            min_small,
            2.8,
        )
    )
    masks.append(
        (
            "yellow_accents",
            _mask_where(
                (hue >= 12) & (hue <= 38) & (saturation >= 55) & (value >= 120),
                silhouette,
                close_px=7,
                open_px=3,
            ),
            "#ffd84d",
            70,
            min_small,
            2.6,
        )
    )
    masks.append(
        (
            "white_details",
            _mask_where(
                (saturation <= 55) & (value >= 190),
                silhouette,
                close_px=5,
                open_px=3,
            ),
            "#ffffff",
            50,
            min_small,
            2.2,
        )
    )
    masks.append(
        (
            "black_details",
            _mask_where(
                (value <= 55) & (saturation <= 125),
                black_scope,
                close_px=5,
                open_px=2,
            ),
            "#050505",
            60,
            min_small,
            2.0,
        )
    )

    shapes: list[VectorShape] = []
    for shape_id, mask, fill, z_index, min_area, epsilon_px in masks:
        shape = _shape_from_mask(
            mask,
            shape_id=shape_id,
            fill=fill,
            z_index=z_index,
            min_area=min_area,
            epsilon_px=epsilon_px,
        )
        if shape is not None:
            shapes.append(shape)
    if purple_eye_shape is not None:
        shapes.append(purple_eye_shape)
    black_eye_lids = _black_eye_lids_from_eyes(purple_eye_shape)
    if black_eye_lids is not None:
        shapes.append(black_eye_lids)

    return VectorArtwork(width=width, height=height, shapes=shapes)


def _artwork_bbox(artwork: VectorArtwork) -> tuple[int, int, int, int]:
    """Return the bounding box around every point in the artwork."""
    xs: list[float] = []
    ys: list[float] = []
    for shape in artwork.shapes:
        for contour in shape.contours:
            xs.extend(point[0] for point in contour)
            ys.extend(point[1] for point in contour)
    if not xs or not ys:
        raise ValueError("Vector artwork has no points")
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _desmos_latex_from_segment(segment: dict[str, Any]) -> str:
    """Format one segment dictionary as Desmos-compatible LaTeX."""
    if segment["type"] == "vertical":
        c = format_number(float(segment["c"]))
        y_min = format_number(float(segment["restriction"]["min"]))
        y_max = format_number(float(segment["restriction"]["max"]))
        return f"x={c}\\left\\{{{y_min}\\le y\\le {y_max}\\right\\}}"

    m = format_number(float(segment["m"]))
    b_value = float(segment["b"])
    b = format_number(b_value)
    sign = "+" if b_value >= 0 else ""
    x_min = format_number(float(segment["restriction"]["min"]))
    x_max = format_number(float(segment["restriction"]["max"]))
    return f"y={m}x{sign}{b}\\left\\{{{x_min}\\le x\\le {x_max}\\right\\}}"


def _segments_from_artwork(
    artwork: VectorArtwork,
    *,
    scale_width: float,
) -> tuple[list[dict[str, Any]], float]:
    """Convert vector contours to restricted equations plus their scale."""
    x_min, _, x_max, _ = _artwork_bbox(artwork)
    scale = scale_width / max(1, x_max - x_min + 1)
    segments: list[dict[str, Any]] = []
    next_segment_id = 1
    next_contour_id = 0
    for shape in artwork.ordered_shapes():
        cartesian_contours = map_contours_to_cartesian(
            shape.contours,
            width=artwork.width,
            height=artwork.height,
            scale=scale,
        )
        shape_segments = equations_from_contours(cartesian_contours)
        for offset, segment in enumerate(shape_segments):
            segment["segment_id"] = next_segment_id + offset
            segment["contour_id"] = next_contour_id + int(segment["contour_id"])
            segment["shape_id"] = shape.shape_id
            segment["color"] = shape.stroke or shape.fill or "#000000"
            segment["group"] = shape.shape_id
        segments.extend(shape_segments)
        next_segment_id += len(shape_segments)
        next_contour_id += len(shape.contours)
    return segments, scale


def _write_desmos_vector_exports(
    out_dir: Path,
    segments: list[dict[str, Any]],
) -> None:
    """Write Desmos-ready JSON and text exports for vector segments."""
    expressions = [
        {
            "id": f"lucas_vector_{index}",
            "latex": _desmos_latex_from_segment(segment),
            "color": segment.get("color", "#000000"),
            "lineWidth": "1",
        }
        for index, segment in enumerate(segments, start=1)
    ]
    (out_dir / "desmos_vector_expressions.json").write_text(
        json.dumps(expressions, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "desmos_vector_latex.txt").write_text(
        "\n".join(expression["latex"] for expression in expressions) + "\n",
        encoding="utf-8",
    )
    (out_dir / "desmos_vector_equations.txt").write_text(
        "\n".join(segment["equation"] for segment in segments) + "\n",
        encoding="utf-8",
    )


def run_lucas_vector_pipeline(
    *,
    image_path: Path,
    out_dir: Path,
    scale_width: float = 20.0,
) -> LucasVectorPipelineResult:
    """Run vector reconstruction and write SVG, previews, JSON, and equations."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(Path(image_path))
    clean = normalize_white_background(image)
    cv2.imwrite(str(out_dir / "lucas_vector_clean_input.png"), clean)

    artwork = build_lucas_vector_artwork(clean)
    write_svg(artwork, out_dir / "lucas_clean.svg")
    render_vector_artwork(artwork, out_dir / "lucas_vector_preview.png")

    segments, scale = _segments_from_artwork(artwork, scale_width=scale_width)
    payload = {
        "metadata": {
            "image_width": artwork.width,
            "image_height": artwork.height,
            "scale": scale,
            "shape_count": len(artwork.shapes),
            "segment_count": len(segments),
        },
        "shapes": [
            {
                "shape_id": shape.shape_id,
                "fill": shape.fill,
                "stroke": shape.stroke,
                "stroke_width": shape.stroke_width,
                "z_index": shape.z_index,
                "contour_count": len(shape.contours),
            }
            for shape in artwork.ordered_shapes()
        ],
        "segments": segments,
    }
    (out_dir / "lucas_vector_segments.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    render_function_segments_payload(payload, out_dir / "lucas_function_render.png")
    _write_desmos_vector_exports(out_dir, segments)

    return LucasVectorPipelineResult(
        out_dir=out_dir,
        shape_count=len(artwork.shapes),
        segment_count=len(segments),
    )
