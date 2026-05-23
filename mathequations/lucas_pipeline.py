from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .equations import equation_from_segment, equations_from_contours, format_number
from .geometry import (
    allocate_targets,
    contour_to_points,
    map_contours_to_cartesian,
    polyline_length,
    resample_closed_points,
)
from .image_processing import (
    foreground_bbox,
    foreground_mask,
    load_image,
    normalize_white_background,
)
from .layers import EquationLayer, FillRegion
from .render import render_filled_regions, render_segments


@dataclass(frozen=True)
class LucasPipelineResult:
    out_dir: Path
    total_equations: int
    outline_equations: int
    fill_equations: int


def extract_line_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    near_black = ((gray < 85) & (saturation < 120)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(near_black, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)


def extract_outline_contours(
    foreground_mask: np.ndarray,
    *,
    min_area: float = 50.0,
) -> list[np.ndarray]:
    closed = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=2,
    )
    smoothed = cv2.GaussianBlur(closed, (0, 0), 1.0)
    _, outline_mask = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    useful = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    useful.sort(key=cv2.contourArea, reverse=True)
    return useful


def extract_internal_line_contours(
    line_mask: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    min_component_area: int = 180,
    min_component_extent: int = 14,
    foreground_erode_px: int = 21,
) -> list[np.ndarray]:
    erode_px = max(1, foreground_erode_px)
    interior = cv2.erode(foreground_mask, np.ones((erode_px, erode_px), np.uint8), iterations=1)
    details = cv2.bitwise_and(line_mask, interior)
    details = cv2.morphologyEx(details, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    details = cv2.morphologyEx(details, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    details = cv2.GaussianBlur(details, (0, 0), 0.8)
    _, details = cv2.threshold(details, 127, 255, cv2.THRESH_BINARY)

    count, labels, stats, _ = cv2.connectedComponentsWithStats((details > 0).astype(np.uint8), 8)
    filtered = np.zeros_like(details)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= min_component_area and max(width, height) >= min_component_extent:
            filtered[labels == label] = 255

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    useful = [contour for contour in contours if cv2.contourArea(contour) >= 60.0]
    useful.sort(key=_contour_length, reverse=True)
    return useful


def _smooth_feature_mask(mask: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    mask = cv2.bitwise_and(mask.astype(np.uint8) * 255, foreground_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def _contours_from_feature_mask(
    mask: np.ndarray,
    *,
    min_area: float,
    limit: int,
    keep_contour: Any | None = None,
) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    useful: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        if keep_contour is not None and not keep_contour(contour):
            continue
        useful.append(contour)
    useful.sort(key=cv2.contourArea, reverse=True)
    return useful[:limit]


def extract_feature_contours(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    foreground_erode_px: int = 35,
) -> list[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = foreground_mask.shape
    erode_px = max(1, foreground_erode_px)
    interior = cv2.erode(foreground_mask, np.ones((erode_px, erode_px), np.uint8), iterations=1)

    purple_mask = _smooth_feature_mask(
        (hue >= 105) & (hue <= 145) & (saturation >= 45) & (value >= 60),
        foreground_mask,
    )
    dark_mask = _smooth_feature_mask(
        (gray < 80) & (interior > 0),
        foreground_mask,
    )
    yellow_mask = _smooth_feature_mask(
        (hue >= 15) & (hue <= 40) & (saturation >= 60) & (value >= 120),
        foreground_mask,
    )

    def keep_dark_feature(contour: np.ndarray) -> bool:
        area = cv2.contourArea(contour)
        x, y, _, _ = cv2.boundingRect(contour)
        if area > max(30000.0, width * height * 0.02):
            return False
        top_accessory = width * 0.42 <= x <= width * 0.58 and y < height * 0.22
        face_feature = y > height * 0.37
        return top_accessory or face_feature

    feature_contours: list[np.ndarray] = []
    feature_contours.extend(
        _contours_from_feature_mask(purple_mask, min_area=500.0, limit=6)
    )
    feature_contours.extend(
        _contours_from_feature_mask(
            dark_mask,
            min_area=120.0,
            limit=6,
            keep_contour=keep_dark_feature,
        )
    )
    feature_contours.extend(
        _contours_from_feature_mask(yellow_mask, min_area=120.0, limit=4)
    )
    feature_contours.sort(key=_contour_length, reverse=True)
    return feature_contours


def quantize_to_gray_regions(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    levels: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    source_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(source_gray, 9, 30, 30)
    foreground_values = smoothed[foreground_mask > 0]
    if len(foreground_values) == 0:
        raise ValueError("foreground_mask has no pixels")

    levels = max(2, min(levels, len(np.unique(foreground_values))))
    percentiles = np.linspace(0, 100, levels + 1)
    edges = np.percentile(foreground_values, percentiles)
    if len(np.unique(edges)) <= 2:
        edges = np.linspace(int(foreground_values.min()), int(foreground_values.max()) + 1, levels + 1)

    labels = np.digitize(smoothed, edges[1:-1], right=True).astype(np.uint8) + 1
    region_mask = np.zeros(foreground_mask.shape, dtype=np.uint8)
    region_mask[foreground_mask > 0] = labels[foreground_mask > 0]
    region_mask = cv2.medianBlur(region_mask, 5)
    region_mask[foreground_mask == 0] = 0

    gray_image = np.full(foreground_mask.shape, 255, dtype=np.uint8)
    for label in sorted(int(v) for v in np.unique(region_mask) if v != 0):
        pixels = source_gray[region_mask == label]
        if len(pixels) > 0:
            gray_image[region_mask == label] = int(np.median(pixels))
    return gray_image, region_mask


def source_grayscale_fill(image: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    source_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(source_gray, 9, 30, 30)
    result = np.full(foreground_mask.shape, 255, dtype=np.uint8)
    result[foreground_mask > 0] = smoothed[foreground_mask > 0]
    return result


def extract_region_contours(
    region_mask: np.ndarray,
    *,
    min_area: float = 80.0,
) -> dict[int, list[np.ndarray]]:
    result: dict[int, list[np.ndarray]] = {}
    for label in sorted(int(v) for v in np.unique(region_mask) if v != 0):
        mask = (region_mask == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        useful = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
        if useful:
            result[label] = useful
    return result


def _write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _contour_length(contour: np.ndarray) -> float:
    return polyline_length(contour_to_points(contour), closed=True)


def _select_contours(contours: list[np.ndarray], target: int, *, min_points: int) -> list[np.ndarray]:
    max_contours = max(1, target // min_points)
    ordered = sorted(contours, key=_contour_length, reverse=True)
    return ordered[:max_contours]


def _allocate_contour_targets(
    lengths: list[float],
    total_target: int,
    *,
    min_points: int,
) -> list[int]:
    allocations = allocate_targets(lengths, total_target)
    if not allocations:
        return []

    allocations = [max(min_points, allocated) for allocated in allocations]
    overage = sum(allocations) - total_target
    if overage <= 0:
        return allocations

    order = sorted(range(len(allocations)), key=lambda index: allocations[index], reverse=True)
    while overage > 0:
        changed = False
        for index in order:
            if overage == 0:
                break
            if allocations[index] > min_points:
                allocations[index] -= 1
                overage -= 1
                changed = True
        if not changed:
            break
    return allocations


def _contours_to_target_points(contours: list[np.ndarray], target: int) -> list[list[tuple[float, float]]]:
    return _contours_to_target_points_with_epsilon(contours, target, epsilon_px=0.5, closed=True)


def _simplify_points(contour: np.ndarray, *, epsilon_px: float, closed: bool) -> list[tuple[float, float]]:
    approx = cv2.approxPolyDP(contour, epsilon_px, closed)
    return contour_to_points(approx)


def _resample_open_points(
    points: list[tuple[float, float]],
    target_count: int,
) -> list[tuple[float, float]]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(points) <= 1:
        return points[:]

    total_length = polyline_length(points, closed=False)
    if total_length == 0:
        return [points[0]]

    target_count = max(2, target_count)
    distances = np.linspace(0, total_length, target_count)
    sampled: list[tuple[float, float]] = []
    current_distance = 0.0
    target_index = 0
    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        segment_length = float(np.hypot(x2 - x1, y2 - y1))
        if segment_length == 0:
            continue
        while target_index < len(distances) and distances[target_index] <= current_distance + segment_length:
            ratio = (distances[target_index] - current_distance) / segment_length
            x = x1 + (x2 - x1) * ratio
            y = y1 + (y2 - y1) * ratio
            sampled.append((round(float(x), 6), round(float(y), 6)))
            target_index += 1
        current_distance += segment_length
    if not sampled or sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _contours_to_target_points_with_epsilon(
    contours: list[np.ndarray],
    target: int,
    *,
    epsilon_px: float,
    closed: bool,
    min_points: int = 3,
    preserve_detail: bool = False,
) -> list[list[tuple[float, float]]]:
    if not contours:
        return []

    contours = _select_contours(contours, target, min_points=min_points)
    simplified = [
        _simplify_points(contour, epsilon_px=epsilon_px, closed=closed)
        for contour in contours
    ]
    lengths = [polyline_length(points, closed=closed) for points in simplified]
    allocations = _allocate_contour_targets(lengths, target, min_points=min_points)

    output: list[list[tuple[float, float]]] = []
    for contour, points, allocated in zip(contours, simplified, allocations):
        lower_bound = max(3, round(allocated * 0.75))
        upper_bound = max(3, round(allocated * 1.25))
        if lower_bound <= len(points) <= upper_bound:
            output.append(points)
        elif len(points) > allocated and not preserve_detail:
            if closed:
                output.append(resample_closed_points(points, allocated))
            else:
                output.append(_resample_open_points(points, allocated))
        else:
            output.append(points)
    return output


def _line_contours(mask: np.ndarray, *, min_area: float = 6.0) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    useful = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    useful.sort(key=_contour_length, reverse=True)
    return useful


def _segments_from_contours(
    contours: list[np.ndarray],
    *,
    target: int,
    width: int,
    height: int,
    scale: float,
    segment_id_start: int,
    contour_id_start: int,
    layer: str,
    label: int | None = None,
    gray: int | None = None,
    epsilon_px: float = 0.5,
    closed: bool = True,
    min_points: int = 3,
    preserve_detail: bool = False,
) -> list[dict[str, Any]]:
    pixel_contours = _contours_to_target_points_with_epsilon(
        contours,
        target,
        epsilon_px=epsilon_px,
        closed=closed,
        min_points=min_points,
        preserve_detail=preserve_detail,
    )
    cartesian_contours = map_contours_to_cartesian(
        pixel_contours,
        width=width,
        height=height,
        scale=scale,
    )
    if closed:
        segments = equations_from_contours(cartesian_contours)
    else:
        segments = []
        segment_id = 1
        for contour_id, points in enumerate(cartesian_contours):
            if len(points) < 2:
                continue
            for index in range(len(points) - 1):
                start = points[index]
                end = points[index + 1]
                if start == end:
                    continue
                segments.append(equation_from_segment(segment_id, contour_id, start, end))
                segment_id += 1
    for offset, segment in enumerate(segments):
        segment["segment_id"] = segment_id_start + offset
        segment["contour_id"] = contour_id_start + int(segment["contour_id"])
        segment["layer"] = layer
        if label is not None:
            segment["region_label"] = label
            segment["gray"] = gray
            segment["group"] = f"region_{label}_gray_{gray}"
    return segments


def _label_to_gray(gray_image: np.ndarray, region_mask: np.ndarray) -> dict[int, int]:
    result: dict[int, int] = {}
    for label in sorted(int(v) for v in np.unique(region_mask) if v != 0):
        values = gray_image[region_mask == label]
        if len(values) > 0:
            result[label] = int(np.median(values))
    return result


def _region_payload(
    regions: list[FillRegion],
    *,
    image_size: tuple[int, int],
    scale: float,
    gray_levels: int,
) -> dict[str, Any]:
    width, height = image_size
    return {
        "metadata": {
            "image_width": width,
            "image_height": height,
            "scale": scale,
            "gray_levels": gray_levels,
            "region_count": len(regions),
        },
        "regions": [
            {
                "region_id": region.region_id,
                "gray": region.gray,
                "label": region.label,
                "boundary_segment_ids": region.boundary_segment_ids,
            }
            for region in regions
        ],
    }


def _desmos_latex_from_segment(segment: dict[str, Any]) -> str:
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


def _write_desmos_exports(out_dir: Path, segments: list[dict[str, Any]]) -> None:
    expressions = [
        {
            "id": f"lucas_{index}",
            "latex": _desmos_latex_from_segment(segment),
            "color": "#000000",
            "lineWidth": "1",
        }
        for index, segment in enumerate(segments, start=1)
    ]
    (out_dir / "desmos_expressions.json").write_text(
        json.dumps(expressions, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_text(out_dir / "desmos_latex.txt", [expression["latex"] for expression in expressions])
    _write_text(out_dir / "desmos_equations.txt", [segment["equation"] for segment in segments])

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lucas Desmos Equations</title>
  <style>
    html, body {{ height: 100%; margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }}
    #calculator {{ position: fixed; inset: 0; }}
    #status {{ position: fixed; top: 12px; right: 12px; z-index: 10; background: rgba(255,255,255,.92); border: 1px solid #ddd; border-radius: 8px; padding: 8px 10px; font-size: 13px; box-shadow: 0 2px 12px rgba(0,0,0,.12); }}
  </style>
  <script src="https://www.desmos.com/api/v1.11/calculator.js?apiKey=desmos"></script>
</head>
<body>
  <div id="calculator"></div>
  <div id="status">Preparing Lucas equations...</div>
  <script>
    const expressions = {json.dumps(expressions, ensure_ascii=False)};
    const calculator = Desmos.GraphingCalculator(document.getElementById('calculator'), {{
      expressions: true,
      settingsMenu: true,
      zoomButtons: true,
      lockViewport: false,
      border: false
    }});
    calculator.setMathBounds({{ left: -11.8, right: 11.8, bottom: -14.8, top: 17.2 }});
    const status = document.getElementById('status');
    let index = 0;
    const batchSize = 150;
    function loadBatch() {{
      const end = Math.min(index + batchSize, expressions.length);
      for (; index < end; index++) {{
        calculator.setExpression(expressions[index]);
      }}
      status.textContent = `Loaded ${{index}} / ${{expressions.length}} equations`;
      if (index < expressions.length) {{
        setTimeout(loadBatch, 20);
      }} else {{
        status.textContent = `Lucas loaded: ${{expressions.length}} equations`;
      }}
    }}
    window.LUCAS_EXPRESSIONS = expressions;
    window.LUCAS_CALCULATOR = calculator;
    loadBatch();
  </script>
</body>
</html>
"""
    (out_dir / "desmos_lucas.html").write_text(html, encoding="utf-8")


def run_lucas_pipeline(
    *,
    image_path: Path,
    out_dir: Path,
    outline_target: int = 1500,
    region_target: int = 1200,
    gray_levels: int = 6,
    scale_width: float = 20.0,
) -> LucasPipelineResult:
    if outline_target < 3:
        raise ValueError("outline_target must be at least 3")
    if region_target < 3:
        raise ValueError("region_target must be at least 3")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(Path(image_path))
    clean = normalize_white_background(image)
    height, width = clean.shape[:2]
    cv2.imwrite(str(out_dir / "clean_input.png"), clean)

    foreground = foreground_mask(clean)
    x_min, _, x_max, _ = foreground_bbox(foreground)
    foreground_width = max(1, x_max - x_min + 1)
    scale = scale_width / foreground_width

    line_mask = cv2.bitwise_and(extract_line_mask(clean), foreground)
    cv2.imwrite(str(out_dir / "debug_line_mask.png"), line_mask)
    outline_contours = extract_outline_contours(foreground)
    feature_contours = extract_feature_contours(clean, foreground)
    if not outline_contours and not feature_contours:
        raise ValueError("No outline contours found")

    outline_segments: list[dict[str, Any]] = []
    silhouette_target = outline_target
    internal_target = 0
    if outline_contours and feature_contours:
        silhouette_target = max(3, round(outline_target * 0.74))
        internal_target = max(3, outline_target - silhouette_target)

    if outline_contours:
        outline_segments.extend(
            _segments_from_contours(
                outline_contours,
                target=silhouette_target,
                width=width,
                height=height,
                scale=scale,
                segment_id_start=1,
                contour_id_start=0,
                layer="outline",
                epsilon_px=0.2,
                closed=True,
                min_points=20,
            )
        )
    if feature_contours:
        outline_segments.extend(
            _segments_from_contours(
                feature_contours,
                target=internal_target or outline_target,
                width=width,
                height=height,
                scale=scale,
                segment_id_start=len(outline_segments) + 1,
                contour_id_start=len(outline_contours),
                layer="outline",
                epsilon_px=1.8,
                closed=True,
                min_points=10,
            )
        )
    outline_equations = [segment["equation"] for segment in outline_segments]
    outline_layer = EquationLayer("lucas_outline", "line", outline_equations)

    gray_image, region_mask = quantize_to_gray_regions(clean, foreground, levels=gray_levels)
    cv2.imwrite(str(out_dir / "debug_regions.png"), gray_image)
    region_contours = extract_region_contours(region_mask)
    label_gray = _label_to_gray(gray_image, region_mask)

    total_region_area = sum(
        cv2.contourArea(contour)
        for contours in region_contours.values()
        for contour in contours
    )
    fill_segments: list[dict[str, Any]] = []
    regions: list[FillRegion] = []
    next_segment_id = len(outline_segments) + 1
    next_contour_id = len(outline_contours) + len(feature_contours)
    for label, contours in sorted(region_contours.items()):
        area = sum(cv2.contourArea(contour) for contour in contours)
        if total_region_area > 0:
            target = max(3 * len(contours), round(region_target * area / total_region_area))
        else:
            target = max(3 * len(contours), region_target // max(1, len(region_contours)))
        gray = label_gray.get(label, 200)
        segments = _segments_from_contours(
            contours,
            target=target,
            width=width,
            height=height,
            scale=scale,
            segment_id_start=next_segment_id,
            contour_id_start=next_contour_id,
            layer="fill_boundary",
            label=label,
            gray=gray,
            epsilon_px=2.0,
            min_points=8,
        )
        segment_ids = [int(segment["segment_id"]) for segment in segments]
        regions.append(
            FillRegion(
                region_id=label,
                gray=gray,
                boundary_segment_ids=segment_ids,
                label=f"gray_{gray}",
            )
        )
        fill_segments.extend(segments)
        next_segment_id += len(segments)
        next_contour_id += len(contours)

    fill_equations = [segment["equation"] for segment in fill_segments]
    fill_layer = EquationLayer("lucas_fill_boundaries", "fill_boundary", fill_equations)
    all_equations = outline_layer.equations + fill_layer.equations

    _write_text(out_dir / "equations_outline.txt", outline_layer.equations)
    _write_text(out_dir / "equations_fill.txt", fill_layer.equations)
    _write_text(out_dir / "equations_all.txt", all_equations)

    selected_stride = max(1, len(all_equations) // 25)
    _write_text(out_dir / "selected_equations.txt", all_equations[::selected_stride][:25])

    segments_payload = {
        "metadata": {
            "image_width": width,
            "image_height": height,
            "scale": scale,
            "outline_target": outline_target,
            "region_target": region_target,
            "gray_levels": gray_levels,
            "equation_count": len(all_equations),
            "outline_equation_count": len(outline_layer.equations),
            "fill_equation_count": len(fill_layer.equations),
        },
        "segments": outline_segments + fill_segments,
    }
    (out_dir / "segments.json").write_text(json.dumps(segments_payload, indent=2), encoding="utf-8")
    (out_dir / "regions.json").write_text(
        json.dumps(
            _region_payload(
                regions,
                image_size=(width, height),
                scale=scale,
                gray_levels=gray_levels,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_desmos_exports(out_dir, outline_segments + fill_segments)

    render_segments(
        outline_segments,
        out_dir / "outline_preview.png",
        image_size=(width, height),
        scale=scale,
    )
    render_filled_regions(
        region_contours,
        label_gray,
        outline_segments,
        out_dir / "filled_preview.png",
        image_size=(width, height),
        scale=scale,
        base_gray_image=source_grayscale_fill(clean, foreground),
    )

    return LucasPipelineResult(
        out_dir=out_dir,
        total_equations=len(all_equations),
        outline_equations=len(outline_layer.equations),
        fill_equations=len(fill_layer.equations),
    )
