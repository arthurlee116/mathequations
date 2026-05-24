"""Render DOCX point-cloud coordinates as colored Desmos line segments."""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from .color_masks import KOALA_COLOR_SPECS
from .curve_equations import linear_segment, segments_payload
from .curve_render import render_curve_segment_previews
from .desmos_export import write_desmos_html
from .geometry import pixel_to_cartesian
from .image_processing import foreground_bbox, load_image, normalize_white_background
from .lineart_pipeline import _write_text

PixelPoint = tuple[int, int]
ColorRun = tuple[int, int, int, str, str]

COORDINATE_RE = re.compile(r"\((-?\d+),\s*(-?\d+)\)")


@dataclass(frozen=True)
class PointCloudPipelineResult:
    out_dir: Path
    point_count: int
    equation_count: int
    color_counts: dict[str, int]
    function_preview_path: Path
    direct_preview_path: Path
    json_path: Path
    equations_path: Path


def extract_docx_points(docx_path: Path | str) -> set[PixelPoint]:
    """Extract ``(x, y_pixel)`` points from DOCX text like ``(410, -365)``."""
    with zipfile.ZipFile(docx_path) as archive:
        xml = archive.read("word/document.xml")
    text = "".join(ET.fromstring(xml).itertext())
    return {(int(x), -int(y)) for x, y in COORDINATE_RE.findall(text)}


def source_foreground_points(image: np.ndarray, *, threshold: int = 250) -> set[PixelPoint]:
    """Return non-white source pixels so incomplete DOCX clouds can be repaired."""
    mask = np.any(image < threshold, axis=2)
    ys, xs = np.where(mask)
    return {(int(x), int(y)) for y, x in zip(ys, xs)}


def _clean_binary_mask(
    mask: np.ndarray,
    *,
    clean_kernel_size: int,
    min_component_area: int,
) -> np.ndarray:
    if clean_kernel_size < 1:
        raise ValueError("clean_kernel_size must be at least 1")
    if min_component_area < 1:
        raise ValueError("min_component_area must be at least 1")
    cleaned = mask.copy()
    if clean_kernel_size > 1:
        kernel = np.ones((clean_kernel_size, clean_kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((cleaned > 0).astype(np.uint8), 8)
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_area:
            filtered[labels == label] = 255
    return filtered


def _classify_color(image: np.ndarray, point: PixelPoint) -> tuple[str, str]:
    x, y = point
    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return ("black", KOALA_COLOR_SPECS["black"].hex_color)
    blue, green, red = [int(value) for value in image[y, x]]
    hsv = cv2.cvtColor(np.uint8([[[blue, green, red]]]), cv2.COLOR_BGR2HSV)[0, 0]
    hue, saturation, value = [int(item) for item in hsv]
    if 15 <= hue <= 45 and saturation > 30 and value > 120:
        return ("yellow", KOALA_COLOR_SPECS["yellow"].hex_color)
    if red > green + 20 and red > blue + 10 and saturation > 30 and red > 105:
        return ("pink", KOALA_COLOR_SPECS["pink"].hex_color)
    return ("black", KOALA_COLOR_SPECS["black"].hex_color)


def clean_colored_points(
    points: Iterable[PixelPoint],
    image: np.ndarray,
    *,
    clean_kernel_size: int = 3,
    min_component_area: int = 6,
) -> set[PixelPoint]:
    """Remove isolated edge debris from classified point-cloud masks."""
    height, width = image.shape[:2]
    masks = {name: np.zeros((height, width), dtype=np.uint8) for name in KOALA_COLOR_SPECS}
    for point in points:
        x, y = point
        if 0 <= x < width and 0 <= y < height:
            color_name, _ = _classify_color(image, point)
            masks[color_name][y, x] = 255

    cleaned_points: set[PixelPoint] = set()
    for mask in masks.values():
        cleaned = _clean_binary_mask(
            mask,
            clean_kernel_size=clean_kernel_size,
            min_component_area=min_component_area,
        )
        ys, xs = np.where(cleaned > 0)
        cleaned_points.update((int(x), int(y)) for y, x in zip(ys, xs))
    return cleaned_points


def colored_horizontal_runs(points: Iterable[PixelPoint], image: np.ndarray) -> list[ColorRun]:
    """Group point-cloud pixels into same-row, same-color horizontal runs."""
    rows: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for point in points:
        x, y = point
        color_name, color = _classify_color(image, point)
        rows[(y, color_name, color)].append(x)

    runs: list[ColorRun] = []
    for (y, color_name, color), xs in sorted(rows.items()):
        ordered = sorted(set(xs))
        if not ordered:
            continue
        start = previous = ordered[0]
        for x in ordered[1:]:
            if x == previous + 1:
                previous = x
                continue
            runs.append((start, previous, y, color_name, color))
            start = previous = x
        runs.append((start, previous, y, color_name, color))
    return runs


def _run_to_segment(
    run: ColorRun,
    *,
    segment_id: int,
    width: int,
    height: int,
    scale: float,
) -> dict[str, Any]:
    x_start, x_end, y, color_name, color = run
    start_pixel = (float(x_start) - 0.45, float(y))
    end_pixel = (float(x_end) + 0.45, float(y))
    start = pixel_to_cartesian(start_pixel, width=width, height=height, scale=scale)
    end = pixel_to_cartesian(end_pixel, width=width, height=height, scale=scale)
    segment = linear_segment(segment_id, segment_id, start, end, source_points=[start, end])
    segment["color_name"] = color_name
    segment["color"] = color
    segment["point_cloud_run"] = {"x_start": x_start, "x_end": x_end, "y": y}
    return segment


def _stroke_metadata(runs: list[ColorRun], *, width: int, height: int, scale: float) -> list[dict[str, Any]]:
    strokes: list[dict[str, Any]] = []
    for index, (x_start, x_end, y, color_name, _color) in enumerate(runs, start=1):
        start = pixel_to_cartesian((float(x_start), float(y)), width=width, height=height, scale=scale)
        end = pixel_to_cartesian((float(x_end), float(y)), width=width, height=height, scale=scale)
        strokes.append(
            {
                "stroke_id": index,
                "color_name": color_name,
                "point_count": x_end - x_start + 1,
                "length": float(x_end - x_start + 1),
                "closed": False,
                "bbox": {
                    "x_min": min(start[0], end[0]),
                    "y_min": min(start[1], end[1]),
                    "x_max": max(start[0], end[0]),
                    "y_max": max(start[1], end[1]),
                },
            }
        )
    return strokes


def _desmos_expressions(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"lineart_{index}",
            "latex": str(segment["latex"]),
            "color": str(segment.get("color", "#000000")),
            "lineWidth": "1",
        }
        for index, segment in enumerate(segments, start=1)
    ]


def _write_direct_preview(points: set[PixelPoint], source: np.ndarray, path: Path) -> None:
    canvas = np.full_like(source, 255)
    height, width = source.shape[:2]
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            color_name, color = _classify_color(source, (x, y))
            if color_name == "yellow":
                canvas[y, x] = (41, 213, 245)
            elif color_name == "pink":
                canvas[y, x] = (106, 75, 210)
            else:
                canvas[y, x] = (0, 0, 0)
    cv2.imwrite(str(path), canvas)


def run_point_cloud_pipeline(
    *,
    image_path: Path,
    reference_docx: Path,
    out_dir: Path,
    scale_width: float = 20.0,
    render_scale: int = 4,
    clean_kernel_size: int = 3,
    min_component_area: int = 6,
) -> PointCloudPipelineResult:
    """Render DOCX coordinate points directly as colored horizontal expressions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = normalize_white_background(load_image(Path(image_path)))
    height, width = source.shape[:2]
    cv2.imwrite(str(out_dir / "clean_input.png"), source)

    docx_points = extract_docx_points(reference_docx)
    source_points = source_foreground_points(source)
    raw_points = docx_points | source_points
    points = clean_colored_points(
        raw_points,
        source,
        clean_kernel_size=clean_kernel_size,
        min_component_area=min_component_area,
    )
    if not points:
        raise ValueError("reference DOCX contains no coordinate points")
    point_mask = np.zeros((height, width), dtype=np.uint8)
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            point_mask[y, x] = 255
    cv2.imwrite(str(out_dir / "point_cloud_mask.png"), point_mask)

    x_min, _, x_max, _ = foreground_bbox(point_mask)
    scale = scale_width / max(1, x_max - x_min + 1)

    runs = colored_horizontal_runs(points, source)
    segments = [
        _run_to_segment(run, segment_id=index, width=width, height=height, scale=scale)
        for index, run in enumerate(runs, start=1)
    ]
    color_counts = dict(Counter(segment["color_name"] for segment in segments))
    direct_preview_path = out_dir / "point_cloud_direct.png"
    _write_direct_preview(points, source, direct_preview_path)
    preview_paths = render_curve_segment_previews(
        segments,
        out_dir,
        image_size=(width, height),
        scale=scale,
        render_scale=render_scale,
        line_thickness=1,
    )

    equations_path = out_dir / "equations.txt"
    _write_text(equations_path, [str(segment["equation"]) for segment in segments])
    _write_text(out_dir / "desmos_latex.txt", [str(segment["latex"]) for segment in segments])
    selected_stride = max(1, len(segments) // 20)
    _write_text(
        out_dir / "selected_equations.txt",
        [str(segment["equation"]) for segment in segments[::selected_stride][:20]],
    )
    expressions_path = out_dir / "desmos_expressions.json"
    expressions_path.write_text(json.dumps(_desmos_expressions(segments), ensure_ascii=False), encoding="utf-8")

    payload = segments_payload(
        segments,
        strokes=_stroke_metadata(runs, width=width, height=height, scale=scale),
        image_size=(width, height),
        scale=scale,
        target=len(segments),
        fit_mode="point-cloud-runs",
        extra_metadata={
            "trace_mode": "docx-point-cloud",
            "point_count": len(points),
            "docx_point_count": len(docx_points),
            "source_foreground_point_count": len(source_points),
            "raw_union_point_count": len(raw_points),
            "source_repair_point_count": len(source_points - docx_points),
            "cleaned_point_count": len(points),
            "clean_kernel_size": clean_kernel_size,
            "min_component_area": min_component_area,
            "render_scale": render_scale,
            "color_counts": color_counts,
            "reference_docx": str(reference_docx),
        },
    )
    json_path = out_dir / "segments.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_desmos_html(
        expressions_path,
        out_dir / "desmos.html",
        segments_path=json_path,
        title="Koala DOCX point cloud",
    )
    return PointCloudPipelineResult(
        out_dir=out_dir,
        point_count=len(points),
        equation_count=len(segments),
        color_counts=color_counts,
        function_preview_path=preview_paths["preview"],
        direct_preview_path=direct_preview_path,
        json_path=json_path,
        equations_path=equations_path,
    )
