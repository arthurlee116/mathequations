"""Patch small facial features from reference DOCX point clouds into line-art output."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

from .curve_equations import linear_segment
from .curve_render import render_curve_segment_previews, sample_segment_points
from .lineart_pipeline import _desmos_expressions, _write_text

PixelPoint = tuple[int, int]
PixelBox = tuple[int, int, int, int]
Point = tuple[float, float]

COORDINATE_RE = re.compile(r"\((-?\d+),\s*(-?\d+)\)")


@dataclass(frozen=True)
class FeatureSpec:
    """A named feature region in native image-pixel coordinates."""

    name: str
    box: PixelBox


DEFAULT_FEATURES = [
    FeatureSpec("nose", (405, 360, 416, 373)),
    FeatureSpec("mouth", (385, 365, 435, 400)),
]
DEFAULT_REMOVE_BOXES = [
    (405, 360, 416, 373),
    (392, 378, 428, 400),
]


def extract_docx_points(docx_path: Path | str) -> set[PixelPoint]:
    """Extract ``(x, y_pixel)`` points from Word paragraphs like ``(410, -365)``."""
    with zipfile.ZipFile(docx_path) as archive:
        xml = archive.read("word/document.xml")
    text = "".join(ET.fromstring(xml).itertext())
    return {(int(x), -int(y)) for x, y in COORDINATE_RE.findall(text)}


def _points_in_box(points: Iterable[PixelPoint], box: PixelBox) -> set[PixelPoint]:
    x_min, y_min, x_max, y_max = box
    return {(x, y) for x, y in points if x_min <= x <= x_max and y_min <= y <= y_max}


def _connected_components(points: set[PixelPoint]) -> list[set[PixelPoint]]:
    seen: set[PixelPoint] = set()
    components: list[set[PixelPoint]] = []
    for point in points:
        if point in seen:
            continue
        queue = deque([point])
        seen.add(point)
        component: set[PixelPoint] = set()
        while queue:
            x, y = queue.popleft()
            component.add((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if neighbor in points and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
        components.append(component)
    return sorted(components, key=len, reverse=True)


def _pixel_to_cartesian(pixel: tuple[float, float], *, width: int, height: int, scale: float) -> Point:
    x, y = pixel
    return ((x - width / 2) * scale, (height / 2 - y) * scale)


def _cartesian_to_pixel(point: Point, *, width: int, height: int, scale: float) -> tuple[float, float]:
    x, y = point
    return (x / scale + width / 2, height / 2 - y / scale)


def _horizontal_runs(points: set[PixelPoint]) -> list[tuple[int, int, int]]:
    rows: dict[int, list[int]] = defaultdict(list)
    for x, y in points:
        rows[y].append(x)

    runs: list[tuple[int, int, int]] = []
    for y in sorted(rows):
        xs = sorted(rows[y])
        start = previous = xs[0]
        for x in xs[1:]:
            if x == previous + 1:
                previous = x
                continue
            runs.append((start, previous, y))
            start = previous = x
        runs.append((start, previous, y))
    return runs


def _feature_segments(
    points: set[PixelPoint],
    *,
    name: str,
    width: int,
    height: int,
    scale: float,
    start_segment_id: int,
    stroke_id: int,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    next_segment_id = start_segment_id
    for x_start, x_end, y in _horizontal_runs(points):
        if x_start == x_end:
            start_pixel = (x_start - 0.25, float(y))
            end_pixel = (x_end + 0.25, float(y))
        else:
            start_pixel = (float(x_start), float(y))
            end_pixel = (float(x_end), float(y))
        start = _pixel_to_cartesian(start_pixel, width=width, height=height, scale=scale)
        end = _pixel_to_cartesian(end_pixel, width=width, height=height, scale=scale)
        segment = linear_segment(
            next_segment_id,
            stroke_id,
            start,
            end,
            source_points=[start, end],
        )
        segment["feature_override"] = name
        segments.append(segment)
        next_segment_id += 1
    return segments


def _intersects(first: PixelBox, second: PixelBox) -> bool:
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


def _segment_pixel_bbox(segment: dict[str, Any], *, width: int, height: int, scale: float) -> PixelBox:
    pixels = [
        _cartesian_to_pixel(point, width=width, height=height, scale=scale)
        for point in sample_segment_points(segment, samples=64)
    ]
    xs = [point[0] for point in pixels]
    ys = [point[1] for point in pixels]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _segment_hits_boxes(
    segment: dict[str, Any],
    boxes: list[PixelBox],
    *,
    width: int,
    height: int,
    scale: float,
) -> bool:
    bbox = _segment_pixel_bbox(segment, width=width, height=height, scale=scale)
    return any(_intersects(bbox, box) for box in boxes)


def _component_bbox(points: set[PixelPoint], *, width: int, height: int, scale: float) -> dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, y0 = _pixel_to_cartesian((min(xs), max(ys)), width=width, height=height, scale=scale)
    x1, y1 = _pixel_to_cartesian((max(xs), min(ys)), width=width, height=height, scale=scale)
    return {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}


def patch_face_feature_payload(
    payload: dict[str, Any],
    reference_points: set[PixelPoint],
    *,
    features: list[FeatureSpec] | None = None,
    remove_boxes: list[PixelBox] | None = None,
) -> dict[str, Any]:
    """Replace generated feature segments with exact horizontal runs from reference points."""
    metadata = dict(payload["metadata"])
    width = int(metadata["image_width"])
    height = int(metadata["image_height"])
    scale = float(metadata["scale"])
    feature_specs = features or DEFAULT_FEATURES
    boxes = remove_boxes or DEFAULT_REMOVE_BOXES

    kept_segments = [
        segment
        for segment in payload.get("segments", [])
        if not _segment_hits_boxes(segment, boxes, width=width, height=height, scale=scale)
    ]
    removed_count = len(payload.get("segments", [])) - len(kept_segments)

    next_segment_id = max([int(segment.get("segment_id", 0)) for segment in kept_segments] + [0]) + 1
    next_stroke_id = max([int(stroke.get("stroke_id", 0)) for stroke in payload.get("strokes", [])] + [0]) + 1
    added_segments: list[dict[str, Any]] = []
    added_strokes: list[dict[str, Any]] = []
    feature_counts: dict[str, int] = {}

    for feature in feature_specs:
        candidates = _points_in_box(reference_points, feature.box)
        components = _connected_components(candidates)
        if not components:
            feature_counts[feature.name] = 0
            continue
        component = components[0]
        segments = _feature_segments(
            component,
            name=feature.name,
            width=width,
            height=height,
            scale=scale,
            start_segment_id=next_segment_id,
            stroke_id=next_stroke_id,
        )
        added_segments.extend(segments)
        added_strokes.append(
            {
                "stroke_id": next_stroke_id,
                "point_count": len(component),
                "length": len(component),
                "closed": False,
                "bbox": _component_bbox(component, width=width, height=height, scale=scale),
                "feature_override": feature.name,
            }
        )
        feature_counts[feature.name] = len(component)
        next_segment_id += len(segments)
        next_stroke_id += 1

    patched = dict(payload)
    metadata["equation_count"] = len(kept_segments) + len(added_segments)
    metadata["face_feature_patch"] = {
        "removed_segment_count": removed_count,
        "added_segment_count": len(added_segments),
        "feature_point_counts": feature_counts,
    }
    patched["metadata"] = metadata
    patched["strokes"] = list(payload.get("strokes", [])) + added_strokes
    patched["segments"] = kept_segments + added_segments
    return patched


def feature_function_lines(segments: list[dict[str, Any]]) -> list[str]:
    """Return a TXT-friendly list of patched feature functions and domains."""
    feature_segments = [segment for segment in segments if segment.get("feature_override")]
    lines = ["# Face feature override functions", f"# count={len(feature_segments)}"]
    for index, segment in enumerate(feature_segments, start=1):
        restriction = segment.get("restriction", {})
        variable = restriction.get("variable", "x")
        lower = restriction.get("min", "")
        upper = restriction.get("max", "")
        equation = str(segment.get("equation", "")).split(" {", 1)[0]
        lines.append(
            f"{index}. segment_id={segment.get('segment_id')} "
            f"feature={segment.get('feature_override')} "
            f"function: {equation}; domain: {lower} <= {variable} <= {upper}"
        )
    return lines


def patch_face_feature_output(
    *,
    segments_path: Path,
    reference_docx: Path,
    out_dir: Path,
    source_dir: Path | None = None,
    features: list[FeatureSpec] | None = None,
    remove_boxes: list[PixelBox] | None = None,
) -> Path:
    """Patch an existing line-art output directory and rewrite renderable artifacts."""
    source = source_dir or segments_path.parent
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(source, out_dir)

    payload = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    patched = patch_face_feature_payload(
        payload,
        extract_docx_points(reference_docx),
        features=features,
        remove_boxes=remove_boxes,
    )
    segments = patched["segments"]
    metadata = patched["metadata"]
    image_size = (int(metadata["image_width"]), int(metadata["image_height"]))
    scale = float(metadata["scale"])
    render_scale = int(metadata.get("render_scale", 4))

    (out_dir / "segments.json").write_text(json.dumps(patched, indent=2), encoding="utf-8")
    _write_text(out_dir / "equations.txt", [str(segment["equation"]) for segment in segments])
    _write_text(out_dir / "desmos_latex.txt", [str(segment["latex"]) for segment in segments])
    _write_text(out_dir / "face_feature_functions.txt", feature_function_lines(segments))
    selected_stride = max(1, len(segments) // 20)
    _write_text(
        out_dir / "selected_equations.txt",
        [str(segment["equation"]) for segment in segments[::selected_stride][:20]],
    )
    (out_dir / "desmos_expressions.json").write_text(
        json.dumps(_desmos_expressions(segments), ensure_ascii=False),
        encoding="utf-8",
    )
    render_curve_segment_previews(
        segments,
        out_dir,
        image_size=image_size,
        scale=scale,
        render_scale=render_scale,
    )
    return out_dir / "function_preview.png"
