from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from .equations import equations_from_contours, segments_to_jsonable
from .geometry import (
    allocate_targets,
    contour_to_points,
    map_contours_to_cartesian,
    polyline_length,
    resample_closed_points,
    simplify_contour,
)
from .image_processing import (
    find_foreground_contours,
    foreground_bbox,
    foreground_mask,
    load_image,
)
from .render import render_segments


@dataclass(frozen=True)
class PipelineResult:
    out_dir: Path
    segment_count: int
    mask_path: Path
    preview_path: Path
    equations_path: Path
    json_path: Path
    selected_path: Path


def _contours_to_target_points(contours, target: int) -> list[list[tuple[float, float]]]:
    simplified = [simplify_contour(contour) for contour in contours]
    lengths = [polyline_length(points, closed=True) for points in simplified]
    allocations = allocate_targets(lengths, target)

    output: list[list[tuple[float, float]]] = []
    for contour, points, allocated in zip(contours, simplified, allocations):
        lower_bound = max(3, round(allocated * 0.75))
        upper_bound = max(3, round(allocated * 1.25))
        if lower_bound <= len(points) <= upper_bound:
            output.append(points)
        elif len(points) > allocated:
            output.append(resample_closed_points(points, allocated))
        else:
            output.append(resample_closed_points(contour_to_points(contour), allocated))
    return output


def _write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    *,
    image_path: Path,
    out_dir: Path,
    target: int = 200,
    scale_width: float = 20.0,
    saturation_threshold: int = 40,
) -> PipelineResult:
    if target < 3:
        raise ValueError("target must be at least 3")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(Path(image_path))
    height, width = image.shape[:2]
    mask = foreground_mask(image, saturation_threshold=saturation_threshold)
    mask_path = out_dir / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    x_min, _, x_max, _ = foreground_bbox(mask)
    foreground_width = max(1, x_max - x_min + 1)
    scale = scale_width / foreground_width

    contours = find_foreground_contours(mask)
    if not contours:
        raise ValueError("No contours found")

    pixel_contours = _contours_to_target_points(contours, target)
    cartesian_contours = map_contours_to_cartesian(
        pixel_contours,
        width=width,
        height=height,
        scale=scale,
    )
    segments = equations_from_contours(cartesian_contours)

    equations = [segment["equation"] for segment in segments]
    equations_path = out_dir / "equations.txt"
    _write_text(equations_path, equations)

    selected_path = out_dir / "selected_equations.txt"
    stride = max(1, len(equations) // 10)
    selected = equations[::stride][:10]
    _write_text(selected_path, selected)

    json_path = out_dir / "segments.json"
    payload = segments_to_jsonable(
        segments,
        image_size=(width, height),
        scale=scale,
        target=target,
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    preview_path = out_dir / "preview.png"
    render_segments(
        segments,
        preview_path,
        image_size=(width, height),
        scale=scale,
    )

    return PipelineResult(
        out_dir=out_dir,
        segment_count=len(segments),
        mask_path=mask_path,
        preview_path=preview_path,
        equations_path=equations_path,
        json_path=json_path,
        selected_path=selected_path,
    )
