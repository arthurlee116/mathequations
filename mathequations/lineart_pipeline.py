"""End-to-end line-art-to-equations pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .curve_equations import segments_payload
from .curve_fit import fit_stroke_paths
from .curve_render import render_curve_segments, render_stroke_paths
from .geometry import map_contours_to_cartesian
from .image_processing import foreground_bbox, load_image
from .lineart_preprocess import build_line_mask, clean_lineart_image, skeletonize_line_mask
from .skeleton_graph import StrokePath, trace_skeleton_paths


@dataclass(frozen=True)
class LineartPipelineResult:
    """Paths and counts produced by one line-art pipeline run."""

    out_dir: Path
    equation_count: int
    stroke_count: int
    function_preview_path: Path
    json_path: Path
    equations_path: Path


def _write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stroke_to_cartesian(stroke: StrokePath, *, width: int, height: int, scale: float) -> StrokePath:
    contour = [[point for point in stroke.points]]
    mapped = map_contours_to_cartesian(contour, width=width, height=height, scale=scale)[0]
    return StrokePath(stroke.stroke_id, mapped, stroke.closed)


def _stroke_metadata(strokes: list[StrokePath]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stroke in strokes:
        xs = [point[0] for point in stroke.points]
        ys = [point[1] for point in stroke.points]
        records.append(
            {
                "stroke_id": stroke.stroke_id,
                "point_count": len(stroke.points),
                "length": stroke.length,
                "closed": stroke.closed,
                "bbox": {
                    "x_min": min(xs),
                    "y_min": min(ys),
                    "x_max": max(xs),
                    "y_max": max(ys),
                },
            }
        )
    return records


def _desmos_expressions(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"lineart_{index}",
            "latex": str(segment["latex"]),
            "color": "#000000",
            "lineWidth": "1",
        }
        for index, segment in enumerate(segments, start=1)
    ]


def run_lineart_pipeline(
    *,
    image_path: Path,
    out_dir: Path,
    target: int = 1200,
    scale_width: float = 20.0,
    threshold_mode: str = "auto",
    line_thickness: int = 1,
    fit_mode: str = "mixed",
    cleaned_input: Path | None = None,
) -> LineartPipelineResult:
    """Run the line-art conversion and write previews, equations, and JSON."""
    if target < 1:
        raise ValueError("target must be at least 1")
    if fit_mode not in {"linear", "quadratic", "mixed"}:
        raise ValueError("fit_mode must be one of: linear, quadratic, mixed")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(cleaned_input) if cleaned_input is not None else Path(image_path)
    image = load_image(source_path)
    clean = clean_lineart_image(image)
    height, width = clean.shape[:2]
    cv2.imwrite(str(out_dir / "clean_input.png"), clean)

    mask = build_line_mask(clean, threshold_mode=threshold_mode)
    cv2.imwrite(str(out_dir / "line_mask.png"), mask)
    skeleton = skeletonize_line_mask(mask)
    cv2.imwrite(str(out_dir / "skeleton.png"), skeleton)

    x_min, _, x_max, _ = foreground_bbox(mask)
    foreground_width = max(1, x_max - x_min + 1)
    scale = scale_width / foreground_width

    pixel_strokes = trace_skeleton_paths(skeleton, min_length=8)
    if not pixel_strokes:
        raise ValueError("No line-art strokes found")

    render_stroke_paths(
        pixel_strokes,
        out_dir / "stroke_preview.png",
        image_size=(width, height),
        line_thickness=line_thickness,
    )

    cartesian_strokes = [
        _stroke_to_cartesian(stroke, width=width, height=height, scale=scale)
        for stroke in pixel_strokes
    ]
    segments = fit_stroke_paths(cartesian_strokes, target=target, fit_mode=fit_mode)

    function_preview_path = out_dir / "function_preview.png"
    render_curve_segments(
        segments,
        function_preview_path,
        image_size=(width, height),
        scale=scale,
        line_thickness=line_thickness,
    )

    equations_path = out_dir / "equations.txt"
    _write_text(equations_path, [str(segment["equation"]) for segment in segments])
    _write_text(out_dir / "desmos_latex.txt", [str(segment["latex"]) for segment in segments])

    selected_stride = max(1, len(segments) // 20)
    _write_text(
        out_dir / "selected_equations.txt",
        [str(segment["equation"]) for segment in segments[::selected_stride][:20]],
    )

    expressions = _desmos_expressions(segments)
    (out_dir / "desmos_expressions.json").write_text(
        json.dumps(expressions, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = segments_payload(
        segments,
        strokes=_stroke_metadata(cartesian_strokes),
        image_size=(width, height),
        scale=scale,
        target=target,
        fit_mode=fit_mode,
    )
    json_path = out_dir / "segments.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return LineartPipelineResult(
        out_dir=out_dir,
        equation_count=len(segments),
        stroke_count=len(pixel_strokes),
        function_preview_path=function_preview_path,
        json_path=json_path,
        equations_path=equations_path,
    )
