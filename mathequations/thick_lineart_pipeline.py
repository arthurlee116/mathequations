"""Koala-specific thick line-art pipeline using stacked offset expressions."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .centerline_graph import (
    StrokeChain,
    build_skeleton_graph,
    build_stroke_chains,
    pair_junction_branches,
    trace_raw_branches,
)
from .color_masks import (
    KOALA_COLOR_SPECS,
    ExtractedColorMask,
    extract_koala_color_masks,
    require_color_masks,
)
from .curve_equations import segments_payload
from .curve_fit import fit_stroke_chains
from .curve_render import render_curve_segment_previews
from .desmos_export import write_desmos_html
from .geometry import map_contours_to_cartesian
from .image_processing import foreground_bbox, load_image, normalize_white_background
from .lineart_pipeline import _write_text
from .lineart_preprocess import skeletonize_line_mask
from .offset_segments import expand_segment_offsets, offset_indices
from .stroke_width import stroke_distance_map, stroke_radius_by_id


@dataclass(frozen=True)
class ThickLineartPipelineResult:
    out_dir: Path
    equation_count: int
    source_segment_count: int
    color_counts: dict[str, int]
    function_preview_path: Path
    json_path: Path
    equations_path: Path


def _chain_to_cartesian(
    chain: StrokeChain,
    *,
    width: int,
    height: int,
    scale: float,
) -> StrokeChain:
    contour = [[point for point in chain.points]]
    mapped = map_contours_to_cartesian(contour, width=width, height=height, scale=scale)[0]
    return StrokeChain(chain.stroke_id, mapped, branch_ids=chain.branch_ids, closed=chain.closed)


def _stroke_metadata(
    chains: list[StrokeChain],
    *,
    color_name: str,
    radii: dict[int, float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chain in chains:
        xs = [point[0] for point in chain.points]
        ys = [point[1] for point in chain.points]
        records.append(
            {
                "stroke_id": int(chain.stroke_id),
                "color_name": color_name,
                "point_count": len(chain.points),
                "length": float(chain.length),
                "closed": bool(chain.closed),
                "stroke_radius_pixels": float(radii.get(chain.stroke_id, 1.0)),
                "bbox": {
                    "x_min": min(xs),
                    "y_min": min(ys),
                    "x_max": max(xs),
                    "y_max": max(ys),
                },
            }
        )
    return records


def _trace_color_chains(mask: np.ndarray, *, min_chain_length: float) -> list[StrokeChain]:
    skeleton = skeletonize_line_mask(mask)
    graph = build_skeleton_graph(skeleton, mask=mask)
    raw_branches = trace_raw_branches(graph)
    pairs = pair_junction_branches(graph)
    return build_stroke_chains(graph, raw_branches, [], pairs, min_chain_length=min_chain_length)


def _combined_mask(masks: dict[str, ExtractedColorMask]) -> np.ndarray:
    combined: np.ndarray | None = None
    for color_mask in masks.values():
        combined = color_mask.mask.copy() if combined is None else cv2.bitwise_or(combined, color_mask.mask)
    if combined is None:
        raise ValueError("No color masks available")
    return combined


def _desmos_expressions(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"thick_lineart_{index}",
            "latex": str(segment["latex"]),
            "color": str(segment.get("color", "#000000")),
            "lineWidth": "1",
        }
        for index, segment in enumerate(segments, start=1)
    ]


def _write_outputs(
    *,
    out_dir: Path,
    payload: dict[str, Any],
    segments: list[dict[str, Any]],
    image_size: tuple[int, int],
    scale: float,
    render_scale: int,
) -> tuple[Path, Path]:
    json_path = out_dir / "segments.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
    preview_paths = render_curve_segment_previews(
        segments,
        out_dir,
        image_size=image_size,
        scale=scale,
        render_scale=render_scale,
        line_thickness=1,
    )
    preview = cv2.imread(str(preview_paths["preview"]), cv2.IMREAD_COLOR)
    if preview is None:
        raise ValueError(f"failed to read rendered preview: {preview_paths['preview']}")
    cv2.imwrite(str(out_dir / "stacked_preview.png"), preview)
    write_desmos_html(
        expressions_path,
        out_dir / "desmos.html",
        segments_path=json_path,
        title="Koala thick lineart",
    )
    return equations_path, preview_paths["preview"]


def run_thick_lineart_pipeline(
    *,
    image_path: Path,
    out_dir: Path,
    target: int = 1800,
    scale_width: float = 20.0,
    offset_step_pixels: float = 1.0,
    max_offsets: int = 9,
    render_scale: int = 4,
    min_component_area: int = 12,
    min_chain_length: float = 8.0,
    min_radius_pixels: float = 1.0,
    max_radius_pixels: float = 10.0,
    keep_diagnostics: bool = False,
) -> ThickLineartPipelineResult:
    """Run the koala thick-stroke equation export."""
    if target < 1:
        raise ValueError("target must be at least 1")
    offset_indices(
        max_offsets=max_offsets,
        radius_pixels=max_radius_pixels,
        offset_step_pixels=offset_step_pixels,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(Path(image_path))
    clean = normalize_white_background(image)
    height, width = clean.shape[:2]
    cv2.imwrite(str(out_dir / "clean_input.png"), clean)

    color_masks = extract_koala_color_masks(clean, min_component_area=min_component_area)
    require_color_masks(color_masks)
    combined = _combined_mask(color_masks)
    x_min, _, x_max, _ = foreground_bbox(combined)
    scale = scale_width / max(1, x_max - x_min + 1)

    all_source_segments: list[dict[str, Any]] = []
    all_offset_segments: list[dict[str, Any]] = []
    all_strokes: list[dict[str, Any]] = []
    radius_diagnostics: dict[str, Any] = {}
    next_segment_id = 1
    per_color_target = max(1, round(target / len(KOALA_COLOR_SPECS)))

    for color_name in KOALA_COLOR_SPECS:
        color_mask = color_masks[color_name]
        cv2.imwrite(str(out_dir / f"{color_name}_mask.png"), color_mask.mask)
        skeleton = skeletonize_line_mask(color_mask.mask)
        cv2.imwrite(str(out_dir / f"{color_name}_skeleton.png"), skeleton)
        pixel_chains = _trace_color_chains(color_mask.mask, min_chain_length=min_chain_length)
        if not pixel_chains:
            raise ValueError(f"No strokes found for required color: {color_name}")

        distance = stroke_distance_map(color_mask.mask)
        radii = stroke_radius_by_id(
            pixel_chains,
            distance,
            min_radius_pixels=min_radius_pixels,
            max_radius_pixels=max_radius_pixels,
        )
        radius_values = np.array(list(radii.values()), dtype=float)
        radius_diagnostics[color_name] = {
            "stroke_count": len(pixel_chains),
            "min_radius_pixels": float(radius_values.min()),
            "max_radius_pixels": float(radius_values.max()),
            "median_radius_pixels": float(np.median(radius_values)),
        }

        cartesian_chains = [
            _chain_to_cartesian(chain, width=width, height=height, scale=scale)
            for chain in pixel_chains
        ]
        all_strokes.extend(_stroke_metadata(cartesian_chains, color_name=color_name, radii=radii))
        source_segments = fit_stroke_chains(
            cartesian_chains,
            target=per_color_target,
            fit_mode="mixed",
            min_length=0.05,
        )
        for source_segment in source_segments:
            source_segment["source_segment_id"] = int(source_segment["segment_id"])
            source_segment["color_name"] = color_name
            source_segment["color"] = color_mask.hex_color
        all_source_segments.extend(source_segments)

        for source_segment in source_segments:
            stack = expand_segment_offsets(
                source_segment,
                radius_pixels=radii.get(int(source_segment["stroke_id"]), min_radius_pixels),
                scale=scale,
                offset_step_pixels=offset_step_pixels,
                max_offsets=max_offsets,
                color_name=color_name,
                color=color_mask.hex_color,
                start_segment_id=next_segment_id,
            )
            all_offset_segments.extend(stack)
            next_segment_id += len(stack)

    color_counts = dict(Counter(str(segment["color_name"]) for segment in all_offset_segments))
    metadata = {
        "trace_mode": "thick-lineart",
        "source_segment_count": len(all_source_segments),
        "offset_step_pixels": float(offset_step_pixels),
        "max_offsets": int(max_offsets),
        "render_scale": int(render_scale),
        "required_colors": list(KOALA_COLOR_SPECS),
        "color_counts": color_counts,
        "keep_diagnostics": bool(keep_diagnostics),
    }
    (out_dir / "stroke_width_diagnostics.json").write_text(
        json.dumps(radius_diagnostics, indent=2),
        encoding="utf-8",
    )

    payload = segments_payload(
        all_offset_segments,
        strokes=all_strokes,
        image_size=(width, height),
        scale=scale,
        target=target,
        fit_mode="mixed",
        extra_metadata=metadata,
    )
    equations_path, preview_path = _write_outputs(
        out_dir=out_dir,
        payload=payload,
        segments=all_offset_segments,
        image_size=(width, height),
        scale=scale,
        render_scale=render_scale,
    )

    return ThickLineartPipelineResult(
        out_dir=out_dir,
        equation_count=len(all_offset_segments),
        source_segment_count=len(all_source_segments),
        color_counts=color_counts,
        function_preview_path=preview_path,
        json_path=out_dir / "segments.json",
        equations_path=equations_path,
    )
