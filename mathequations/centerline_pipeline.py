"""Centerline V2 line-art reconstruction pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from .centerline_bridge import find_bridge_candidates, select_bridges
from .centerline_graph import (
    Endpoint,
    RawBranch,
    StrokeChain,
    build_skeleton_graph,
    build_stroke_chains,
    pair_junction_branches,
    trace_raw_branches,
)
from .curve_equations import segments_payload
from .curve_fit import fit_stroke_chains
from .curve_render import (
    render_curve_segment_previews,
    render_endpoint_overlay,
    render_stroke_chains,
    render_stroke_paths,
)
from .geometry import map_contours_to_cartesian
from .image_processing import foreground_bbox, load_image
from .lineart_preprocess import (
    build_local_line_mask,
    clean_lineart_image,
    line_mask_diagnostics,
    prepare_highres_lineart,
    skeletonize_line_mask,
)
from .lineart_pipeline import LineartPipelineResult, _desmos_expressions, _stroke_metadata, _write_text


def _downsample_mask(mask: Any, *, image_size: tuple[int, int]) -> Any:
    width, height = image_size
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _scale_point(point: tuple[float, float], factor: float) -> tuple[float, float]:
    return (point[0] * factor, point[1] * factor)


def _scale_branches(branches: list[RawBranch], *, factor: float) -> list[RawBranch]:
    return [
        RawBranch(
            branch.branch_id,
            [_scale_point(point, factor) for point in branch.points],
            start_endpoint_id=branch.start_endpoint_id,
            end_endpoint_id=branch.end_endpoint_id,
            start_junction_id=branch.start_junction_id,
            end_junction_id=branch.end_junction_id,
            closed=branch.closed,
        )
        for branch in branches
    ]


def _scale_chains(chains: list[StrokeChain], *, factor: float) -> list[StrokeChain]:
    return [
        StrokeChain(
            chain.stroke_id,
            [_scale_point(point, factor) for point in chain.points],
            branch_ids=chain.branch_ids,
            closed=chain.closed,
            diagnostics=chain.diagnostics,
        )
        for chain in chains
    ]


def _scale_endpoints(endpoints: list[Endpoint], *, factor: float) -> list[Endpoint]:
    return [
        Endpoint(
            endpoint.endpoint_id,
            _scale_point(endpoint.point, factor),
            (round(endpoint.pixel[0] * factor), round(endpoint.pixel[1] * factor)),
            endpoint.degree,
        )
        for endpoint in endpoints
    ]


def _chain_to_cartesian(chain: StrokeChain, *, width: int, height: int, scale: float) -> StrokeChain:
    contour = [[point for point in chain.points]]
    mapped = map_contours_to_cartesian(contour, width=width, height=height, scale=scale)[0]
    return StrokeChain(chain.stroke_id, mapped, branch_ids=chain.branch_ids, closed=chain.closed)


def _bridge_records(candidates: list[Any], accepted: list[Any]) -> list[dict[str, Any]]:
    accepted_keys = {
        (min(bridge.endpoint_a_id, bridge.endpoint_b_id), max(bridge.endpoint_a_id, bridge.endpoint_b_id))
        for bridge in accepted
    }
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        record = asdict(candidate)
        key = (min(candidate.endpoint_a_id, candidate.endpoint_b_id), max(candidate.endpoint_a_id, candidate.endpoint_b_id))
        record["accepted"] = key in accepted_keys
        records.append(record)
    return records


def run_centerline_pipeline(
    *,
    image_path: Path,
    out_dir: Path,
    target: int = 1200,
    scale_width: float = 20.0,
    line_thickness: int = 1,
    fit_mode: str = "mixed",
    cleaned_input: Path | None = None,
    preprocess_scale: int = 4,
    render_scale: int = 4,
    max_bridge_gap: float = 16,
    bridge_angle_threshold: float = 45,
    local_threshold: str = "sauvola",
    keep_diagnostics: bool = False,
) -> LineartPipelineResult:
    """Run Centerline V2 and write previews, diagnostics, equations, and JSON."""
    if preprocess_scale < 1:
        raise ValueError("preprocess_scale must be at least 1")
    if render_scale < 1:
        raise ValueError("render_scale must be at least 1")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(cleaned_input) if cleaned_input is not None else Path(image_path)
    image = load_image(source_path)
    height, width = image.shape[:2]

    clean = clean_lineart_image(image)
    cv2.imwrite(str(out_dir / "clean_input.png"), clean)
    highres = prepare_highres_lineart(image, scale=preprocess_scale)
    cv2.imwrite(str(out_dir / "clean_input_highres.png"), highres)

    mask_highres = build_local_line_mask(highres, threshold_mode=local_threshold, min_component_area=2)
    cv2.imwrite(str(out_dir / "line_mask_highres.png"), mask_highres)
    mask = _downsample_mask(mask_highres, image_size=(width, height))
    cv2.imwrite(str(out_dir / "line_mask.png"), mask)

    skeleton_highres = skeletonize_line_mask(mask_highres)
    cv2.imwrite(str(out_dir / "skeleton_highres.png"), skeleton_highres)
    skeleton = _downsample_mask(skeleton_highres, image_size=(width, height))
    cv2.imwrite(str(out_dir / "skeleton.png"), skeleton)

    graph = build_skeleton_graph(skeleton_highres, mask=mask_highres)
    raw_branches = trace_raw_branches(graph)
    gray_highres = cv2.cvtColor(highres, cv2.COLOR_BGR2GRAY)
    bridge_candidates = find_bridge_candidates(
        graph,
        max_gap=max_bridge_gap,
        angle_threshold_degrees=bridge_angle_threshold,
        mask=mask_highres,
        gray=gray_highres,
    )
    accepted_bridges = select_bridges(bridge_candidates)
    junction_pairs = pair_junction_branches(graph)
    chains_highres = build_stroke_chains(graph, raw_branches, accepted_bridges, junction_pairs)
    if not chains_highres:
        raise ValueError("No centerline stroke chains found")

    scale_factor = 1.0 / preprocess_scale
    base_branches = _scale_branches(raw_branches, factor=scale_factor)
    base_chains = _scale_chains(chains_highres, factor=scale_factor)
    base_endpoints = _scale_endpoints(graph.endpoints, factor=scale_factor)

    render_stroke_paths(
        base_branches,
        out_dir / "stroke_preview.png",
        image_size=(width, height),
        line_thickness=line_thickness,
    )
    if keep_diagnostics:
        render_endpoint_overlay(
            base_endpoints,
            accepted_bridges,
            out_dir / "endpoint_overlay.png",
            image_size=(width, height),
            line_thickness=line_thickness,
        )
        render_stroke_chains(
            base_chains,
            out_dir / "bridged_strokes_preview.png",
            image_size=(width, height),
            line_thickness=line_thickness,
        )
        (out_dir / "bridge_candidates.json").write_text(
            json.dumps(_bridge_records(bridge_candidates, accepted_bridges), indent=2),
            encoding="utf-8",
        )
        (out_dir / "mask_diagnostics.json").write_text(
            json.dumps(line_mask_diagnostics(mask_highres), indent=2),
            encoding="utf-8",
        )

    x_min, _, x_max, _ = foreground_bbox(mask)
    foreground_width = max(1, x_max - x_min + 1)
    scale = scale_width / foreground_width
    cartesian_chains = [
        _chain_to_cartesian(chain, width=width, height=height, scale=scale)
        for chain in base_chains
    ]
    segments = fit_stroke_chains(cartesian_chains, target=target, fit_mode=fit_mode)

    preview_paths = render_curve_segment_previews(
        segments,
        out_dir,
        image_size=(width, height),
        scale=scale,
        render_scale=render_scale,
        line_thickness=line_thickness,
    )
    function_preview_path = preview_paths["preview"]

    equations_path = out_dir / "equations.txt"
    _write_text(equations_path, [str(segment["equation"]) for segment in segments])
    _write_text(out_dir / "desmos_latex.txt", [str(segment["latex"]) for segment in segments])
    selected_stride = max(1, len(segments) // 20)
    _write_text(
        out_dir / "selected_equations.txt",
        [str(segment["equation"]) for segment in segments[::selected_stride][:20]],
    )
    (out_dir / "desmos_expressions.json").write_text(
        json.dumps(_desmos_expressions(segments), ensure_ascii=False),
        encoding="utf-8",
    )

    trace_metadata = {
        "trace_mode": "centerline-v2",
        "preprocess_scale": preprocess_scale,
        "render_scale": render_scale,
        "raw_branch_count": len(raw_branches),
        "endpoint_count": len(graph.endpoints),
        "accepted_bridge_count": len(accepted_bridges),
        "final_stroke_chain_count": len(chains_highres),
        "dropped_fragment_count": int(graph.diagnostics.get("dropped_fragment_count", 0)),
        "equation_count": len(segments),
    }
    if keep_diagnostics:
        (out_dir / "trace_diagnostics.json").write_text(
            json.dumps({**graph.diagnostics, **trace_metadata}, indent=2),
            encoding="utf-8",
        )

    payload = segments_payload(
        segments,
        strokes=_stroke_metadata(cartesian_chains),
        image_size=(width, height),
        scale=scale,
        target=target,
        fit_mode=fit_mode,
        extra_metadata=trace_metadata,
    )
    json_path = out_dir / "segments.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return LineartPipelineResult(
        out_dir=out_dir,
        equation_count=len(segments),
        stroke_count=len(chains_highres),
        function_preview_path=function_preview_path,
        json_path=json_path,
        equations_path=equations_path,
        trace_mode="centerline-v2",
        raw_branch_count=len(raw_branches),
        endpoint_count=len(graph.endpoints),
        accepted_bridge_count=len(accepted_bridges),
        final_chain_count=len(chains_highres),
    )
