"""Command-line entry point for the image-to-equation pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from .lineart_pipeline import run_lineart_pipeline
from .lucas_pipeline import run_lucas_pipeline
from .lucas_vector_pipeline import run_lucas_vector_pipeline
from .face_feature_patch import patch_face_feature_output
from .pipeline import run_pipeline
from .point_cloud_pipeline import run_point_cloud_pipeline
from .thick_lineart_pipeline import run_thick_lineart_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by generic and Lucas-specific pipelines."""
    parser = argparse.ArgumentParser(
        description="Convert a clean emblem image into restricted linear equations."
    )
    subparsers = parser.add_subparsers(dest="command")

    lucas = subparsers.add_parser(
        "lucas",
        help="Convert the Lucas character image into layered math-art equations.",
    )
    lucas.add_argument("--input", required=True, type=Path, help="Input image path.")
    lucas.add_argument("--out", required=True, type=Path, help="Output directory.")
    lucas.add_argument(
        "--outline-target",
        default=1500,
        type=int,
        help="Approximate equation count for line details.",
    )
    lucas.add_argument(
        "--region-target",
        default=1200,
        type=int,
        help="Approximate equation count for region boundaries.",
    )
    lucas.add_argument(
        "--gray-levels",
        default=6,
        type=int,
        help="Number of grayscale fill levels.",
    )
    lucas.add_argument(
        "--scale-width",
        default=20.0,
        type=float,
        help="Coordinate-plane width for the foreground bounding box.",
    )

    lucas_vector = subparsers.add_parser(
        "lucas-vector",
        help="Reconstruct Lucas as clean semantic SVG/vector art before Desmos export.",
    )
    lucas_vector.add_argument("--input", required=True, type=Path, help="Input image path.")
    lucas_vector.add_argument("--out", required=True, type=Path, help="Output directory.")
    lucas_vector.add_argument(
        "--scale-width",
        default=20.0,
        type=float,
        help="Coordinate-plane width for the vector artwork foreground.",
    )

    lineart = subparsers.add_parser(
        "lineart",
        help="Convert pencil or black-line artwork into mixed line and parametric equations.",
    )
    lineart.add_argument("--input", required=True, type=Path, help="Input image path.")
    lineart.add_argument("--out", required=True, type=Path, help="Output directory.")
    lineart.add_argument(
        "--target",
        default=1200,
        type=int,
        help="Approximate equation or curve segment count.",
    )
    lineart.add_argument(
        "--scale-width",
        default=20.0,
        type=float,
        help="Coordinate-plane width for the drawing bounding box.",
    )
    lineart.add_argument(
        "--threshold-mode",
        choices=["auto", "fixed", "adaptive"],
        default="auto",
        help="Line extraction threshold strategy.",
    )
    lineart.add_argument(
        "--line-thickness",
        default=1,
        type=int,
        help="Preview line thickness in pixels.",
    )
    lineart.add_argument(
        "--fit-mode",
        choices=["linear", "quadratic", "mixed"],
        default="mixed",
        help="Curve fitting mode.",
    )
    lineart.add_argument(
        "--cleaned-input",
        type=Path,
        help="Optional cleaned line-art image used instead of the raw input.",
    )
    lineart.add_argument(
        "--trace-mode",
        choices=["skeleton-v1", "centerline-v2"],
        default="skeleton-v1",
        help="Line-art trace pipeline.",
    )
    lineart.add_argument(
        "--preprocess-scale",
        default=4,
        type=int,
        help="Centerline V2 preprocessing scale.",
    )
    lineart.add_argument(
        "--render-scale",
        default=4,
        type=int,
        help="Centerline V2 function preview render scale.",
    )
    lineart.add_argument(
        "--max-bridge-gap",
        default=16,
        type=float,
        help="Centerline V2 maximum endpoint bridge gap in high-res pixels.",
    )
    lineart.add_argument(
        "--bridge-angle-threshold",
        default=45,
        type=float,
        help="Centerline V2 endpoint bridge tangent threshold in degrees.",
    )
    lineart.add_argument(
        "--local-threshold",
        choices=["sauvola", "niblack", "adaptive", "fixed"],
        default="sauvola",
        help="Centerline V2 local threshold mode.",
    )
    lineart.add_argument(
        "--keep-diagnostics",
        action="store_true",
        help="Write Centerline V2 diagnostic overlays and JSON.",
    )
    lineart.add_argument(
        "--min-component-area",
        default=9,
        type=int,
        help="Centerline V2 minimum connected component area in high-res pixels.",
    )
    lineart.add_argument(
        "--min-chain-length",
        default=16.0,
        type=float,
        help="Centerline V2 minimum stroke chain length in high-res pixels.",
    )
    lineart.add_argument(
        "--min-segment-length",
        default=0.1,
        type=float,
        help="Centerline V2 minimum fitted segment Cartesian length.",
    )

    thick_lineart = subparsers.add_parser(
        "thick-lineart",
        help="Convert koala-style colored thick line art into stacked offset equations.",
    )
    thick_lineart.add_argument("--input", required=True, type=Path, help="Input image path.")
    thick_lineart.add_argument("--out", required=True, type=Path, help="Output directory.")
    thick_lineart.add_argument(
        "--target",
        default=1800,
        type=int,
        help="Approximate centerline source segment budget before offset stacking.",
    )
    thick_lineart.add_argument(
        "--scale-width",
        default=20.0,
        type=float,
        help="Coordinate-plane width for the combined color foreground.",
    )
    thick_lineart.add_argument(
        "--offset-step",
        default=1.0,
        type=float,
        help="Distance between stacked offset copies in source-image pixels.",
    )
    thick_lineart.add_argument(
        "--max-offsets",
        default=9,
        type=int,
        help="Odd total stack count per source segment, including offset_index=0.",
    )
    thick_lineart.add_argument(
        "--render-scale",
        default=4,
        type=int,
        help="Supersampling factor for thick-lineart previews.",
    )
    thick_lineart.add_argument(
        "--min-component-area",
        default=12,
        type=int,
        help="Minimum connected component area for required color masks.",
    )
    thick_lineart.add_argument(
        "--min-chain-length",
        default=8.0,
        type=float,
        help="Minimum traced chain length in source-image pixels.",
    )
    thick_lineart.add_argument(
        "--min-radius-pixels",
        default=1.0,
        type=float,
        help="Minimum sampled stroke radius in source-image pixels.",
    )
    thick_lineart.add_argument(
        "--max-radius-pixels",
        default=10.0,
        type=float,
        help="Maximum sampled stroke radius in source-image pixels.",
    )
    thick_lineart.add_argument(
        "--keep-diagnostics",
        action="store_true",
        help="Write thick-lineart diagnostic JSON files.",
    )

    docx_point_cloud = subparsers.add_parser(
        "docx-point-cloud",
        help="Render a DOCX coordinate point cloud as colored horizontal equations.",
    )
    docx_point_cloud.add_argument("--input", required=True, type=Path, help="Source image used for color sampling.")
    docx_point_cloud.add_argument("--reference-docx", required=True, type=Path, help="DOCX containing (x, -y) point coordinates.")
    docx_point_cloud.add_argument("--out", required=True, type=Path, help="Output directory.")
    docx_point_cloud.add_argument(
        "--scale-width",
        default=20.0,
        type=float,
        help="Coordinate-plane width for the point-cloud foreground.",
    )
    docx_point_cloud.add_argument(
        "--render-scale",
        default=4,
        type=int,
        help="Supersampling factor for point-cloud previews.",
    )
    docx_point_cloud.add_argument(
        "--clean-kernel-size",
        default=3,
        type=int,
        help="Morphology kernel used to remove edge debris from point masks.",
    )
    docx_point_cloud.add_argument(
        "--min-component-area",
        default=6,
        type=int,
        help="Smallest connected point-mask component to keep.",
    )

    patch_face = subparsers.add_parser(
        "patch-face-features",
        help="Replace small face features in a line-art output with reference DOCX points.",
    )
    patch_face.add_argument("--segments", required=True, type=Path, help="Input segments.json path.")
    patch_face.add_argument("--reference-docx", required=True, type=Path, help="Reference DOCX point list.")
    patch_face.add_argument("--out", required=True, type=Path, help="Patched output directory.")
    patch_face.add_argument(
        "--source-dir",
        type=Path,
        help="Output directory to copy before patching. Defaults to the segments.json directory.",
    )

    parser.add_argument("--input", type=Path, help="Input image path.")
    parser.add_argument("--target", default=200, type=int, help="Approximate equation count.")
    parser.add_argument("--out", type=Path, help="Output directory.")
    parser.add_argument(
        "--scale-width",
        default=20.0,
        type=float,
        help="Coordinate-plane width for the foreground bounding box.",
    )
    parser.add_argument(
        "--saturation-threshold",
        default=40,
        type=int,
        help="HSV saturation threshold used to isolate colored foreground.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the requested pipeline."""
    args = build_parser().parse_args()
    if args.command == "lucas":
        result = run_lucas_pipeline(
            image_path=args.input,
            out_dir=args.out,
            outline_target=args.outline_target,
            region_target=args.region_target,
            gray_levels=args.gray_levels,
            scale_width=args.scale_width,
        )
        print(f"Wrote {result.total_equations} equations to {result.out_dir}")
        print(f"Outline equations: {result.outline_equations}")
        print(f"Fill equations: {result.fill_equations}")
        return
    if args.command == "lucas-vector":
        result = run_lucas_vector_pipeline(
            image_path=args.input,
            out_dir=args.out,
            scale_width=args.scale_width,
        )
        print(f"Wrote {result.shape_count} vector shapes to {result.out_dir}")
        print(f"Desmos vector segments: {result.segment_count}")
        return
    if args.command == "lineart":
        result = run_lineart_pipeline(
            image_path=args.input,
            out_dir=args.out,
            target=args.target,
            scale_width=args.scale_width,
            threshold_mode=args.threshold_mode,
            line_thickness=args.line_thickness,
            fit_mode=args.fit_mode,
            cleaned_input=args.cleaned_input,
            trace_mode=args.trace_mode,
            preprocess_scale=args.preprocess_scale,
            render_scale=args.render_scale,
            max_bridge_gap=args.max_bridge_gap,
            bridge_angle_threshold=args.bridge_angle_threshold,
            local_threshold=args.local_threshold,
            keep_diagnostics=args.keep_diagnostics,
            min_component_area=args.min_component_area,
            min_chain_length=args.min_chain_length,
            min_segment_length=args.min_segment_length,
        )
        print(f"Wrote {result.equation_count} line-art equations to {result.out_dir}")
        print(f"Trace mode: {result.trace_mode}")
        print(f"Traced strokes: {result.stroke_count}")
        if result.trace_mode == "centerline-v2":
            print(f"Raw branches: {result.raw_branch_count}")
            print(f"Accepted bridges: {result.accepted_bridge_count}")
            print(f"Final chains: {result.final_chain_count}")
        print(f"Preview: {result.function_preview_path}")
        print(f"JSON: {result.json_path}")
        return
    if args.command == "thick-lineart":
        result = run_thick_lineart_pipeline(
            image_path=args.input,
            out_dir=args.out,
            target=args.target,
            scale_width=args.scale_width,
            offset_step_pixels=args.offset_step,
            max_offsets=args.max_offsets,
            render_scale=args.render_scale,
            min_component_area=args.min_component_area,
            min_chain_length=args.min_chain_length,
            min_radius_pixels=args.min_radius_pixels,
            max_radius_pixels=args.max_radius_pixels,
            keep_diagnostics=args.keep_diagnostics,
        )
        print(f"Wrote {result.equation_count} thick line-art equations to {result.out_dir}")
        print(f"Source segments: {result.source_segment_count}")
        print(f"Color counts: {result.color_counts}")
        print(f"Preview: {result.function_preview_path}")
        print(f"JSON: {result.json_path}")
        return
    if args.command == "docx-point-cloud":
        result = run_point_cloud_pipeline(
            image_path=args.input,
            reference_docx=args.reference_docx,
            out_dir=args.out,
            scale_width=args.scale_width,
            render_scale=args.render_scale,
            clean_kernel_size=args.clean_kernel_size,
            min_component_area=args.min_component_area,
        )
        print(f"Wrote {result.equation_count} point-cloud equations to {result.out_dir}")
        print(f"Point count: {result.point_count}")
        print(f"Color counts: {result.color_counts}")
        print(f"Preview: {result.function_preview_path}")
        print(f"Direct point render: {result.direct_preview_path}")
        print(f"JSON: {result.json_path}")
        return
    if args.command == "patch-face-features":
        preview = patch_face_feature_output(
            segments_path=args.segments,
            reference_docx=args.reference_docx,
            out_dir=args.out,
            source_dir=args.source_dir,
        )
        print(f"Patched preview: {preview}")
        print(f"Patched JSON: {args.out / 'segments.json'}")
        return

    if args.input is None or args.out is None:
        raise SystemExit("--input and --out are required unless using a subcommand")

    result = run_pipeline(
        image_path=args.input,
        out_dir=args.out,
        target=args.target,
        scale_width=args.scale_width,
        saturation_threshold=args.saturation_threshold,
    )
    print(f"Wrote {result.segment_count} equations to {result.out_dir}")
    print(f"Preview: {result.preview_path}")
    print(f"Equations: {result.equations_path}")
    print(f"JSON: {result.json_path}")


if __name__ == "__main__":
    main()
