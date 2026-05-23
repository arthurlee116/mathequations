"""Command-line entry point for the image-to-equation pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from .lineart_pipeline import run_lineart_pipeline
from .lucas_pipeline import run_lucas_pipeline
from .lucas_vector_pipeline import run_lucas_vector_pipeline
from .pipeline import run_pipeline


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
