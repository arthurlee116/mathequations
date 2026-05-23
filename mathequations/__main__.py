"""Command-line entry point for the image-to-equation pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

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
