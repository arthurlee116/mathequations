"""SVG and raster renderers for ``VectorArtwork`` objects."""

from __future__ import annotations

from html import escape
from pathlib import Path

import cv2
import numpy as np

from .vector_model import VectorArtwork, VectorShape


def _paint_value(color: str | None, fallback: str = "none") -> str:
    """Return a valid SVG paint value for optional colors."""
    return fallback if color is None else color


def svg_string(artwork: VectorArtwork) -> str:
    """Serialize vector artwork to a standalone SVG document."""
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{artwork.width}" height="{artwork.height}" '
        f'viewBox="0 0 {artwork.width} {artwork.height}">'
    ]
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    for shape in artwork.ordered_shapes():
        path_data = shape.path_data()
        if not path_data:
            continue
        lines.append(
            f'<path id="{escape(shape.shape_id)}" d="{path_data}" '
            f'fill="{_paint_value(shape.fill)}" '
            f'stroke="{_paint_value(shape.stroke)}" '
            f'stroke-width="{shape.stroke_width:g}" '
            'stroke-linejoin="round" stroke-linecap="round" fill-rule="evenodd"/>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_svg(artwork: VectorArtwork, path: Path) -> None:
    """Write vector artwork as UTF-8 SVG."""
    Path(path).write_text(svg_string(artwork), encoding="utf-8")


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    """Convert ``#RRGGBB`` colors to OpenCV's BGR channel order."""
    if len(color) != 7 or not color.startswith("#"):
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    return (blue, green, red)


def _contour_array(contour: list[tuple[float, float]]) -> np.ndarray:
    """Convert contour points into the integer shape OpenCV expects."""
    points = np.array(contour, dtype=np.float32)
    return np.round(points).astype(np.int32).reshape((-1, 1, 2))


def _draw_shape(canvas: np.ndarray, shape: VectorShape) -> None:
    """Draw one filled and/or stroked vector shape onto a raster canvas."""
    contours = [_contour_array(contour) for contour in shape.contours if contour]
    if not contours:
        return

    if shape.fill is not None and shape.closed:
        cv2.drawContours(canvas, contours, -1, _hex_to_bgr(shape.fill), thickness=-1)
    if shape.stroke is not None and shape.stroke_width > 0:
        thickness = max(1, round(shape.stroke_width))
        for contour in contours:
            cv2.polylines(
                canvas,
                [contour],
                isClosed=shape.closed,
                color=_hex_to_bgr(shape.stroke),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )


def render_vector_artwork(artwork: VectorArtwork, path: Path) -> None:
    """Render vector artwork to a PNG-like raster path supported by OpenCV."""
    canvas = np.full((artwork.height, artwork.width, 3), 255, dtype=np.uint8)
    for shape in artwork.ordered_shapes():
        _draw_shape(canvas, shape)
    cv2.imwrite(str(path), canvas)
