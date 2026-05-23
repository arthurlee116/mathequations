from __future__ import annotations

from dataclasses import dataclass


Point = tuple[float, float]


def _format_svg_number(value: float) -> str:
    rounded = round(float(value), 3)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:g}"


def svg_path_from_points(points: list[Point], *, closed: bool = True) -> str:
    if not points:
        return ""

    start = points[0]
    commands = [f"M {_format_svg_number(start[0])} {_format_svg_number(start[1])}"]
    for x, y in points[1:]:
        commands.append(f"L {_format_svg_number(x)} {_format_svg_number(y)}")
    if closed:
        commands.append("Z")
    return " ".join(commands)


@dataclass(frozen=True)
class VectorShape:
    shape_id: str
    contours: list[list[Point]]
    fill: str | None
    stroke: str | None
    stroke_width: float
    z_index: int = 0
    closed: bool = True
    label: str | None = None

    def path_data(self) -> str:
        return " ".join(
            svg_path_from_points(contour, closed=self.closed)
            for contour in self.contours
            if contour
        )


@dataclass(frozen=True)
class VectorArtwork:
    width: int
    height: int
    shapes: list[VectorShape]

    def ordered_shapes(self) -> list[VectorShape]:
        return sorted(self.shapes, key=lambda shape: shape.z_index)
