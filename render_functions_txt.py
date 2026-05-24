"""Render function/domain TXT exports into a colored PNG preview.

Accepted line formats:

    y=2x+1    {-1 <= x <= 1}
    y=2x+1    {-1 <= x <= 1}    #f5d529
    x=3       {-2 <= y <= 2}    color=#000000
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


LINE_RE = re.compile(
    r"^\s*(?P<function>[xy]\s*=\s*.+?)\s+"
    r"\{\s*(?P<lower>-?\d+(?:\.\d+)?)\s*<=\s*(?P<variable>[xy])\s*<=\s*"
    r"(?P<upper>-?\d+(?:\.\d+)?)\s*\}"
    r"(?:\s+(?:color=)?(?P<color>#[0-9a-fA-F]{6}))?\s*$"
)
LINEAR_Y_RE = re.compile(
    r"^(?:(?P<m>[+-]?(?:\d+(?:\.\d+)?)?)x(?P<b>[+-]\d+(?:\.\d+)?)?|(?P<c>[+-]?\d+(?:\.\d+)?))$"
)


@dataclass(frozen=True)
class FunctionRecord:
    expression: str
    variable: str
    lower: float
    upper: float
    color: str


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    if len(color) != 7 or not color.startswith("#"):
        return (0, 0, 0)
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    return (blue, green, red)


def _parse_records(path: Path) -> list[FunctionRecord]:
    records: list[FunctionRecord] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE_RE.match(line)
        if match is None:
            raise ValueError(f"line {line_number} is not a supported function/domain line: {raw_line}")
        records.append(
            FunctionRecord(
                expression=match.group("function").replace(" ", ""),
                variable=match.group("variable"),
                lower=float(match.group("lower")),
                upper=float(match.group("upper")),
                color=match.group("color") or "#000000",
            )
        )
    if not records:
        raise ValueError(f"no functions found in {path}")
    return records


def _evaluate(record: FunctionRecord, samples: int) -> list[tuple[float, float]]:
    expression = record.expression
    left, right = expression.split("=", 1)
    values = np.linspace(record.lower, record.upper, max(2, samples))
    if left == "y":
        if record.variable != "x":
            raise ValueError(f"y= functions must use an x domain: {expression}")
        linear = LINEAR_Y_RE.match(right)
        if linear is None:
            raise ValueError(f"unsupported y function, expected constant or mx+b: {expression}")
        if linear.group("c") is not None:
            slope = 0.0
            intercept = float(linear.group("c"))
        else:
            slope_raw = linear.group("m")
            if slope_raw in (None, "", "+"):
                slope = 1.0
            elif slope_raw == "-":
                slope = -1.0
            else:
                slope = float(slope_raw)
            intercept = float(linear.group("b") or 0.0)
        points = []
        for value in values:
            x = float(value)
            y = slope * x + intercept
            points.append((x, y))
        return points
    if left == "x":
        if record.variable != "y":
            raise ValueError(f"x= functions must use a y domain: {expression}")
        x_value = float(right)
        return [(x_value, float(y)) for y in values]
    raise ValueError(f"unsupported function: {expression}")


def _bounds(records: list[FunctionRecord], *, samples: int) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for record in records:
        for x, y in _evaluate(record, samples=samples):
            xs.append(x)
            ys.append(y)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = max(0.5, (x_max - x_min) * 0.06)
    pad_y = max(0.5, (y_max - y_min) * 0.06)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def render_functions_txt(
    input_path: Path,
    output_path: Path,
    *,
    width: int = 1290,
    height: int = 1269,
    samples: int = 8,
    line_thickness: int = 1,
) -> None:
    records = _parse_records(input_path)
    x_min, x_max, y_min, y_max = _bounds(records, samples=samples)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    def to_pixel(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        px = (x - x_min) / (x_max - x_min) * (width - 1)
        py = (y_max - y) / (y_max - y_min) * (height - 1)
        return int(round(px)), int(round(py))

    for record in records:
        points = np.array([to_pixel(point) for point in _evaluate(record, samples=samples)], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(
                canvas,
                [points.reshape((-1, 1, 2))],
                isClosed=False,
                color=_hex_to_bgr(record.color),
                thickness=max(1, line_thickness),
                lineType=cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render function/domain TXT into a PNG image.")
    parser.add_argument("input", type=Path, help="TXT file with lines like: y=1 {0 <= x <= 1}")
    parser.add_argument("output", type=Path, help="Output PNG path.")
    parser.add_argument("--width", type=int, default=1290, help="Output image width.")
    parser.add_argument("--height", type=int, default=1269, help="Output image height.")
    parser.add_argument("--samples", type=int, default=8, help="Samples per function segment.")
    parser.add_argument("--line-thickness", type=int, default=1, help="Rendered line thickness.")
    args = parser.parse_args()
    render_functions_txt(
        args.input,
        args.output,
        width=args.width,
        height=args.height,
        samples=args.samples,
        line_thickness=args.line_thickness,
    )


if __name__ == "__main__":
    main()
