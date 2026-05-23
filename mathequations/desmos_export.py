"""Export Desmos API HTML pages from saved expression JSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://www.desmos.com/api/v1.12/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
  <style>
    html, body {{
      height: 100%;
      margin: 0;
    }}
    body {{
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
      background: white;
    }}
    #calculator {{
      width: min(100vw, calc(100vh * {aspect_ratio}));
      height: min(100vh, calc(100vw / {aspect_ratio}));
    }}
    #status {{
      position: fixed;
      right: 10px;
      top: 10px;
      z-index: 10;
      padding: 4px 7px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.85);
      color: #333;
      font: 12px/1.3 system-ui, sans-serif;
    }}
  </style>
</head>
<body>
  <div id="status">Loading Desmos...</div>
  <div id="calculator"></div>
  <script>
    const expressions = {expressions_json};
    const status = document.getElementById("status");
    const calculatorElt = document.getElementById("calculator");
    const calculator = Desmos.GraphingCalculator(
      calculatorElt,
      {{
        expressions: true,
        settingsMenu: true,
        zoomButtons: true,
        expressionsCollapsed: true,
      }}
    );
    const rawBounds = {bounds_json};
    calculator.setExpressions(expressions);
    function applyFinalBounds() {{
      calculator.setMathBounds(rawBounds);
    }}
    applyFinalBounds();
    setTimeout(applyFinalBounds, 2000);
    setTimeout(applyFinalBounds, 8000);
    window.calculator = calculator;
    status.textContent = `Loaded ${{expressions.length}} expressions`;
    status.dataset.expressionCount = String(expressions.length);
  </script>
</body>
</html>
"""


def expression_bounds(expressions: list[dict[str, Any]]) -> dict[str, float]:
    """Return graph bounds that frame all expression source points when available."""
    xs: list[float] = []
    ys: list[float] = []
    for expression in expressions:
        segment = expression.get("segment")
        if not isinstance(segment, dict):
            continue
        for key in ("source_points", "points", "control_points"):
            points = segment.get(key, [])
            if isinstance(points, list):
                for point in points:
                    if isinstance(point, dict) and "x" in point and "y" in point:
                        xs.append(float(point["x"]))
                        ys.append(float(point["y"]))
    if not xs or not ys:
        return {"left": -12.0, "right": 12.0, "bottom": -18.0, "top": 18.0}

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = max(0.5, (x_max - x_min) * 0.08)
    pad_y = max(0.5, (y_max - y_min) * 0.08)
    return {"left": x_min - pad_x, "right": x_max + pad_x, "bottom": y_min - pad_y, "top": y_max + pad_y}


def canvas_bounds(metadata: dict[str, Any], *, padding_ratio: float = 0.04) -> dict[str, float]:
    """Return bounds for the full source image canvas in Cartesian coordinates."""
    width = float(metadata["image_width"])
    height = float(metadata["image_height"])
    scale = float(metadata["scale"])
    half_width = width * scale / 2
    half_height = height * scale / 2
    pad_x = half_width * padding_ratio
    pad_y = half_height * padding_ratio
    return {
        "left": -half_width - pad_x,
        "right": half_width + pad_x,
        "bottom": -half_height - pad_y,
        "top": half_height + pad_y,
    }


def load_desmos_expressions(expressions_path: Path, segments_path: Path | None = None) -> list[dict[str, Any]]:
    """Load Desmos expressions and optionally attach segment metadata for bounds."""
    expressions = json.loads(expressions_path.read_text(encoding="utf-8"))
    if segments_path is None:
        return expressions

    payload = json.loads(segments_path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    segment_by_id = {f"lineart_{index}": segment for index, segment in enumerate(segments, start=1)}
    result = []
    for expression in expressions:
        record = dict(expression)
        segment = segment_by_id.get(str(record.get("id")))
        if segment is not None:
            record["segment"] = segment
        result.append(record)
    return result


def write_desmos_html(
    expressions_path: Path,
    output_path: Path,
    *,
    segments_path: Path | None = None,
    title: str = "NAN line art - fixed",
) -> Path:
    """Write an HTML page that loads the official Desmos API and expressions."""
    expressions_with_segments = load_desmos_expressions(expressions_path, segments_path)
    aspect_ratio = "0.7075"
    if segments_path is not None:
        payload = json.loads(segments_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        if {"image_width", "image_height", "scale"} <= set(metadata):
            bounds = canvas_bounds(metadata)
            aspect_ratio = str(float(metadata["image_width"]) / float(metadata["image_height"]))
        else:
            bounds = expression_bounds(expressions_with_segments)
    else:
        bounds = expression_bounds(expressions_with_segments)
        aspect_ratio = str((bounds["right"] - bounds["left"]) / (bounds["top"] - bounds["bottom"]))
    expressions = [{key: value for key, value in expression.items() if key != "segment"} for expression in expressions_with_segments]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        HTML_TEMPLATE.format(
            title=html.escape(title),
            aspect_ratio=aspect_ratio,
            expressions_json=json.dumps(expressions, ensure_ascii=False),
            bounds_json=json.dumps(bounds),
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Parse CLI arguments and export a Desmos API HTML file."""
    parser = argparse.ArgumentParser(description="Export saved Desmos expressions to an interactive HTML page.")
    parser.add_argument("--expressions", required=True, type=Path, help="Input desmos_expressions.json path.")
    parser.add_argument("--out", required=True, type=Path, help="Output HTML path.")
    parser.add_argument("--segments", type=Path, help="Optional segments.json path used to compute graph bounds.")
    parser.add_argument("--title", default="NAN line art - fixed", help="HTML page title.")
    args = parser.parse_args()

    path = write_desmos_html(args.expressions, args.out, segments_path=args.segments, title=args.title)
    print(path)


if __name__ == "__main__":
    main()
