import json
import tempfile
import unittest
from pathlib import Path

from mathequations.desmos_export import expression_bounds, write_desmos_html


class DesmosExportTests(unittest.TestCase):
    def test_expression_bounds_uses_attached_segment_points(self):
        bounds = expression_bounds(
            [
                {
                    "latex": "y=1",
                    "segment": {
                        "source_points": [
                            {"x": -2.0, "y": 1.0},
                            {"x": 3.0, "y": 5.0},
                        ]
                    },
                }
            ]
        )

        self.assertLess(bounds["left"], -2.0)
        self.assertGreater(bounds["right"], 3.0)
        self.assertLess(bounds["bottom"], 1.0)
        self.assertGreater(bounds["top"], 5.0)

    def test_write_desmos_html_embeds_expressions_and_calculator_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expressions_path = root / "desmos_expressions.json"
            segments_path = root / "segments.json"
            output_path = root / "desmos.html"
            expressions_path.write_text(
                json.dumps([{"id": "lineart_1", "latex": "y=1", "color": "#000000", "lineWidth": "1"}]),
                encoding="utf-8",
            )
            segments_path.write_text(
                json.dumps({"segments": [{"source_points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]}]}),
                encoding="utf-8",
            )

            write_desmos_html(expressions_path, output_path, segments_path=segments_path)
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("www.desmos.com/api", html)
        self.assertIn("calculator.setExpressions(expressions)", html)
        self.assertIn('"latex": "y=1"', html)


if __name__ == "__main__":
    unittest.main()
