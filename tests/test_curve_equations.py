import json
import unittest

from mathequations.curve_equations import (
    bezier_cubic_segment,
    linear_segment,
    parametric_polyline_segment,
    quadratic_segment,
    segments_payload,
    vertical_segment,
)


class CurveEquationTests(unittest.TestCase):
    def test_linear_and_vertical_segments_format_latex(self):
        linear = linear_segment(1, 2, (0.0, 1.0), (2.0, 5.0))
        vertical = vertical_segment(2, 2, (3.0, -1.0), (3.0, 4.0))

        self.assertEqual(linear["type"], "linear")
        self.assertIn("y=2x+1", linear["latex"])
        self.assertEqual(vertical["type"], "vertical")
        self.assertIn("x=3", vertical["latex"])

    def test_quadratic_segment_formats_restricted_equation(self):
        segment = quadratic_segment(
            3,
            4,
            coefficients=(1.0, -2.0, 1.0),
            x_range=(-1.0, 3.0),
            endpoints=((-1.0, 4.0), (3.0, 4.0)),
            fit_error=0.01,
        )

        self.assertEqual(segment["type"], "quadratic")
        self.assertIn("x^2", segment["latex"])
        self.assertEqual(segment["restriction"]["variable"], "x")

    def test_bezier_cubic_segment_formats_parametric_latex(self):
        segment = bezier_cubic_segment(
            4,
            5,
            control_points=((0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0)),
            fit_error=0.2,
        )

        self.assertEqual(segment["type"], "bezier_cubic")
        self.assertIn("0\\le t\\le 1", segment["latex"])
        self.assertIn("x(t)", segment["equation"])
        self.assertIn("y(t)", segment["equation"])

    def test_parametric_polyline_segment_formats_required_payload(self):
        points = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0)]

        segment = parametric_polyline_segment(7, 9, points, fit_error=0.4)

        self.assertEqual(segment["segment_id"], 7)
        self.assertEqual(segment["stroke_id"], 9)
        self.assertEqual(segment["type"], "parametric_polyline")
        self.assertIn("0\\le t\\le 2", segment["latex"])
        self.assertEqual(segment["restriction"]["variable"], "t")
        self.assertEqual(len(segment["source_points"]), 3)

    def test_segments_payload_is_json_serializable(self):
        segment = linear_segment(1, 1, (0.0, 0.0), (1.0, 1.0))
        payload = segments_payload(
            [segment],
            strokes=[{"stroke_id": 1, "point_count": 2, "length": 1.4}],
            image_size=(100, 120),
            scale=0.2,
            target=1200,
            fit_mode="mixed",
        )

        decoded = json.loads(json.dumps(payload))

        self.assertEqual(decoded["metadata"]["equation_count"], 1)
        self.assertEqual(decoded["metadata"]["stroke_count"], 1)
        self.assertEqual(decoded["segments"][0]["type"], "linear")


if __name__ == "__main__":
    unittest.main()
