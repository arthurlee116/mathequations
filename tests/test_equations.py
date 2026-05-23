import json
import unittest

from mathequations.equations import equation_from_segment, segments_to_jsonable


class EquationTests(unittest.TestCase):
    def test_sloped_segment_becomes_restricted_linear_equation(self):
        segment = equation_from_segment(
            segment_id=1,
            contour_id=0,
            start=(-2.0, 1.0),
            end=(2.0, 5.0),
        )

        self.assertEqual(segment["type"], "linear")
        self.assertAlmostEqual(segment["m"], 1.0)
        self.assertAlmostEqual(segment["b"], 3.0)
        self.assertEqual(segment["restriction"], {"variable": "x", "min": -2.0, "max": 2.0})
        self.assertEqual(segment["equation"], "y = 1x + 3 {-2 <= x <= 2}")

    def test_vertical_segment_becomes_restricted_vertical_equation(self):
        segment = equation_from_segment(
            segment_id=2,
            contour_id=0,
            start=(4.0, -3.0),
            end=(4.0, 7.0),
        )

        self.assertEqual(segment["type"], "vertical")
        self.assertEqual(segment["c"], 4.0)
        self.assertEqual(segment["restriction"], {"variable": "y", "min": -3.0, "max": 7.0})
        self.assertEqual(segment["equation"], "x = 4 {-3 <= y <= 7}")

    def test_segments_json_contains_required_fields(self):
        segment = equation_from_segment(1, 3, (0.0, 0.0), (1.0, 1.0))
        payload = segments_to_jsonable([segment], image_size=(100, 80), scale=0.2, target=200)
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["metadata"]["image_width"], 100)
        self.assertEqual(decoded["metadata"]["image_height"], 80)
        self.assertEqual(decoded["metadata"]["scale"], 0.2)
        self.assertEqual(decoded["metadata"]["target"], 200)
        self.assertEqual(decoded["segments"][0]["segment_id"], 1)
        self.assertEqual(decoded["segments"][0]["contour_id"], 3)
        self.assertIn("start", decoded["segments"][0])
        self.assertIn("end", decoded["segments"][0])
        self.assertIn("equation", decoded["segments"][0])


if __name__ == "__main__":
    unittest.main()
