import math
import unittest
import warnings

from mathequations.curve_fit import fit_points_to_segment, fit_stroke_paths
from mathequations.skeleton_graph import StrokePath


class CurveFitTests(unittest.TestCase):
    def test_fit_points_to_segment_uses_linear_for_straight_line(self):
        points = [(float(x), 2.0 * x + 1.0) for x in range(8)]

        segment = fit_points_to_segment(1, 1, points, fit_mode="mixed")

        self.assertEqual(segment["type"], "linear")
        self.assertLess(segment["fit_error"], 1e-6)

    def test_fit_points_to_segment_uses_quadratic_for_parabola(self):
        points = [(float(x), float((x - 3) ** 2)) for x in range(7)]

        segment = fit_points_to_segment(1, 1, points, fit_mode="mixed")

        self.assertEqual(segment["type"], "quadratic")
        self.assertLess(segment["fit_error"], 1e-6)

    def test_fit_points_to_segment_uses_bezier_for_non_monotonic_arc(self):
        points = []
        for index in range(18):
            theta = math.pi * index / 17
            points.append((math.cos(theta), math.sin(theta)))

        segment = fit_points_to_segment(1, 1, points, fit_mode="mixed")

        self.assertEqual(segment["type"], "bezier_cubic")
        self.assertLess(segment["fit_error"], 0.35)

    def test_fit_points_to_segment_handles_vertical_endpoints_with_minor_wobble(self):
        points = [
            (-2.583799, 5.391061),
            (-2.611732, 5.363128),
            (-2.583799, 5.335196),
            (-2.583799, 5.307263),
            (-2.583799, 5.27933),
        ]

        segment = fit_points_to_segment(1, 1, points, fit_mode="mixed")

        self.assertEqual(segment["type"], "vertical")
        self.assertEqual(segment["c"], -2.583799)

    def test_fit_points_to_segment_skips_quadratic_when_x_support_is_too_small(self):
        points = [(0.0, 0.0), (0.0, 1.0), (1.0, 2.0), (1.0, 3.0)]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            segment = fit_points_to_segment(1, 1, points, fit_mode="mixed")

        self.assertEqual(segment["type"], "bezier_cubic")

    def test_fit_stroke_paths_respects_target_budget(self):
        stroke = StrokePath(
            stroke_id=1,
            points=[(float(x), math.sin(x / 8.0)) for x in range(120)],
            closed=False,
        )

        segments = fit_stroke_paths([stroke], target=12, fit_mode="mixed")

        self.assertGreater(len(segments), 1)
        self.assertLessEqual(len(segments), 14)
        self.assertEqual(segments[0]["stroke_id"], 1)


if __name__ == "__main__":
    unittest.main()
