import unittest

import cv2
import numpy as np

from mathequations.centerline_graph import StrokeChain
from mathequations.stroke_width import (
    stroke_distance_map,
    stroke_radius_by_id,
    summarize_radius,
)


class StrokeWidthTests(unittest.TestCase):
    def test_distance_map_reports_larger_radius_for_thicker_line(self):
        thin = np.zeros((40, 80), dtype=np.uint8)
        thick = np.zeros((40, 80), dtype=np.uint8)
        cv2.line(thin, (10, 20), (70, 20), 255, thickness=3)
        cv2.line(thick, (10, 20), (70, 20), 255, thickness=9)

        thin_radius = summarize_radius([(40.0, 20.0)], stroke_distance_map(thin))
        thick_radius = summarize_radius([(40.0, 20.0)], stroke_distance_map(thick))

        self.assertGreater(thick_radius, thin_radius)
        self.assertGreaterEqual(thin_radius, 1.0)

    def test_stroke_radius_by_id_uses_chain_points_and_clamps(self):
        mask = np.zeros((30, 50), dtype=np.uint8)
        cv2.line(mask, (5, 15), (45, 15), 255, thickness=7)
        chains = [
            StrokeChain(stroke_id=8, points=[(10.0, 15.0), (20.0, 15.0), (30.0, 15.0)])
        ]

        radii = stroke_radius_by_id(
            chains,
            stroke_distance_map(mask),
            min_radius_pixels=1.0,
            max_radius_pixels=3.0,
        )

        self.assertEqual(set(radii), {8})
        self.assertGreaterEqual(radii[8], 1.0)
        self.assertLessEqual(radii[8], 3.0)


if __name__ == "__main__":
    unittest.main()
