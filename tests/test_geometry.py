import unittest

import numpy as np

from mathequations.geometry import (
    contour_to_points,
    pixel_to_cartesian,
    resample_closed_points,
)


class GeometryTests(unittest.TestCase):
    def test_pixel_to_cartesian_centers_and_flips_y_axis(self):
        point = pixel_to_cartesian((75, 25), width=100, height=100, scale=0.2)

        self.assertEqual(point, (5.0, 5.0))

    def test_resample_closed_points_returns_approximately_requested_count(self):
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        sampled = resample_closed_points(square, target_count=18)

        self.assertGreaterEqual(len(sampled), 16)
        self.assertLessEqual(len(sampled), 20)

    def test_contour_to_points_flattens_opencv_shape(self):
        contour = np.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.int32)

        points = contour_to_points(contour)

        self.assertEqual(points, [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])


if __name__ == "__main__":
    unittest.main()
