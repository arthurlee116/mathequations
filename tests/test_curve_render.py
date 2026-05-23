import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mathequations.curve_equations import bezier_cubic_segment, linear_segment, quadratic_segment
from mathequations.curve_render import render_curve_segments, sample_segment_points


class CurveRenderTests(unittest.TestCase):
    def test_sample_segment_points_supports_mixed_types(self):
        segments = [
            linear_segment(1, 1, (-2.0, 0.0), (2.0, 2.0)),
            quadratic_segment(
                2,
                1,
                coefficients=(1.0, 0.0, 0.0),
                x_range=(-1.0, 1.0),
                endpoints=((-1.0, 1.0), (1.0, 1.0)),
                fit_error=0.0,
            ),
            bezier_cubic_segment(
                3,
                2,
                control_points=((0.0, 0.0), (1.0, 2.0), (2.0, 2.0), (3.0, 0.0)),
                fit_error=0.0,
            ),
        ]

        counts = [len(sample_segment_points(segment, samples=24)) for segment in segments]

        self.assertEqual(counts, [24, 24, 24])

    def test_render_curve_segments_writes_nonblank_preview(self):
        segments = [
            linear_segment(1, 1, (-5.0, 0.0), (5.0, 0.0)),
            bezier_cubic_segment(
                2,
                1,
                control_points=((-4.0, -2.0), (-2.0, 4.0), (2.0, 4.0), (4.0, -2.0)),
                fit_error=0.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preview.png"
            render_curve_segments(segments, path, image_size=(100, 100), scale=5.0)
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        self.assertIsNotNone(image)
        self.assertLess(int(np.min(image)), 250)


if __name__ == "__main__":
    unittest.main()
