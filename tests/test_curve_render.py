import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mathequations.curve_equations import (
    bezier_cubic_segment,
    linear_segment,
    parametric_polyline_segment,
    quadratic_segment,
)
from mathequations.curve_render import (
    render_curve_segment_previews,
    render_curve_segments,
    sample_segment_points,
)


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
            parametric_polyline_segment(4, 3, [(-2.0, -2.0), (0.0, 2.0), (2.0, -1.0)]),
        ]

        counts = [len(sample_segment_points(segment, samples=24)) for segment in segments]

        self.assertEqual(counts, [24, 24, 24, 24])

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

    def test_render_curve_segments_uses_segment_color_metadata(self):
        segment = linear_segment(1, 1, (-5.0, 0.0), (5.0, 0.0))
        segment["color"] = "#f5d529"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yellow.png"
            render_curve_segments(
                [segment],
                path,
                image_size=(80, 40),
                scale=4.0,
                line_thickness=3,
            )
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        self.assertIsNotNone(image)
        nonwhite = image[np.any(image < 250, axis=2)]
        self.assertGreater(len(nonwhite), 0)
        mean_bgr = nonwhite.mean(axis=0)
        self.assertGreater(mean_bgr[1], mean_bgr[0])
        self.assertGreater(mean_bgr[2], mean_bgr[0])

    def test_render_curve_segment_previews_writes_highres_and_downsampled(self):
        segments = [linear_segment(1, 1, (-5.0, 0.0), (5.0, 0.0))]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            paths = render_curve_segment_previews(
                segments,
                out_dir,
                image_size=(60, 40),
                scale=5.0,
                render_scale=4,
            )
            preview = cv2.imread(str(paths["preview"]), cv2.IMREAD_GRAYSCALE)
            highres = cv2.imread(str(paths["highres"]), cv2.IMREAD_GRAYSCALE)

        self.assertEqual(preview.shape, (40, 60))
        self.assertEqual(highres.shape, (160, 240))
        self.assertLess(int(np.min(preview)), 250)


if __name__ == "__main__":
    unittest.main()
