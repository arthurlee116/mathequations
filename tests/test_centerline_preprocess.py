import unittest

import cv2
import numpy as np

from mathequations.lineart_preprocess import (
    build_local_line_mask,
    line_mask_diagnostics,
    prepare_highres_lineart,
)


class CenterlinePreprocessTests(unittest.TestCase):
    def test_prepare_highres_lineart_scales_image(self):
        image = np.full((12, 18, 3), 255, dtype=np.uint8)

        highres = prepare_highres_lineart(image, scale=4)

        self.assertEqual(highres.shape[:2], (48, 72))

    def test_local_mask_keeps_faint_one_pixel_line(self):
        image = np.full((40, 60, 3), 255, dtype=np.uint8)
        cv2.line(image, (8, 20), (52, 20), (220, 220, 220), thickness=1)
        highres = prepare_highres_lineart(image, scale=4)

        mask, _ = build_local_line_mask(highres, threshold_mode="sauvola", min_component_area=2)

        self.assertGreater(cv2.countNonZero(mask), 120)
        self.assertGreater(int(mask[80, 120]), 0)

    def test_component_filter_removes_speckles_without_erasing_stroke(self):
        image = np.full((50, 50, 3), 255, dtype=np.uint8)
        cv2.line(image, (8, 25), (42, 25), (30, 30, 30), thickness=1)
        image[4, 4] = [0, 0, 0]
        image[45, 45] = [0, 0, 0]

        mask, removed_count = build_local_line_mask(image, threshold_mode="fixed", min_component_area=3)

        self.assertEqual(removed_count, 2)
        self.assertEqual(int(mask[4, 4]), 0)
        self.assertEqual(int(mask[45, 45]), 0)
        self.assertGreater(int(mask[25, 25]), 0)
        self.assertGreater(cv2.countNonZero(mask), 25)

    def test_line_mask_diagnostics_reports_density_and_components(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5, 5:10] = 255
        mask[12:15, 12:15] = 255

        diagnostics = line_mask_diagnostics(mask, removed_speckle_count=5)

        self.assertAlmostEqual(diagnostics["foreground_density"], 14 / 400)
        self.assertEqual(diagnostics["component_count"], 2)
        self.assertEqual(diagnostics["median_component_size"], 7.0)
        self.assertEqual(diagnostics["removed_speckle_count"], 5)


if __name__ == "__main__":
    unittest.main()
