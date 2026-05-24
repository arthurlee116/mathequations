import unittest

import cv2
import numpy as np

from mathequations.lineart_preprocess import (
    build_line_mask,
    clean_lineart_image,
    skeletonize_line_mask,
)


class LineartPreprocessTests(unittest.TestCase):
    def test_clean_lineart_image_snaps_near_white_background(self):
        image = np.full((20, 20, 3), 249, dtype=np.uint8)
        image[10, 3:17] = [160, 160, 160]

        clean = clean_lineart_image(image)

        self.assertEqual(clean[0, 0].tolist(), [255, 255, 255])
        self.assertLess(clean[10, 10, 0], 220)

    def test_build_line_mask_keeps_faint_gray_strokes(self):
        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        cv2.line(image, (10, 40), (70, 40), (205, 205, 205), thickness=2)

        mask = build_line_mask(image, threshold_mode="auto")

        self.assertGreater(int(mask[40, 40]), 0)
        self.assertEqual(int(mask[0, 0]), 0)

    def test_build_line_mask_keeps_single_pixel_faint_strokes(self):
        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        cv2.line(image, (10, 40), (70, 40), (205, 205, 205), thickness=1)

        mask = build_line_mask(image, threshold_mode="auto")

        self.assertGreater(int(mask[40, 40]), 0)
        self.assertGreater(cv2.countNonZero(mask), 40)

    def test_skeletonize_line_mask_reduces_thick_line_to_centerline(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.line(mask, (10, 40), (70, 40), 255, thickness=7)

        skeleton = skeletonize_line_mask(mask)

        self.assertGreater(cv2.countNonZero(skeleton), 30)
        self.assertLess(cv2.countNonZero(skeleton), cv2.countNonZero(mask) // 3)
        self.assertGreater(int(skeleton[40, 40]), 0)


if __name__ == "__main__":
    unittest.main()
