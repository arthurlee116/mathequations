import unittest

import cv2
import numpy as np

from mathequations.image_processing import foreground_mask
from mathequations.lucas_pipeline import (
    extract_feature_contours,
    extract_internal_line_contours,
    extract_line_mask,
    extract_outline_contours,
)


class LucasEdgeTests(unittest.TestCase):
    def test_extract_line_mask_finds_dark_internal_lines(self):
        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (70, 70), (80, 220, 160), thickness=-1)
        cv2.line(image, (20, 40), (60, 40), (0, 0, 0), thickness=4)

        mask = extract_line_mask(image)

        self.assertGreater(mask[40, 40], 0)
        self.assertEqual(mask[0, 0], 0)

    def test_extract_outline_contours_traces_single_silhouette(self):
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        cv2.circle(image, (60, 60), 40, (100, 230, 160), thickness=-1)
        cv2.circle(image, (60, 60), 40, (0, 0, 0), thickness=8)

        contours = extract_outline_contours(foreground_mask(image))

        self.assertEqual(len(contours), 1)
        self.assertGreater(cv2.contourArea(contours[0]), 4000)

    def test_extract_internal_line_contours_filters_tiny_speckles(self):
        foreground = np.full((160, 160), 255, dtype=np.uint8)
        line_mask = np.zeros((160, 160), dtype=np.uint8)
        cv2.rectangle(line_mask, (45, 65), (115, 85), 255, thickness=-1)
        cv2.circle(line_mask, (25, 25), 2, 255, thickness=-1)
        cv2.circle(line_mask, (130, 35), 2, 255, thickness=-1)

        contours = extract_internal_line_contours(
            line_mask,
            foreground,
            min_component_area=100,
            min_component_extent=10,
            foreground_erode_px=3,
        )

        self.assertEqual(len(contours), 1)
        x, y, width, height = cv2.boundingRect(contours[0])
        self.assertGreaterEqual(width, 60)
        self.assertGreaterEqual(height, 15)

    def test_extract_feature_contours_keeps_symmetric_features_without_dark_smudge(self):
        image = np.full((180, 180, 3), 255, dtype=np.uint8)
        cv2.circle(image, (90, 92), 70, (100, 230, 160), thickness=-1)
        cv2.ellipse(image, (48, 78), (22, 34), 0, 0, 360, (170, 90, 150), thickness=-1)
        cv2.ellipse(image, (132, 78), (22, 34), 0, 0, 360, (170, 90, 150), thickness=-1)
        cv2.ellipse(image, (48, 100), (20, 9), 0, 0, 360, (20, 20, 25), thickness=-1)
        cv2.ellipse(image, (132, 100), (20, 9), 0, 0, 360, (20, 20, 25), thickness=-1)
        nose = np.array([[(82, 126), (98, 126), (90, 108)]], dtype=np.int32)
        cv2.fillPoly(image, nose, (20, 20, 25))
        cv2.rectangle(image, (8, 45), (15, 105), (30, 30, 35), thickness=-1)

        contours = extract_feature_contours(
            image,
            foreground_mask(image),
            foreground_erode_px=5,
        )

        boxes = [cv2.boundingRect(contour) for contour in contours]
        self.assertTrue(any(20 <= x <= 35 and 40 <= y <= 55 for x, y, _, _ in boxes))
        self.assertTrue(any(105 <= x <= 120 and 40 <= y <= 55 for x, y, _, _ in boxes))
        self.assertTrue(any(78 <= x <= 86 and 105 <= y <= 120 for x, y, _, _ in boxes))
        self.assertFalse(any(x <= 16 and height >= 45 for x, _, _, height in boxes))


if __name__ == "__main__":
    unittest.main()
