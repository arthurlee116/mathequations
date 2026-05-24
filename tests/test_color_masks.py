import unittest

import cv2
import numpy as np

from mathequations.color_masks import (
    KOALA_COLOR_SPECS,
    extract_koala_color_masks,
    require_color_masks,
)


class ColorMaskTests(unittest.TestCase):
    def test_extract_koala_color_masks_separates_required_colors(self):
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        cv2.line(image, (10, 20), (100, 20), (0, 0, 0), thickness=7)
        cv2.line(image, (10, 45), (100, 45), (0, 220, 255), thickness=5)
        cv2.ellipse(image, (60, 65), (10, 7), 0, 0, 360, (90, 70, 210), thickness=-1)

        masks = extract_koala_color_masks(image, min_component_area=4)

        self.assertEqual(set(masks), {"black", "yellow", "pink"})
        self.assertEqual(KOALA_COLOR_SPECS["black"].hex_color, "#000000")
        self.assertEqual(KOALA_COLOR_SPECS["yellow"].hex_color, "#f5d529")
        self.assertEqual(KOALA_COLOR_SPECS["pink"].hex_color, "#d24b6a")
        self.assertGreater(cv2.countNonZero(masks["black"].mask), 300)
        self.assertGreater(cv2.countNonZero(masks["yellow"].mask), 200)
        self.assertGreater(cv2.countNonZero(masks["pink"].mask), 80)

        overlap = cv2.bitwise_and(masks["black"].mask, masks["yellow"].mask)
        self.assertEqual(cv2.countNonZero(overlap), 0)

    def test_require_color_masks_rejects_missing_required_color(self):
        masks = extract_koala_color_masks(np.full((30, 30, 3), 255, dtype=np.uint8))

        with self.assertRaisesRegex(
            ValueError,
            "missing required color masks: black, yellow, pink",
        ):
            require_color_masks(masks)


if __name__ == "__main__":
    unittest.main()
