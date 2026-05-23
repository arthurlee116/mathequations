import unittest

import numpy as np

from mathequations.lucas_pipeline import extract_region_contours


class LucasFillExportTests(unittest.TestCase):
    def test_extract_region_contours_returns_one_region_per_label(self):
        region_mask = np.zeros((40, 40), dtype=np.uint8)
        region_mask[5:20, 5:20] = 1
        region_mask[22:35, 22:35] = 2

        regions = extract_region_contours(region_mask, min_area=20)

        self.assertEqual(sorted(regions.keys()), [1, 2])
        self.assertEqual(len(regions[1]), 1)
        self.assertEqual(len(regions[2]), 1)


if __name__ == "__main__":
    unittest.main()
