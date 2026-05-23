import unittest

import numpy as np

from mathequations.lucas_pipeline import quantize_to_gray_regions


class LucasRegionTests(unittest.TestCase):
    def test_quantize_to_gray_regions_produces_limited_gray_values(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[:10, :] = [20, 20, 80]
        image[10:, :] = [200, 200, 40]
        foreground = np.full((20, 20), 255, dtype=np.uint8)

        gray_image, region_mask = quantize_to_gray_regions(image, foreground, levels=4)

        values = sorted(set(gray_image[foreground > 0].flatten().tolist()))
        self.assertLessEqual(len(values), 4)
        self.assertEqual(region_mask.shape, (20, 20))


if __name__ == "__main__":
    unittest.main()
