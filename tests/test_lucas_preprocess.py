import unittest

import numpy as np

from mathequations.image_processing import normalize_white_background


class LucasPreprocessTests(unittest.TestCase):
    def test_nearly_white_background_becomes_pure_white(self):
        image = np.full((5, 5, 3), 248, dtype=np.uint8)
        image[2, 2] = [0, 0, 0]

        result = normalize_white_background(image, white_threshold=245)

        self.assertEqual(result[0, 0].tolist(), [255, 255, 255])
        self.assertEqual(result[2, 2].tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
