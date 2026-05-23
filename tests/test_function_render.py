import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mathequations.function_render import render_function_segments_payload


class FunctionRenderTests(unittest.TestCase):
    def test_render_function_segments_payload_fills_shapes_with_segment_colors(self):
        payload = {
            "metadata": {
                "image_width": 30,
                "image_height": 30,
                "scale": 1.0,
            },
            "shapes": [
                {"shape_id": "red_square", "fill": "#ff0000", "z_index": 0},
                {"shape_id": "green_square", "fill": "#00ff00", "z_index": 1},
            ],
            "segments": [
                {
                    "shape_id": "red_square",
                    "contour_id": 0,
                    "start": {"x": -10, "y": 10},
                    "end": {"x": 0, "y": 10},
                    "color": "#ff0000",
                },
                {
                    "shape_id": "red_square",
                    "contour_id": 0,
                    "start": {"x": 0, "y": 10},
                    "end": {"x": 0, "y": 0},
                    "color": "#ff0000",
                },
                {
                    "shape_id": "red_square",
                    "contour_id": 0,
                    "start": {"x": 0, "y": 0},
                    "end": {"x": -10, "y": 0},
                    "color": "#ff0000",
                },
                {
                    "shape_id": "red_square",
                    "contour_id": 0,
                    "start": {"x": -10, "y": 0},
                    "end": {"x": -10, "y": 10},
                    "color": "#ff0000",
                },
                {
                    "shape_id": "green_square",
                    "contour_id": 1,
                    "start": {"x": 0, "y": 0},
                    "end": {"x": 10, "y": 0},
                    "color": "#00ff00",
                },
                {
                    "shape_id": "green_square",
                    "contour_id": 1,
                    "start": {"x": 10, "y": 0},
                    "end": {"x": 10, "y": -10},
                    "color": "#00ff00",
                },
                {
                    "shape_id": "green_square",
                    "contour_id": 1,
                    "start": {"x": 10, "y": -10},
                    "end": {"x": 0, "y": -10},
                    "color": "#00ff00",
                },
                {
                    "shape_id": "green_square",
                    "contour_id": 1,
                    "start": {"x": 0, "y": -10},
                    "end": {"x": 0, "y": 0},
                    "color": "#00ff00",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "functions.png"
            render_function_segments_payload(payload, path)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        self.assertIsNotNone(image)
        unique_colors = np.unique(image.reshape((-1, 3)), axis=0)
        self.assertGreaterEqual(len(unique_colors), 3)
        self.assertEqual(image[10, 10].tolist(), [0, 0, 255])
        self.assertEqual(image[20, 20].tolist(), [0, 255, 0])


if __name__ == "__main__":
    unittest.main()
