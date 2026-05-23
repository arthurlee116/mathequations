import tempfile
import unittest
from pathlib import Path

import cv2

from mathequations.vector_model import VectorArtwork, VectorShape
from mathequations.vector_render import render_vector_artwork, svg_string


class VectorRenderTests(unittest.TestCase):
    def test_svg_string_contains_paths_for_filled_and_stroked_shapes(self):
        artwork = VectorArtwork(
            width=20,
            height=20,
            shapes=[
                VectorShape(
                    shape_id="square",
                    contours=[[(2, 2), (18, 2), (18, 18), (2, 18)]],
                    fill="#ff0000",
                    stroke="#000000",
                    stroke_width=2,
                    z_index=0,
                )
            ],
        )

        svg = svg_string(artwork)

        self.assertIn('<svg xmlns="http://www.w3.org/2000/svg"', svg)
        self.assertIn('id="square"', svg)
        self.assertIn('fill="#ff0000"', svg)
        self.assertIn('stroke="#000000"', svg)

    def test_render_vector_artwork_draws_filled_shape(self):
        artwork = VectorArtwork(
            width=20,
            height=20,
            shapes=[
                VectorShape(
                    shape_id="square",
                    contours=[[(2, 2), (18, 2), (18, 18), (2, 18)]],
                    fill="#ff0000",
                    stroke=None,
                    stroke_width=0,
                    z_index=0,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preview.png"
            render_vector_artwork(artwork, path)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        self.assertIsNotNone(image)
        self.assertEqual(image[10, 10].tolist(), [0, 0, 255])


if __name__ == "__main__":
    unittest.main()
