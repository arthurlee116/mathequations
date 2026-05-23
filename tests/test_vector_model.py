import unittest

from mathequations.vector_model import VectorArtwork, VectorShape, svg_path_from_points


class VectorModelTests(unittest.TestCase):
    def test_svg_path_from_points_serializes_closed_path(self):
        path = svg_path_from_points([(0, 0), (10, 0), (10, 5)], closed=True)

        self.assertEqual(path, "M 0 0 L 10 0 L 10 5 Z")

    def test_vector_artwork_orders_shapes_by_z_index(self):
        back = VectorShape(
            shape_id="back",
            contours=[[(0, 0), (10, 0), (10, 10), (0, 10)]],
            fill="#000000",
            stroke=None,
            stroke_width=0,
            z_index=1,
        )
        front = VectorShape(
            shape_id="front",
            contours=[[(2, 2), (8, 2), (8, 8), (2, 8)]],
            fill="#ffffff",
            stroke=None,
            stroke_width=0,
            z_index=5,
        )

        artwork = VectorArtwork(width=10, height=10, shapes=[front, back])

        self.assertEqual([shape.shape_id for shape in artwork.ordered_shapes()], ["back", "front"])


if __name__ == "__main__":
    unittest.main()
