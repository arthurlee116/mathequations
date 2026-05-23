import unittest

from mathequations.layers import EquationLayer, FillRegion


class LayerTests(unittest.TestCase):
    def test_equation_layer_stores_name_kind_and_equations(self):
        layer = EquationLayer(
            name="outer_outline",
            kind="line",
            equations=["y = x {0 <= x <= 1}"],
        )

        self.assertEqual(layer.name, "outer_outline")
        self.assertEqual(layer.kind, "line")
        self.assertEqual(layer.equations, ["y = x {0 <= x <= 1}"])

    def test_fill_region_has_gray_value_and_boundary_ids(self):
        region = FillRegion(
            region_id=3,
            gray=160,
            boundary_segment_ids=[1, 2, 3],
            label="shirt_mid_gray",
        )

        self.assertEqual(region.gray, 160)
        self.assertEqual(region.label, "shirt_mid_gray")
        self.assertEqual(region.boundary_segment_ids, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
