import unittest

from mathequations.curve_equations import (
    bezier_cubic_segment,
    linear_segment,
    vertical_segment,
)
from mathequations.offset_segments import expand_segment_offsets, offset_indices


class OffsetSegmentTests(unittest.TestCase):
    def test_offset_indices_are_total_odd_stack_count(self):
        self.assertEqual(
            offset_indices(max_offsets=5, radius_pixels=9.0, offset_step_pixels=1.0),
            [-2, -1, 0, 1, 2],
        )
        self.assertEqual(
            offset_indices(max_offsets=9, radius_pixels=1.2, offset_step_pixels=1.0),
            [-1, 0, 1],
        )

        with self.assertRaisesRegex(ValueError, "max_offsets must be an odd positive integer"):
            offset_indices(max_offsets=4, radius_pixels=5.0, offset_step_pixels=1.0)

        with self.assertRaisesRegex(ValueError, "offset_step_pixels must be positive"):
            offset_indices(max_offsets=5, radius_pixels=5.0, offset_step_pixels=0.0)

    def test_linear_segment_expands_to_parallel_colored_segments(self):
        source = linear_segment(7, 3, (-2.0, 0.0), (2.0, 0.0))

        segments = expand_segment_offsets(
            source,
            radius_pixels=2.0,
            scale=0.5,
            offset_step_pixels=1.0,
            max_offsets=5,
            color_name="black",
            color="#000000",
            start_segment_id=20,
        )

        self.assertEqual([segment["offset_index"] for segment in segments], [-2, -1, 0, 1, 2])
        self.assertEqual([segment["segment_id"] for segment in segments], [20, 21, 22, 23, 24])
        self.assertEqual({segment["source_segment_id"] for segment in segments}, {7})
        self.assertEqual({segment["color_name"] for segment in segments}, {"black"})
        self.assertEqual(segments[0]["offset_pixels"], -2.0)
        self.assertEqual(segments[0]["offset_distance"], -1.0)
        self.assertEqual(segments[2]["b"], 0.0)

    def test_vertical_segment_offsets_x_coordinate(self):
        source = vertical_segment(1, 9, (4.0, -2.0), (4.0, 2.0))

        segments = expand_segment_offsets(
            source,
            radius_pixels=1.0,
            scale=0.25,
            offset_step_pixels=1.0,
            max_offsets=3,
            color_name="yellow",
            color="#f5d529",
            start_segment_id=5,
        )

        self.assertEqual([round(segment["c"], 2) for segment in segments], [3.75, 4.0, 4.25])
        self.assertEqual({segment["color"] for segment in segments}, {"#f5d529"})

    def test_curve_segment_offsets_as_parametric_polyline(self):
        source = bezier_cubic_segment(
            3,
            2,
            control_points=((0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0)),
            fit_error=0.1,
        )

        segments = expand_segment_offsets(
            source,
            radius_pixels=1.0,
            scale=0.1,
            offset_step_pixels=1.0,
            max_offsets=3,
            color_name="pink",
            color="#d24b6a",
            start_segment_id=30,
        )

        self.assertEqual(len(segments), 3)
        self.assertTrue(all(segment["type"] == "parametric_polyline" for segment in segments))
        self.assertTrue(all(segment["source_segment_id"] == 3 for segment in segments))
        self.assertTrue(all(segment["stroke_radius_pixels"] == 1.0 for segment in segments))


if __name__ == "__main__":
    unittest.main()
