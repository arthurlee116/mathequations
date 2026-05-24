import unittest

import cv2
import numpy as np

from mathequations.skeleton_graph import StrokePath, trace_skeleton_paths


class SkeletonGraphTests(unittest.TestCase):
    def test_trace_skeleton_paths_returns_one_open_horizontal_stroke(self):
        skeleton = np.zeros((30, 50), dtype=np.uint8)
        skeleton[15, 5:45] = 255

        strokes = trace_skeleton_paths(skeleton, min_length=5)

        self.assertEqual(len(strokes), 1)
        self.assertFalse(strokes[0].closed)
        self.assertGreaterEqual(len(strokes[0].points), 35)
        self.assertEqual(strokes[0].points[0][1], 15.0)

    def test_trace_skeleton_paths_splits_at_junction(self):
        skeleton = np.zeros((40, 40), dtype=np.uint8)
        skeleton[20, 5:35] = 255
        skeleton[5:21, 20] = 255

        strokes = trace_skeleton_paths(skeleton, min_length=5)

        self.assertGreaterEqual(len(strokes), 3)
        self.assertTrue(all(isinstance(stroke, StrokePath) for stroke in strokes))

    def test_trace_skeleton_paths_handles_closed_loop(self):
        skeleton = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(skeleton, (30, 30), 15, 255, thickness=1)

        strokes = trace_skeleton_paths(skeleton, min_length=10)

        self.assertEqual(len(strokes), 1)
        self.assertTrue(strokes[0].closed)
        self.assertGreater(len(strokes[0].points), 40)


if __name__ == "__main__":
    unittest.main()
