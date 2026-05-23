import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mathequations.centerline_bridge import find_bridge_candidates, select_bridges
from mathequations.centerline_graph import (
    build_skeleton_graph,
    build_stroke_chains,
    pair_junction_branches,
    trace_raw_branches,
)
from mathequations.curve_render import render_endpoint_overlay, render_stroke_chains


class CenterlinePipelineRenderingTests(unittest.TestCase):
    def test_endpoint_overlay_marks_endpoints_and_accepted_bridges(self):
        skeleton = np.zeros((20, 45), dtype=np.uint8)
        skeleton[10, 4:14] = 255
        skeleton[10, 22:34] = 255
        graph = build_skeleton_graph(skeleton)
        trace_raw_branches(graph)
        bridges = select_bridges(find_bridge_candidates(graph, max_gap=10, angle_threshold_degrees=40))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "endpoint_overlay.png"
            render_endpoint_overlay(graph.endpoints, bridges, path, image_size=(45, 20))
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        self.assertIsNotNone(image)
        self.assertLess(int(np.min(image)), 250)
        self.assertGreater(int(image[:, :, 1].max()), 180)

    def test_bridged_stroke_preview_renders_chain_points(self):
        skeleton = np.zeros((20, 50), dtype=np.uint8)
        skeleton[10, 4:14] = 255
        skeleton[10, 20:30] = 255
        skeleton[10, 36:46] = 255
        graph = build_skeleton_graph(skeleton)
        branches = trace_raw_branches(graph)
        bridges = select_bridges(find_bridge_candidates(graph, max_gap=8, angle_threshold_degrees=35))
        chains = build_stroke_chains(graph, branches, bridges, pair_junction_branches(graph))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bridged_strokes_preview.png"
            render_stroke_chains(chains, path, image_size=(50, 20))
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        self.assertEqual(len(chains), 1)
        self.assertIsNotNone(image)
        self.assertLess(int(np.min(image)), 250)


if __name__ == "__main__":
    unittest.main()
