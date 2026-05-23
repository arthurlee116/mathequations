import unittest

import numpy as np

from mathequations.centerline_graph import (
    Endpoint,
    Junction,
    build_skeleton_graph,
    estimate_endpoint_tangent,
    trace_raw_branches,
)


class CenterlineGraphTests(unittest.TestCase):
    def test_extracts_endpoints_from_broken_horizontal_skeleton(self):
        skeleton = np.zeros((20, 40), dtype=np.uint8)
        skeleton[10, 4:16] = 255
        skeleton[10, 22:34] = 255

        graph = build_skeleton_graph(skeleton)

        self.assertEqual(len(graph.endpoints), 4)
        self.assertTrue(all(isinstance(endpoint, Endpoint) for endpoint in graph.endpoints))
        self.assertIn((4.0, 10.0), [endpoint.point for endpoint in graph.endpoints])
        self.assertIn((33.0, 10.0), [endpoint.point for endpoint in graph.endpoints])

    def test_extracts_junction_from_t_skeleton(self):
        skeleton = np.zeros((30, 30), dtype=np.uint8)
        skeleton[15, 5:25] = 255
        skeleton[5:16, 15] = 255

        graph = build_skeleton_graph(skeleton)

        self.assertGreaterEqual(len(graph.junctions), 1)
        self.assertTrue(all(isinstance(junction, Junction) for junction in graph.junctions))
        self.assertIn((15.0, 15.0), [junction.point for junction in graph.junctions])

    def test_traces_raw_branches_between_endpoints_and_junctions(self):
        skeleton = np.zeros((30, 30), dtype=np.uint8)
        skeleton[15, 5:25] = 255
        skeleton[5:16, 15] = 255
        graph = build_skeleton_graph(skeleton)

        branches = trace_raw_branches(graph)

        self.assertGreaterEqual(len(branches), 3)
        self.assertTrue(all(branch.point_count >= 2 for branch in branches))
        self.assertTrue(any(branch.start_junction_id is not None for branch in branches))

    def test_estimates_endpoint_tangent_from_branch_pixels(self):
        skeleton = np.zeros((20, 30), dtype=np.uint8)
        skeleton[10, 5:20] = 255
        graph = build_skeleton_graph(skeleton)
        branch = trace_raw_branches(graph)[0]
        left = min(graph.endpoints, key=lambda endpoint: endpoint.point[0])
        right = max(graph.endpoints, key=lambda endpoint: endpoint.point[0])

        left_tangent = estimate_endpoint_tangent(branch, left)
        right_tangent = estimate_endpoint_tangent(branch, right)

        self.assertLess(left_tangent[0], -0.9)
        self.assertAlmostEqual(left_tangent[1], 0.0, places=6)
        self.assertGreater(right_tangent[0], 0.9)
        self.assertAlmostEqual(right_tangent[1], 0.0, places=6)

    def test_preserves_closed_loops_as_raw_branch_candidates(self):
        skeleton = np.zeros((30, 30), dtype=np.uint8)
        skeleton[8, 8:22] = 255
        skeleton[21, 8:22] = 255
        skeleton[8:22, 8] = 255
        skeleton[8:22, 21] = 255
        graph = build_skeleton_graph(skeleton)

        branches = trace_raw_branches(graph)

        self.assertEqual(len(graph.endpoints), 0)
        self.assertEqual(len(graph.junctions), 0)
        self.assertEqual(len(branches), 1)
        self.assertTrue(branches[0].closed)


if __name__ == "__main__":
    unittest.main()
