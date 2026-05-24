import unittest

import numpy as np

from mathequations.centerline_bridge import (
    find_bridge_candidates,
    score_bridge_candidate,
    select_bridges,
)
from mathequations.centerline_graph import build_skeleton_graph, trace_raw_branches
from mathequations.centerline_graph import build_stroke_chains, pair_junction_branches


class CenterlineBridgeTests(unittest.TestCase):
    def test_bridges_two_aligned_broken_line_segments(self):
        skeleton = np.zeros((20, 40), dtype=np.uint8)
        skeleton[10, 4:14] = 255
        skeleton[10, 20:30] = 255
        graph = build_skeleton_graph(skeleton)
        trace_raw_branches(graph)

        candidates = find_bridge_candidates(graph, max_gap=8, angle_threshold_degrees=35)
        bridges = select_bridges(candidates)

        self.assertEqual(len(bridges), 1)
        self.assertEqual({bridges[0].endpoint_a_id, bridges[0].endpoint_b_id}, {2, 3})

    def test_rejects_nearby_segments_with_bad_tangents(self):
        skeleton = np.zeros((30, 35), dtype=np.uint8)
        skeleton[10, 4:14] = 255
        skeleton[14:25, 16] = 255
        graph = build_skeleton_graph(skeleton)
        trace_raw_branches(graph)

        candidates = find_bridge_candidates(graph, max_gap=8, angle_threshold_degrees=30)

        self.assertEqual(candidates, [])

    def test_shorter_high_confidence_continuation_beats_farther_candidate(self):
        skeleton = np.zeros((20, 60), dtype=np.uint8)
        skeleton[10, 4:14] = 255
        skeleton[10, 20:30] = 255
        skeleton[10, 36:48] = 255
        graph = build_skeleton_graph(skeleton)
        trace_raw_branches(graph)

        candidates = find_bridge_candidates(graph, max_gap=24, angle_threshold_degrees=35)
        bridges = select_bridges(candidates)

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual({bridges[0].endpoint_a_id, bridges[0].endpoint_b_id}, {2, 3})

    def test_selected_bridges_do_not_reuse_endpoint(self):
        skeleton = np.zeros((30, 50), dtype=np.uint8)
        skeleton[12, 4:14] = 255
        skeleton[12, 20:30] = 255
        skeleton[16, 20:30] = 255
        graph = build_skeleton_graph(skeleton)
        trace_raw_branches(graph)

        bridges = select_bridges(find_bridge_candidates(graph, max_gap=12, angle_threshold_degrees=45))
        endpoint_ids = [endpoint_id for bridge in bridges for endpoint_id in (bridge.endpoint_a_id, bridge.endpoint_b_id)]

        self.assertEqual(len(endpoint_ids), len(set(endpoint_ids)))

    def test_bridge_diagnostics_include_score_components(self):
        skeleton = np.zeros((20, 40), dtype=np.uint8)
        skeleton[10, 4:14] = 255
        skeleton[10, 20:30] = 255
        graph = build_skeleton_graph(skeleton)
        trace_raw_branches(graph)
        candidate = find_bridge_candidates(graph, max_gap=8, angle_threshold_degrees=35)[0]

        score = score_bridge_candidate(candidate)

        self.assertEqual(score, candidate.score)
        for key in [
            "distance_cost",
            "tangent_cost",
            "smoothness_cost",
            "mask_evidence_bonus",
            "score",
        ]:
            self.assertIn(key, candidate.diagnostics)

    def test_broken_stroke_becomes_one_chain_after_bridge_acceptance(self):
        skeleton = np.zeros((20, 50), dtype=np.uint8)
        skeleton[10, 4:14] = 255
        skeleton[10, 20:30] = 255
        skeleton[10, 36:46] = 255
        graph = build_skeleton_graph(skeleton)
        branches = trace_raw_branches(graph)
        bridges = select_bridges(find_bridge_candidates(graph, max_gap=8, angle_threshold_degrees=35))

        chains = build_stroke_chains(graph, branches, bridges, pair_junction_branches(graph))

        self.assertEqual(len(branches), 3)
        self.assertEqual(len(chains), 1)
        self.assertGreater(chains[0].point_count, max(branch.point_count for branch in branches))


if __name__ == "__main__":
    unittest.main()
