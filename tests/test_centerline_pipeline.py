import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mathequations.__main__ import build_parser
from mathequations.centerline_bridge import find_bridge_candidates, select_bridges
from mathequations.centerline_graph import (
    build_skeleton_graph,
    build_stroke_chains,
    pair_junction_branches,
    trace_raw_branches,
)
from mathequations.curve_render import render_endpoint_overlay, render_stroke_chains
from mathequations.lineart_pipeline import run_lineart_pipeline


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

    def test_lineart_parser_accepts_centerline_v2_flags(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "lineart",
                "--input",
                "source.png",
                "--out",
                "output/v2",
                "--trace-mode",
                "centerline-v2",
                "--preprocess-scale",
                "3",
                "--render-scale",
                "2",
                "--max-bridge-gap",
                "9",
                "--bridge-angle-threshold",
                "40",
                "--local-threshold",
                "sauvola",
                "--keep-diagnostics",
            ]
        )

        self.assertEqual(args.trace_mode, "centerline-v2")
        self.assertEqual(args.preprocess_scale, 3)
        self.assertEqual(args.render_scale, 2)
        self.assertEqual(args.max_bridge_gap, 9)
        self.assertEqual(args.bridge_angle_threshold, 40)
        self.assertEqual(args.local_threshold, "sauvola")
        self.assertTrue(args.keep_diagnostics)

    def test_centerline_v2_pipeline_writes_outputs_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "broken.png"
            out_dir = root / "out"
            image = np.full((50, 90, 3), 255, dtype=np.uint8)
            cv2.line(image, (8, 25), (28, 25), (0, 0, 0), thickness=1)
            cv2.line(image, (36, 25), (58, 25), (0, 0, 0), thickness=1)
            cv2.line(image, (66, 25), (82, 25), (0, 0, 0), thickness=1)
            cv2.imwrite(str(image_path), image)

            result = run_lineart_pipeline(
                image_path=image_path,
                out_dir=out_dir,
                target=20,
                fit_mode="mixed",
                trace_mode="centerline-v2",
                preprocess_scale=2,
                render_scale=2,
                max_bridge_gap=18,
                bridge_angle_threshold=40,
                local_threshold="sauvola",
                keep_diagnostics=True,
            )

            self.assertGreater(result.equation_count, 0)
            for name in [
                "clean_input.png",
                "clean_input_highres.png",
                "line_mask.png",
                "line_mask_highres.png",
                "skeleton.png",
                "skeleton_highres.png",
                "stroke_preview.png",
                "endpoint_overlay.png",
                "bridge_candidates.json",
                "bridged_strokes_preview.png",
                "function_preview.png",
                "function_preview_highres.png",
                "trace_diagnostics.json",
                "segments.json",
            ]:
                self.assertTrue((out_dir / name).exists(), name)

            payload = __import__("json").loads((out_dir / "segments.json").read_text(encoding="utf-8"))
            metadata = payload["metadata"]
            self.assertEqual(metadata["trace_mode"], "centerline-v2")
            self.assertGreater(metadata["raw_branch_count"], metadata["final_stroke_chain_count"])
            self.assertGreater(metadata["accepted_bridge_count"], 0)
            self.assertEqual(metadata["equation_count"], result.equation_count)
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
