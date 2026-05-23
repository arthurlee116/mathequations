import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mathequations.__main__ import build_parser
from mathequations.lineart_pipeline import run_lineart_pipeline


class LineartPipelineTests(unittest.TestCase):
    def test_run_lineart_pipeline_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "lineart.png"
            out_dir = root / "out"
            image = np.full((120, 120, 3), 255, dtype=np.uint8)
            cv2.line(image, (15, 30), (105, 30), (180, 180, 180), thickness=2)
            cv2.ellipse(image, (60, 70), (35, 20), 0, 0, 180, (0, 0, 0), thickness=2)
            cv2.imwrite(str(image_path), image)

            result = run_lineart_pipeline(
                image_path=image_path,
                out_dir=out_dir,
                target=40,
                fit_mode="mixed",
            )

            self.assertGreater(result.equation_count, 0)
            for name in [
                "clean_input.png",
                "line_mask.png",
                "skeleton.png",
                "stroke_preview.png",
                "function_preview.png",
                "equations.txt",
                "desmos_latex.txt",
                "desmos_expressions.json",
                "segments.json",
                "selected_equations.txt",
            ]:
                self.assertTrue((out_dir / name).exists(), name)

            payload = json.loads((out_dir / "segments.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["fit_mode"], "mixed")
            self.assertGreater(payload["metadata"]["stroke_count"], 0)

    def test_parser_has_lineart_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "lineart",
                "--input",
                "source.png",
                "--out",
                "output/lineart",
                "--target",
                "80",
                "--fit-mode",
                "mixed",
            ]
        )

        self.assertEqual(args.command, "lineart")
        self.assertEqual(args.target, 80)
        self.assertEqual(args.fit_mode, "mixed")

    def test_run_lineart_pipeline_rejects_unknown_trace_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "lineart.png"
            image = np.full((20, 20, 3), 255, dtype=np.uint8)
            cv2.line(image, (3, 10), (17, 10), (0, 0, 0), thickness=1)
            cv2.imwrite(str(image_path), image)

            with self.assertRaises(ValueError):
                run_lineart_pipeline(
                    image_path=image_path,
                    out_dir=root / "out",
                    trace_mode="nope",
                )


if __name__ == "__main__":
    unittest.main()
