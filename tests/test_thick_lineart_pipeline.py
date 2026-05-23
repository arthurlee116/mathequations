import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from mathequations.__main__ import build_parser
from mathequations.thick_lineart_pipeline import run_thick_lineart_pipeline


class ThickLineartPipelineTests(unittest.TestCase):
    def _write_source(self, path: Path) -> None:
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        cv2.line(image, (20, 30), (130, 30), (0, 0, 0), thickness=9)
        cv2.line(image, (20, 60), (130, 60), (0, 220, 255), thickness=7)
        cv2.ellipse(image, (80, 92), (14, 10), 0, 0, 360, (90, 70, 210), thickness=-1)
        cv2.imwrite(str(path), image)

    def test_parser_accepts_thick_lineart_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "thick-lineart",
                "--input",
                "source.jpeg",
                "--out",
                "output/koala_thick",
                "--target",
                "1800",
                "--offset-step",
                "1",
                "--max-offsets",
                "9",
                "--keep-diagnostics",
            ]
        )

        self.assertEqual(args.command, "thick-lineart")
        self.assertEqual(args.offset_step, 1.0)
        self.assertEqual(args.max_offsets, 9)
        self.assertTrue(args.keep_diagnostics)

    def test_pipeline_writes_colored_stacked_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "koala_like.jpeg"
            out_dir = root / "out"
            self._write_source(image_path)

            result = run_thick_lineart_pipeline(
                image_path=image_path,
                out_dir=out_dir,
                target=90,
                offset_step_pixels=1.0,
                max_offsets=5,
                keep_diagnostics=True,
            )

            self.assertGreater(result.equation_count, result.source_segment_count)
            for name in [
                "clean_input.png",
                "black_mask.png",
                "yellow_mask.png",
                "pink_mask.png",
                "black_skeleton.png",
                "yellow_skeleton.png",
                "pink_skeleton.png",
                "stroke_width_diagnostics.json",
                "stacked_preview.png",
                "function_preview.png",
                "function_preview_highres.png",
                "equations.txt",
                "desmos_latex.txt",
                "desmos_expressions.json",
                "segments.json",
                "selected_equations.txt",
            ]:
                self.assertTrue((out_dir / name).exists(), name)

            payload = json.loads((out_dir / "segments.json").read_text(encoding="utf-8"))
            segments = payload["segments"]
            colors = {segment["color_name"] for segment in segments}
            self.assertEqual(colors, {"black", "yellow", "pink"})
            self.assertTrue(all("offset_pixels" in segment for segment in segments))
            self.assertTrue(all("offset_distance" in segment for segment in segments))

            offsets_by_color_and_source = defaultdict(set)
            for segment in segments:
                key = (segment["color_name"], segment["source_segment_id"])
                offsets_by_color_and_source[key].add(segment["offset_index"])
            for color_name in ["black", "yellow", "pink"]:
                self.assertTrue(
                    any(
                        color == color_name and len(offsets) > 1
                        for (color, _source_id), offsets in offsets_by_color_and_source.items()
                    ),
                    color_name,
                )

    def test_pipeline_rejects_even_max_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "koala_like.jpeg"
            self._write_source(image_path)

            with self.assertRaisesRegex(ValueError, "max_offsets must be an odd positive integer"):
                run_thick_lineart_pipeline(
                    image_path=image_path,
                    out_dir=root / "out",
                    max_offsets=4,
                )


if __name__ == "__main__":
    unittest.main()
