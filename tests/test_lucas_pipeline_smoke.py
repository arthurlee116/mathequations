import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mathequations.lucas_pipeline import run_lucas_pipeline


class LucasPipelineSmokeTests(unittest.TestCase):
    def test_run_lucas_pipeline_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "lucas_like.png"
            out_dir = root / "out"
            image = np.full((120, 100, 3), 255, dtype=np.uint8)
            cv2.circle(image, (50, 45), 30, (100, 230, 160), thickness=-1)
            cv2.rectangle(image, (35, 72), (65, 110), (30, 30, 90), thickness=-1)
            cv2.line(image, (30, 45), (70, 45), (0, 0, 0), thickness=3)
            cv2.imwrite(str(image_path), image)

            result = run_lucas_pipeline(
                image_path=image_path,
                out_dir=out_dir,
                outline_target=120,
                region_target=180,
                gray_levels=4,
            )

            self.assertTrue((out_dir / "clean_input.png").exists())
            self.assertTrue((out_dir / "debug_line_mask.png").exists())
            self.assertTrue((out_dir / "debug_regions.png").exists())
            self.assertTrue((out_dir / "outline_preview.png").exists())
            self.assertTrue((out_dir / "filled_preview.png").exists())
            self.assertTrue((out_dir / "equations_all.txt").exists())
            self.assertTrue((out_dir / "desmos_equations.txt").exists())
            self.assertTrue((out_dir / "desmos_latex.txt").exists())
            self.assertTrue((out_dir / "desmos_expressions.json").exists())
            self.assertTrue((out_dir / "desmos_lucas.html").exists())
            self.assertGreater(result.total_equations, 0)


if __name__ == "__main__":
    unittest.main()
