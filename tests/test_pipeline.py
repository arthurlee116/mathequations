import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mathequations.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_expected_outputs_with_approximate_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "synthetic.png"
            out_dir = root / "out"
            image = np.full((120, 120, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (25, 25), (95, 95), (0, 215, 255), thickness=-1)
            cv2.imwrite(str(image_path), image)

            result = run_pipeline(image_path=image_path, out_dir=out_dir, target=40)

            self.assertTrue((out_dir / "mask.png").exists())
            self.assertTrue((out_dir / "preview.png").exists())
            self.assertTrue((out_dir / "equations.txt").exists())
            self.assertTrue((out_dir / "segments.json").exists())
            self.assertTrue((out_dir / "selected_equations.txt").exists())
            self.assertGreaterEqual(result.segment_count, 20)
            self.assertLessEqual(result.segment_count, 60)


if __name__ == "__main__":
    unittest.main()
