import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mathequations.lucas_vector_pipeline import (
    build_lucas_vector_artwork,
    run_lucas_vector_pipeline,
)


class LucasVectorPipelineTests(unittest.TestCase):
    def test_build_lucas_vector_artwork_creates_semantic_shapes(self):
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        cv2.circle(image, (60, 58), 44, (105, 230, 150), thickness=-1)
        cv2.circle(image, (60, 58), 44, (0, 0, 0), thickness=5)
        cv2.ellipse(image, (42, 54), (16, 22), 0, 0, 360, (170, 90, 150), thickness=-1)
        cv2.ellipse(image, (78, 54), (16, 22), 0, 0, 360, (170, 90, 150), thickness=-1)
        cv2.fillPoly(
            image,
            [np.array([(1, 50), (12, 36), (12, 68)], dtype=np.int32)],
            (170, 90, 150),
        )
        cv2.rectangle(image, (34, 72), (50, 80), (0, 0, 0), thickness=-1)
        cv2.rectangle(image, (70, 72), (86, 80), (0, 0, 0), thickness=-1)
        cv2.rectangle(image, (45, 90), (75, 112), (80, 30, 20), thickness=-1)
        cv2.rectangle(image, (45, 90), (75, 112), (0, 0, 0), thickness=3)
        cv2.rectangle(image, (40, 94), (48, 102), (25, 210, 250), thickness=-1)

        artwork = build_lucas_vector_artwork(image)
        shape_ids = {shape.shape_id for shape in artwork.shapes}

        self.assertIn("silhouette", shape_ids)
        self.assertIn("green_body", shape_ids)
        self.assertIn("purple_eye_regions", shape_ids)
        self.assertIn("black_eye_lids", shape_ids)
        self.assertIn("purple_regions", shape_ids)
        self.assertIn("yellow_accents", shape_ids)
        self.assertIn("black_details", shape_ids)
        self.assertFalse(any("gray" in shape_id for shape_id in shape_ids))

        eye_shape = next(shape for shape in artwork.shapes if shape.shape_id == "purple_eye_regions")
        self.assertEqual(len(eye_shape.contours), 2)
        lid_shape = next(shape for shape in artwork.shapes if shape.shape_id == "black_eye_lids")
        self.assertEqual(len(lid_shape.contours), 2)

    def test_run_lucas_vector_pipeline_writes_svg_preview_and_desmos_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "lucas_like.png"
            out_dir = root / "out"
            image = np.full((120, 120, 3), 255, dtype=np.uint8)
            cv2.circle(image, (60, 58), 44, (105, 230, 150), thickness=-1)
            cv2.circle(image, (60, 58), 44, (0, 0, 0), thickness=5)
            cv2.ellipse(image, (42, 54), (16, 22), 0, 0, 360, (170, 90, 150), thickness=-1)
            cv2.ellipse(image, (78, 54), (16, 22), 0, 0, 360, (170, 90, 150), thickness=-1)
            cv2.imwrite(str(image_path), image)

            result = run_lucas_vector_pipeline(image_path=image_path, out_dir=out_dir)

            self.assertGreater(result.shape_count, 0)
            self.assertTrue((out_dir / "lucas_clean.svg").exists())
            self.assertTrue((out_dir / "lucas_vector_preview.png").exists())
            self.assertTrue((out_dir / "lucas_function_render.png").exists())
            self.assertTrue((out_dir / "lucas_vector_segments.json").exists())
            self.assertTrue((out_dir / "desmos_vector_expressions.json").exists())


if __name__ == "__main__":
    unittest.main()
