import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from mathequations.__main__ import build_parser
from mathequations.point_cloud_pipeline import (
    clean_colored_points,
    colored_horizontal_runs,
    extract_docx_points,
    run_point_cloud_pipeline,
    source_foreground_points,
)


def _write_points_docx(path: Path, points: list[tuple[int, int]]) -> None:
    body = "".join(
        f"<w:p><w:r><w:t>({x}, {-y})</w:t></w:r></w:p>"
        for x, y in points
    )
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


class PointCloudPipelineTests(unittest.TestCase):
    def test_extract_docx_points_reads_negative_y_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "points.docx"
            _write_points_docx(path, [(3, 4), (5, 6)])

            points = extract_docx_points(path)

        self.assertEqual(points, {(3, 4), (5, 6)})

    def test_colored_horizontal_runs_group_by_row_and_color(self):
        image = np.full((10, 20, 3), 255, dtype=np.uint8)
        image[4, 3:6] = (0, 0, 0)
        image[4, 8:10] = (0, 220, 255)

        runs = colored_horizontal_runs({(3, 4), (4, 4), (5, 4), (8, 4), (9, 4)}, image)

        self.assertEqual(runs, [(3, 5, 4, "black", "#000000"), (8, 9, 4, "yellow", "#f5d529")])

    def test_source_foreground_points_repairs_missing_docx_regions(self):
        image = np.full((10, 20, 3), 255, dtype=np.uint8)
        image[4, 8:10] = (0, 220, 255)

        self.assertEqual(source_foreground_points(image), {(8, 4), (9, 4)})

    def test_clean_colored_points_removes_isolated_edge_debris(self):
        image = np.full((12, 20, 3), 255, dtype=np.uint8)
        image[4:8, 5:12] = (0, 0, 0)
        image[2, 3] = (0, 0, 0)
        points = {(x, y) for y in range(4, 8) for x in range(5, 12)} | {(3, 2)}

        cleaned = clean_colored_points(points, image, clean_kernel_size=3, min_component_area=4)

        self.assertNotIn((3, 2), cleaned)
        self.assertIn((8, 5), cleaned)

    def test_parser_accepts_docx_point_cloud_subcommand(self):
        args = build_parser().parse_args(
            [
                "docx-point-cloud",
                "--input",
                "source.png",
                "--reference-docx",
                "points.docx",
                "--out",
                "output/points",
            ]
        )

        self.assertEqual(args.command, "docx-point-cloud")
        self.assertEqual(str(args.reference_docx), "points.docx")

    def test_point_cloud_pipeline_writes_exact_point_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "source.png"
            docx_path = root / "points.docx"
            out_dir = root / "out"
            image = np.full((30, 40, 3), 255, dtype=np.uint8)
            image[10, 5:9] = (0, 0, 0)
            image[12, 20:24] = (0, 220, 255)
            cv2.imwrite(str(image_path), image)
            _write_points_docx(docx_path, [(x, 10) for x in range(5, 9)] + [(x, 12) for x in range(20, 24)])

            result = run_point_cloud_pipeline(
                image_path=image_path,
                reference_docx=docx_path,
                out_dir=out_dir,
                render_scale=2,
                clean_kernel_size=1,
                min_component_area=1,
            )

            self.assertEqual(result.point_count, 8)
            self.assertEqual(result.equation_count, 2)
            self.assertEqual(result.color_counts, {"black": 1, "yellow": 1})
            for name in [
                "clean_input.png",
                "point_cloud_mask.png",
                "point_cloud_direct.png",
                "function_preview.png",
                "function_preview_highres.png",
                "segments.json",
                "desmos.html",
            ]:
                self.assertTrue((out_dir / name).exists(), name)
            payload = json.loads((out_dir / "segments.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["trace_mode"], "docx-point-cloud")
            self.assertEqual(payload["metadata"]["source_repair_point_count"], 0)
            self.assertEqual({segment["color_name"] for segment in payload["segments"]}, {"black", "yellow"})


if __name__ == "__main__":
    unittest.main()
