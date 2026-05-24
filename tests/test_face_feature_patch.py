import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from mathequations.curve_equations import linear_segment
from mathequations.__main__ import build_parser
from mathequations.face_feature_patch import (
    FeatureSpec,
    extract_docx_points,
    feature_function_lines,
    patch_face_feature_payload,
)


class FaceFeaturePatchTests(unittest.TestCase):
    def test_parser_accepts_patch_face_features_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "patch-face-features",
                "--segments",
                "output/current/segments.json",
                "--reference-docx",
                "include240.docx",
                "--out",
                "output/fixed",
            ]
        )

        self.assertEqual(args.command, "patch-face-features")
        self.assertEqual(str(args.reference_docx), "include240.docx")

    def test_extract_docx_points_reads_word_coordinate_paragraphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "points.docx"
            xml = (
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body>"
                "<w:p><w:r><w:t>(410, -365)</w:t></w:r></w:p>"
                "<w:p><w:r><w:t>(411, -365)</w:t></w:r></w:p>"
                "</w:body></w:document>"
            )
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", xml)

            points = extract_docx_points(path)

        self.assertEqual(points, {(410, 365), (411, 365)})

    def test_patch_payload_replaces_old_box_segments_with_reference_runs(self):
        old_feature = linear_segment(1, 1, (-40.0, 40.0), (-38.0, 40.0))
        keep = linear_segment(2, 2, (60.0, 60.0), (70.0, 60.0))
        payload = {
            "metadata": {
                "image_width": 100,
                "image_height": 100,
                "scale": 1.0,
                "target": 2,
                "fit_mode": "mixed",
                "equation_count": 2,
                "stroke_count": 2,
            },
            "strokes": [],
            "segments": [old_feature, keep],
        }
        reference_points = {
            (10, 10),
            (11, 10),
            (12, 11),
            (14, 11),
            (13, 12),
        }

        patched = patch_face_feature_payload(
            payload,
            reference_points,
            features=[FeatureSpec("mouth", (9, 9, 14, 12))],
            remove_boxes=[(9, 9, 14, 12)],
        )

        self.assertEqual(patched["metadata"]["equation_count"], 5)
        self.assertEqual(patched["metadata"]["face_feature_patch"]["removed_segment_count"], 1)
        self.assertEqual(patched["segments"][0]["segment_id"], 2)
        added = [segment for segment in patched["segments"] if segment.get("feature_override") == "mouth"]
        self.assertEqual(len(added), 4)
        self.assertTrue(all(segment["type"] == "linear" for segment in added))

    def test_feature_function_lines_lists_only_feature_overrides(self):
        feature = linear_segment(1, 1, (-40.0, 40.0), (-38.0, 40.0))
        feature["feature_override"] = "mouth"
        ordinary = linear_segment(2, 2, (60.0, 60.0), (70.0, 60.0))

        lines = feature_function_lines([ordinary, feature])

        self.assertEqual(lines[0], "# Face feature override functions")
        self.assertIn("feature=mouth", lines[2])
        self.assertIn("domain:", lines[2])
        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
