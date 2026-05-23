# Lucas Client Character-To-Equations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the cleaned Lucas character image into a coordinate-plane math-art export with visible outlines, internal details, and grayscale fill regions.

**Architecture:** Extend the current single-mask outline pipeline into a layered pipeline. The new pipeline should produce separate layers for outer silhouette, internal edges/details, and simplified grayscale fill regions, then export Desmos-compatible restricted equations plus previews.

**Tech Stack:** Python 3, OpenCV, NumPy, Pillow, pytest/unittest, existing `mathequations` package.

---

## Brainstorm Summary

The hammer-and-sickle pipeline worked because the input was basically one solid foreground shape. Lucas is different: the character has a large black outer stroke, internal eyes, pupils, nose, ears, hanger, shirt, cuffs, skull icon, legs, gradients, and small decorative shapes. Treating it as one foreground mask would throw away the parts that make the character recognizable. Bad bargain.

Recommended approach: build a layered vectorization pipeline:

1. **Line layer:** extract outer and internal contour lines.
2. **Region layer:** segment the character into simplified color/gray regions.
3. **Fill layer:** export those regions as shaded polygon-ish equation groups.

Rejected approaches:

- **Outer contour only:** fastest, but loses the face, shirt, skull, cuffs, and ears.
- **Canny-only edge tracing:** catches details but makes double-lines and noisy gradients.
- **Manual tracing:** accurate, but misses the point of the project.

Default target for the school/client version: enough detail to look like Lucas at a glance, not a pixel-perfect copy. Start around 1000-2000 outline equations and 6-8 grayscale fill groups.

## Existing Project Context

Current files:

- `mathequations/pipeline.py` handles one foreground mask -> contours -> linear/vertical equations.
- `mathequations/image_processing.py` handles image loading and simple foreground masking.
- `mathequations/equations.py` exports line and vertical equations.
- `mathequations/render.py` previews line segments.

Lucas source image:

- `/Users/arthur/Downloads/lucas.png`
- PNG, 1132 x 1390, RGB.
- White background already exists.
- Quick inspection found many foreground/internal contours, so the pipeline must preserve internal regions.

## Output Contract

Create a new Lucas run directory such as:

```text
output/lucas_1500/
```

It should contain:

```text
clean_input.png
debug_line_mask.png
debug_regions.png
outline_preview.png
filled_preview.png
equations_outline.txt
equations_fill.txt
equations_all.txt
segments.json
regions.json
selected_equations.txt
```

Each equation stays in one shared Cartesian coordinate plane. Use the same coordinate transform as v1:

```text
x = (u - W/2) * scale
y = (H/2 - v) * scale
```

Default `scale_width = 20`, based on the foreground bounding box width.

## Implementation Tasks

### Task 1: Add Layer Data Structures

**Files:**

- Create: `mathequations/layers.py`
- Test: `tests/test_layers.py`

- [ ] **Step 1: Write tests for layer records**

```python
import unittest

from mathequations.layers import EquationLayer, FillRegion


class LayerTests(unittest.TestCase):
    def test_equation_layer_stores_name_kind_and_equations(self):
        layer = EquationLayer(
            name="outer_outline",
            kind="line",
            equations=["y = x {0 <= x <= 1}"],
        )

        self.assertEqual(layer.name, "outer_outline")
        self.assertEqual(layer.kind, "line")
        self.assertEqual(layer.equations, ["y = x {0 <= x <= 1}"])

    def test_fill_region_has_gray_value_and_boundary_ids(self):
        region = FillRegion(
            region_id=3,
            gray=160,
            boundary_segment_ids=[1, 2, 3],
            label="shirt_mid_gray",
        )

        self.assertEqual(region.gray, 160)
        self.assertEqual(region.label, "shirt_mid_gray")
        self.assertEqual(region.boundary_segment_ids, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test and confirm failure**

```bash
rtk pytest tests/test_layers.py -q
```

Expected: fails because `mathequations.layers` does not exist.

- [ ] **Step 3: Implement `mathequations/layers.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EquationLayer:
    name: str
    kind: str
    equations: list[str]


@dataclass(frozen=True)
class FillRegion:
    region_id: int
    gray: int
    boundary_segment_ids: list[int]
    label: str
```

- [ ] **Step 4: Verify**

```bash
rtk pytest tests/test_layers.py -q
```

Expected: passes.

### Task 2: Add Clean Input Preparation

**Files:**

- Modify: `mathequations/image_processing.py`
- Test: `tests/test_lucas_preprocess.py`

- [ ] **Step 1: Write tests for white-background normalization**

```python
import unittest

import numpy as np

from mathequations.image_processing import normalize_white_background


class LucasPreprocessTests(unittest.TestCase):
    def test_nearly_white_background_becomes_pure_white(self):
        image = np.full((5, 5, 3), 248, dtype=np.uint8)
        image[2, 2] = [0, 0, 0]

        result = normalize_white_background(image, white_threshold=245)

        self.assertEqual(result[0, 0].tolist(), [255, 255, 255])
        self.assertEqual(result[2, 2].tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing test**

```bash
rtk pytest tests/test_lucas_preprocess.py -q
```

Expected: fails because function is missing.

- [ ] **Step 3: Implement**

Add:

```python
def normalize_white_background(image: np.ndarray, *, white_threshold: int = 245) -> np.ndarray:
    result = image.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result[gray >= white_threshold] = [255, 255, 255]
    return result
```

- [ ] **Step 4: Verify**

```bash
rtk pytest tests/test_lucas_preprocess.py -q
```

Expected: passes.

### Task 3: Extract Internal Line Layer

**Files:**

- Create: `mathequations/lucas_pipeline.py`
- Test: `tests/test_lucas_edges.py`

- [ ] **Step 1: Write test for internal edge extraction**

```python
import unittest

import cv2
import numpy as np

from mathequations.lucas_pipeline import extract_line_mask


class LucasEdgeTests(unittest.TestCase):
    def test_extract_line_mask_finds_dark_internal_lines(self):
        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (70, 70), (80, 220, 160), thickness=-1)
        cv2.line(image, (20, 40), (60, 40), (0, 0, 0), thickness=4)

        mask = extract_line_mask(image)

        self.assertGreater(mask[40, 40], 0)
        self.assertEqual(mask[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing test**

```bash
rtk pytest tests/test_lucas_edges.py -q
```

Expected: fails because `extract_line_mask` is missing.

- [ ] **Step 3: Implement line mask**

Use a hybrid line detector:

```python
def extract_line_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = (gray < 70).astype(np.uint8) * 255
    edges = cv2.Canny(gray, 60, 140)
    mask = cv2.bitwise_or(dark, edges)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

This intentionally includes Canny for Lucas. That is not a contradiction: Canny was wrong for the solid emblem as the first step, but Lucas has real internal detail.

- [ ] **Step 4: Verify**

```bash
rtk pytest tests/test_lucas_edges.py -q
```

Expected: passes.

### Task 4: Segment Grayscale Fill Regions

**Files:**

- Modify: `mathequations/lucas_pipeline.py`
- Test: `tests/test_lucas_regions.py`

- [ ] **Step 1: Write test for color-to-gray region quantization**

```python
import unittest

import numpy as np

from mathequations.lucas_pipeline import quantize_to_gray_regions


class LucasRegionTests(unittest.TestCase):
    def test_quantize_to_gray_regions_produces_limited_gray_values(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[:10, :] = [20, 20, 80]
        image[10:, :] = [200, 200, 40]
        foreground = np.full((20, 20), 255, dtype=np.uint8)

        gray_image, region_mask = quantize_to_gray_regions(image, foreground, levels=4)

        values = sorted(set(gray_image[foreground > 0].flatten().tolist()))
        self.assertLessEqual(len(values), 4)
        self.assertEqual(region_mask.shape, (20, 20))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing test**

```bash
rtk pytest tests/test_lucas_regions.py -q
```

Expected: fails because function is missing.

- [ ] **Step 3: Implement quantization**

Implement k-means on foreground pixels and map clusters to grayscale values:

```python
def quantize_to_gray_regions(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    levels: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = image[foreground_mask > 0].astype(np.float32)
    if len(pixels) == 0:
        raise ValueError("foreground_mask has no pixels")
    levels = max(2, min(levels, len(pixels)))
    _, labels, centers = cv2.kmeans(
        pixels,
        levels,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0),
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    luminance = (0.114 * centers[:, 0] + 0.587 * centers[:, 1] + 0.299 * centers[:, 2])
    order = np.argsort(luminance)
    gray_values = np.linspace(40, 220, levels).astype(np.uint8)
    cluster_to_gray = {int(cluster): int(gray_values[rank]) for rank, cluster in enumerate(order)}
    gray_image = np.full(foreground_mask.shape, 255, dtype=np.uint8)
    region_mask = np.zeros(foreground_mask.shape, dtype=np.uint8)
    coords = np.argwhere(foreground_mask > 0)
    for idx, (y, x) in enumerate(coords):
        cluster = int(labels[idx][0])
        gray_image[y, x] = cluster_to_gray[cluster]
        region_mask[y, x] = cluster + 1
    return gray_image, region_mask
```

- [ ] **Step 4: Verify**

```bash
rtk pytest tests/test_lucas_regions.py -q
```

Expected: passes.

### Task 5: Export Region Boundaries As Fill Groups

**Files:**

- Modify: `mathequations/lucas_pipeline.py`
- Modify: `mathequations/render.py`
- Test: `tests/test_lucas_fill_export.py`

- [ ] **Step 1: Write test for region contour extraction**

```python
import unittest

import numpy as np

from mathequations.lucas_pipeline import extract_region_contours


class LucasFillExportTests(unittest.TestCase):
    def test_extract_region_contours_returns_one_region_per_label(self):
        region_mask = np.zeros((40, 40), dtype=np.uint8)
        region_mask[5:20, 5:20] = 1
        region_mask[22:35, 22:35] = 2

        regions = extract_region_contours(region_mask, min_area=20)

        self.assertEqual(sorted(regions.keys()), [1, 2])
        self.assertEqual(len(regions[1]), 1)
        self.assertEqual(len(regions[2]), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing test**

```bash
rtk pytest tests/test_lucas_fill_export.py -q
```

Expected: fails because function is missing.

- [ ] **Step 3: Implement region contour extraction**

```python
def extract_region_contours(region_mask: np.ndarray, *, min_area: float = 80.0) -> dict[int, list[np.ndarray]]:
    result: dict[int, list[np.ndarray]] = {}
    for label in sorted(int(v) for v in np.unique(region_mask) if v != 0):
        mask = (region_mask == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        useful = [c for c in contours if cv2.contourArea(c) >= min_area]
        if useful:
            result[label] = useful
    return result
```

- [ ] **Step 4: Add filled preview rendering**

Add a renderer that draws each simplified region polygon with its gray value, then draws line equations on top. Keep this separate from the current line-only `render_segments`.

- [ ] **Step 5: Verify**

```bash
rtk pytest tests/test_lucas_fill_export.py -q
```

Expected: passes.

### Task 6: Add Lucas CLI Mode

**Files:**

- Modify: `mathequations/__main__.py`
- Modify: `mathequations/lucas_pipeline.py`
- Test: `tests/test_lucas_pipeline_smoke.py`

- [ ] **Step 1: Write smoke test**

```python
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
            self.assertGreater(result.total_equations, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing test**

```bash
rtk pytest tests/test_lucas_pipeline_smoke.py -q
```

Expected: fails because `run_lucas_pipeline` is missing.

- [ ] **Step 3: Implement `run_lucas_pipeline`**

Pipeline sequence:

1. Load image.
2. Normalize white background.
3. Build foreground mask with existing `foreground_mask`.
4. Extract line mask.
5. Extract line contours and convert them to outline equations.
6. Quantize foreground to grayscale regions.
7. Extract region contours and convert region boundaries to equations.
8. Save debug masks and previews.
9. Save outline, fill, combined equations, JSON, and selected equations.

- [ ] **Step 4: Add CLI flags**

Support:

```bash
rtk python3 -m mathequations lucas \
  --input /Users/arthur/Downloads/lucas.png \
  --out output/lucas_1500 \
  --outline-target 1500 \
  --region-target 1200 \
  --gray-levels 6
```

Keep the old emblem command working. If no subcommand is provided, preserve the existing behavior for now.

- [ ] **Step 5: Verify**

```bash
rtk pytest tests/test_lucas_pipeline_smoke.py -q
```

Expected: passes.

### Task 7: Run The Real Lucas Image

**Files:**

- No source changes expected.
- Output: `output/lucas_1500/`

- [ ] **Step 1: Run all tests**

```bash
rtk pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run real image**

```bash
rtk python3 -m mathequations lucas \
  --input /Users/arthur/Downloads/lucas.png \
  --out output/lucas_1500 \
  --outline-target 1500 \
  --region-target 1200 \
  --gray-levels 6
```

Expected: output files are written.

- [ ] **Step 3: Inspect counts**

```bash
rtk wc -l output/lucas_1500/equations_outline.txt output/lucas_1500/equations_fill.txt output/lucas_1500/equations_all.txt
```

Expected: enough equations to preserve visible details without exploding past what Desmos can reasonably handle.

- [ ] **Step 4: Inspect images**

Open or view:

```text
output/lucas_1500/outline_preview.png
output/lucas_1500/filled_preview.png
output/lucas_1500/debug_regions.png
```

Acceptance:

- Head shape, eyes, pupils, nose, ears, shirt, cuffs, skull, legs, tag, and hanger remain recognizable.
- White background remains white.
- Gray fills roughly preserve light/dark structure.
- Equations all share one coordinate plane.
- Restricted domains/ranges are present.

## Notes For Desmos Fill

Desmos polygon filling from arbitrary contour equations is awkward. For the first client-ready version, do not promise perfect filled solid areas. Export:

1. line equations for boundaries;
2. region boundary equations grouped by gray level;
3. grayscale preview image generated locally.

If the project later needs true Desmos-style shaded fills, add inequality-based fill approximations only for large simple regions. Trying to auto-generate inequalities for every tiny region now would be overkill and would probably be a mess.

## Defaults

- Input: `/Users/arthur/Downloads/lucas.png`
- Output: `output/lucas_1500`
- Coordinate width: 20 units.
- Outline equations: target 1500.
- Region boundary equations: target 1200.
- Gray levels: 6.
- Preserve internal details, even if total equation count increases.

## Self-Review

- No placeholders remain.
- The plan preserves the old emblem pipeline.
- The Lucas workflow has separate line and region layers.
- The plan includes tests before implementation.
- The plan does not pretend color fill in Desmos is trivial. Good. That trap is real.
