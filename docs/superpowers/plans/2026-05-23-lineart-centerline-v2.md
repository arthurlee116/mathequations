# Lineart Centerline V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brittle skeleton-fragment lineart path with a Centerline V2 pipeline that reconstructs continuous strokes, bridges small pencil gaps, preserves likely junction continuations, fits smoother Desmos-compatible curves, and renders high-resolution previews.

**Architecture:** Keep the existing `lineart` CLI and V1 modules as a compatibility path, then add a separate Centerline V2 path behind `--trace-mode centerline-v2`. V2 should process the image at higher resolution, extract local-threshold masks, build a geometry-aware skeleton graph, reconnect broken stroke fragments, reconstruct long stroke chains, fit mixed equation segments, and write diagnostics for every stage.

**Tech Stack:** Python 3, OpenCV, NumPy, Pillow, scikit-image, pytest/unittest, existing `mathequations` geometry/equation/render helpers.

---

## Planning Note

This plan intentionally avoids large implementation snippets. The implementation agent should use the behavior, file boundaries, function names, tests, and verification commands below as constraints, then write the actual code in the style of the existing repo.

Do not paste large code from this plan into implementation. Use the tests and acceptance criteria to guide the design.

---

## File Structure

- Modify `requirements.txt`
  - Add any missing dependency needed for local threshold fallback, likely already covered by `scikit-image`.
  - Do not require `opencv-contrib-python` unless the implementation deliberately replaces `opencv-python`; V2 must work without `cv2.ximgproc`.
- Modify `mathequations/lineart_preprocess.py`
  - Add high-resolution preprocessing and local threshold mask extraction.
  - Keep existing V1 functions stable for current tests.
- Create `mathequations/centerline_graph.py`
  - Geometry-aware skeleton graph records: raw branches, endpoints, junctions, bridge candidates, stroke chains.
- Create `mathequations/centerline_bridge.py`
  - Endpoint candidate scoring, bridge selection, and junction continuation pairing.
- Create `mathequations/centerline_pipeline.py`
  - Orchestrates V2 preprocessing, graph extraction, bridging, chain reconstruction, fitting, rendering, diagnostics, and exports.
- Modify `mathequations/curve_fit.py`
  - Add chain-level fitting behavior where needed, while preserving current segment fitting tests.
- Modify `mathequations/curve_render.py`
  - Add high-resolution render scale support and diagnostic overlays.
- Modify `mathequations/lineart_pipeline.py`
  - Dispatch between `skeleton-v1` and `centerline-v2`.
  - Preserve current output contract for V1.
- Modify `mathequations/__main__.py`
  - Add V2 CLI flags.
- Modify `README.md`
  - Document Centerline V2 after implementation is verified.
- Create tests:
  - `tests/test_centerline_preprocess.py`
  - `tests/test_centerline_graph.py`
  - `tests/test_centerline_bridge.py`
  - `tests/test_centerline_pipeline.py`
  - Extend `tests/test_curve_fit.py`
  - Extend `tests/test_curve_render.py`
  - Extend `tests/test_lineart_pipeline.py`

---

### Task 1: High-Resolution Local Preprocessing

**Files:**
- Modify: `requirements.txt`
- Modify: `mathequations/lineart_preprocess.py`
- Create: `tests/test_centerline_preprocess.py`

- [ ] **Step 1: Write failing tests for high-resolution cleanup**

Create tests that prove:

- `prepare_highres_lineart()` returns a larger image when `scale=4`.
- a faint one-pixel gray line survives preprocessing and local mask extraction.
- component filtering removes isolated speckles without erasing a thin stroke.
- diagnostics include foreground density and component counts.

Use synthetic images only. Keep each test small and visual-intent clear.

- [ ] **Step 2: Run preprocessing tests and confirm failure**

Run:

```bash
rtk pytest tests/test_centerline_preprocess.py -q
```

Expected: fails because the new V2 preprocessing API does not exist.

- [ ] **Step 3: Add V2 preprocessing API**

Add these public functions without breaking current V1 tests:

- `prepare_highres_lineart(image, *, scale=4, white_threshold=246)`
- `build_local_line_mask(image, *, threshold_mode="sauvola", min_component_area=2)`
- `line_mask_diagnostics(mask)`

Implementation requirements:

- Use `cv2.resize` with high-quality interpolation for upscaling.
- Use CLAHE or equivalent contrast normalization.
- Prefer `cv2.ximgproc.niBlackThreshold` only when available.
- Provide a `skimage.filters.threshold_sauvola` fallback.
- Avoid destructive morphological opening.
- Filter speckles with connected-component statistics.

- [ ] **Step 4: Verify preprocessing tests**

Run:

```bash
rtk pytest tests/test_centerline_preprocess.py tests/test_lineart_preprocess.py -q
```

Expected: all preprocessing tests pass, including existing V1 tests.

- [ ] **Step 5: Commit preprocessing work**

Run:

```bash
rtk proxy git add requirements.txt mathequations/lineart_preprocess.py tests/test_centerline_preprocess.py tests/test_lineart_preprocess.py
rtk proxy git commit -m "feat: add centerline high-res preprocessing"
```

---

### Task 2: Geometry-Aware Skeleton Graph

**Files:**
- Create: `mathequations/centerline_graph.py`
- Create: `tests/test_centerline_graph.py`

- [ ] **Step 1: Write failing graph tests**

Create tests for:

- extracting endpoints from a broken horizontal skeleton.
- extracting a junction from a T-shaped skeleton.
- tracing raw branches between endpoints and junctions.
- estimating endpoint tangent direction from nearby path pixels.
- preserving closed loops as branch or chain candidates.

The tests should assert behavior and data shape, not exact internal traversal order unless order is part of the public contract.

- [ ] **Step 2: Run graph tests and confirm failure**

Run:

```bash
rtk pytest tests/test_centerline_graph.py -q
```

Expected: fails because `mathequations.centerline_graph` does not exist.

- [ ] **Step 3: Implement graph records and extraction**

Create focused dataclasses or named records for:

- `Endpoint`
- `Junction`
- `RawBranch`
- `StrokeChain`
- `SkeletonGraph`

Required public functions:

- `build_skeleton_graph(skeleton, *, mask=None) -> SkeletonGraph`
- `trace_raw_branches(graph) -> list[RawBranch]`
- `estimate_endpoint_tangent(branch, endpoint) -> tuple[float, float]`

Implementation requirements:

- Use 8-neighbor connectivity.
- Keep pixel-space coordinates in `(x, y)` order at public boundaries.
- Store degree classification for endpoint, regular node, and junction.
- Attach length and point-count metrics to branches.
- Do not reuse V1 `StrokePath` as the raw graph model; V2 needs richer metadata.

- [ ] **Step 4: Verify graph tests**

Run:

```bash
rtk pytest tests/test_centerline_graph.py tests/test_skeleton_graph.py -q
```

Expected: V2 graph tests pass and V1 skeleton tests remain stable.

- [ ] **Step 5: Commit graph work**

Run:

```bash
rtk proxy git add mathequations/centerline_graph.py tests/test_centerline_graph.py
rtk proxy git commit -m "feat: add centerline skeleton graph"
```

---

### Task 3: Endpoint Bridge Scoring And Selection

**Files:**
- Create: `mathequations/centerline_bridge.py`
- Create: `tests/test_centerline_bridge.py`

- [ ] **Step 1: Write failing bridge tests**

Create tests proving:

- two aligned broken line segments are bridged.
- two nearby segments with opposing or bad tangents are not bridged.
- a shorter high-confidence continuation beats a farther low-confidence candidate.
- accepted bridges do not reuse the same endpoint twice.
- bridge diagnostics include score components.

Use synthetic skeletons and masks. Keep expected geometry simple enough to inspect by eye.

- [ ] **Step 2: Run bridge tests and confirm failure**

Run:

```bash
rtk pytest tests/test_centerline_bridge.py -q
```

Expected: fails because bridge API does not exist.

- [ ] **Step 3: Implement bridge candidate scoring**

Create:

- `BridgeCandidate`
- `find_bridge_candidates(graph, *, max_gap, angle_threshold_degrees, mask=None, gray=None)`
- `score_bridge_candidate(candidate, *, mask=None, gray=None)`
- `select_bridges(candidates) -> list[BridgeCandidate]`

Implementation requirements:

- Penalize distance.
- Penalize tangent mismatch.
- Reward smooth continuation.
- Reward faint mask or grayscale evidence between endpoints.
- Reject candidates over max gap.
- Reject candidates over angle threshold.
- Select non-conflicting bridges first.

- [ ] **Step 4: Verify bridge tests**

Run:

```bash
rtk pytest tests/test_centerline_bridge.py tests/test_centerline_graph.py -q
```

Expected: all bridge and graph tests pass.

- [ ] **Step 5: Commit bridge scoring work**

Run:

```bash
rtk proxy git add mathequations/centerline_bridge.py tests/test_centerline_bridge.py
rtk proxy git commit -m "feat: add centerline gap bridging"
```

---

### Task 4: Junction Pairing And Stroke Chain Reconstruction

**Files:**
- Modify: `mathequations/centerline_bridge.py`
- Modify: `mathequations/centerline_graph.py`
- Modify: `tests/test_centerline_bridge.py`
- Modify: `tests/test_centerline_graph.py`

- [ ] **Step 1: Write failing chain reconstruction tests**

Create tests proving:

- a broken straight stroke becomes one `StrokeChain` after bridge acceptance.
- a straight line passing through a junction prefers the straight continuation.
- a real fork remains split when no unambiguous pairing exists.
- final chain count is lower than raw branch count for a deliberately fragmented test image.

- [ ] **Step 2: Run chain tests and confirm failure**

Run:

```bash
rtk pytest tests/test_centerline_graph.py tests/test_centerline_bridge.py -q
```

Expected: fails on missing chain reconstruction or junction pairing behavior.

- [ ] **Step 3: Implement junction pairing and chain reconstruction**

Add public functions:

- `pair_junction_branches(graph, *, angle_threshold_degrees=35)`
- `build_stroke_chains(graph, branches, bridges, junction_pairs, *, min_chain_length=0)`

Implementation requirements:

- Pair incident branches at a junction by tangent continuity.
- Leave ambiguous junctions unpaired and record diagnostics.
- Use accepted endpoint bridges to join raw branches into longer chains.
- Preserve closed loops.
- Drop or mark tiny fragments according to `min_chain_length`.
- Keep chain point ordering deterministic.

- [ ] **Step 4: Verify chain reconstruction tests**

Run:

```bash
rtk pytest tests/test_centerline_graph.py tests/test_centerline_bridge.py -q
```

Expected: all graph, bridge, and chain tests pass.

- [ ] **Step 5: Commit chain reconstruction work**

Run:

```bash
rtk proxy git add mathequations/centerline_graph.py mathequations/centerline_bridge.py tests/test_centerline_graph.py tests/test_centerline_bridge.py
rtk proxy git commit -m "feat: reconstruct centerline stroke chains"
```

---

### Task 5: Chain-Level Curve Fitting

**Files:**
- Modify: `mathequations/curve_fit.py`
- Modify: `mathequations/curve_equations.py`
- Modify: `tests/test_curve_fit.py`
- Modify: `tests/test_curve_equations.py`

- [ ] **Step 1: Write failing chain fitting tests**

Add tests proving:

- a smooth arc chain fits as one cubic Bezier or a small number of cubic Beziers.
- a straight reconstructed chain still fits as linear or vertical.
- a high-curvature chain splits at a real corner.
- hard-to-fit chains fall back to `parametric_polyline` instead of many tiny unrelated line segments.

- [ ] **Step 2: Run fitting tests and confirm failure**

Run:

```bash
rtk pytest tests/test_curve_fit.py tests/test_curve_equations.py -q
```

Expected: fails for missing chain-level fitting and/or `parametric_polyline`.

- [ ] **Step 3: Add parametric polyline record**

Extend `mathequations/curve_equations.py` with a `parametric_polyline` segment type.

Requirements:

- JSON must include `segment_id`, `stroke_id`, `type`, `equation`, `latex`, `source_points`, `restriction`, and `fit_error`.
- Desmos latex should use a parameter range and piecewise or interpolated point form that is accepted by the existing exporter expectations.
- Keep current linear, vertical, quadratic, and cubic Bezier behavior stable.

- [ ] **Step 4: Add chain-level fitter**

Add a public function:

- `fit_stroke_chains(chains, *, target, fit_mode) -> list[dict]`

Implementation requirements:

- Resample chains by arc length before fitting.
- Split by curvature spikes or fit error, not by raw pixel count alone.
- Prefer cubic Bezier for long smooth curves.
- Keep existing `fit_stroke_paths()` available for V1.
- Make target budget approximate, not a reason to fragment smooth curves.

- [ ] **Step 5: Verify fitting tests**

Run:

```bash
rtk pytest tests/test_curve_fit.py tests/test_curve_equations.py -q
```

Expected: all curve tests pass.

- [ ] **Step 6: Commit fitting work**

Run:

```bash
rtk proxy git add mathequations/curve_fit.py mathequations/curve_equations.py tests/test_curve_fit.py tests/test_curve_equations.py
rtk proxy git commit -m "feat: fit reconstructed centerline chains"
```

---

### Task 6: High-Resolution Rendering And Diagnostics

**Files:**
- Modify: `mathequations/curve_render.py`
- Create: `tests/test_centerline_pipeline.py`
- Modify: `tests/test_curve_render.py`

- [ ] **Step 1: Write failing high-resolution rendering tests**

Add tests proving:

- rendering with `render_scale=4` writes `function_preview_highres.png`.
- downsampled `function_preview.png` is nonblank.
- endpoint overlay can mark endpoints and accepted bridges.
- bridged stroke preview renders chain points, not raw tiny fragments.

- [ ] **Step 2: Run rendering tests and confirm failure**

Run:

```bash
rtk pytest tests/test_curve_render.py tests/test_centerline_pipeline.py -q
```

Expected: fails for missing high-resolution rendering or diagnostics helpers.

- [ ] **Step 3: Implement rendering extensions**

Add or extend functions for:

- high-resolution segment rendering.
- downsampled preview writing.
- endpoint and bridge overlay rendering.
- bridged stroke chain preview rendering.

Implementation requirements:

- Keep current `render_curve_segments()` behavior compatible.
- Use anti-aliased OpenCV polylines.
- Do not blur geometry before sampling equations.
- Write diagnostic images only when pipeline requests them.

- [ ] **Step 4: Verify rendering tests**

Run:

```bash
rtk pytest tests/test_curve_render.py tests/test_centerline_pipeline.py -q
```

Expected: rendering tests pass.

- [ ] **Step 5: Commit rendering work**

Run:

```bash
rtk proxy git add mathequations/curve_render.py tests/test_curve_render.py tests/test_centerline_pipeline.py
rtk proxy git commit -m "feat: render centerline diagnostics"
```

---

### Task 7: Centerline V2 Pipeline Integration

**Files:**
- Create: `mathequations/centerline_pipeline.py`
- Modify: `mathequations/lineart_pipeline.py`
- Modify: `mathequations/__main__.py`
- Modify: `tests/test_centerline_pipeline.py`
- Modify: `tests/test_lineart_pipeline.py`

- [ ] **Step 1: Write failing pipeline integration tests**

Add tests proving:

- CLI accepts `--trace-mode centerline-v2`.
- CLI accepts `--preprocess-scale`, `--render-scale`, `--max-bridge-gap`, `--bridge-angle-threshold`, `--local-threshold`, and `--keep-diagnostics`.
- running V2 on a synthetic broken-line drawing writes V1 outputs plus V2 diagnostic outputs.
- `segments.json` metadata includes trace mode, raw branch count, endpoint count, accepted bridge count, final stroke chain count, dropped fragment count, and equation count.
- synthetic broken lines become fewer final chains than raw branches.

- [ ] **Step 2: Run integration tests and confirm failure**

Run:

```bash
rtk pytest tests/test_centerline_pipeline.py tests/test_lineart_pipeline.py -q
```

Expected: fails because V2 pipeline dispatch and CLI flags are missing.

- [ ] **Step 3: Implement `centerline_pipeline.py`**

Create `run_centerline_pipeline()` with parameters matching the CLI flags.

Implementation requirements:

- Write all V1 output files where meaningful.
- Write V2 diagnostic files when `keep_diagnostics=True`.
- Convert final stroke chains to Cartesian coordinates before fitting.
- Use `fit_stroke_chains()` rather than V1 `fit_stroke_paths()`.
- Return a result object with equation count, raw branch count, accepted bridge count, final chain count, preview paths, and JSON path.

- [ ] **Step 4: Add lineart trace-mode dispatch**

Modify `run_lineart_pipeline()` or its caller so:

- `trace_mode="skeleton-v1"` keeps current behavior.
- `trace_mode="centerline-v2"` calls `run_centerline_pipeline()`.
- default can remain `skeleton-v1` until Nan smoke output is verified, or switch to V2 if tests and smoke clearly beat V1.

- [ ] **Step 5: Add CLI flags**

Modify `build_parser()` in `mathequations/__main__.py`.

Required flags:

- `--trace-mode`
- `--preprocess-scale`
- `--render-scale`
- `--max-bridge-gap`
- `--bridge-angle-threshold`
- `--local-threshold`
- `--keep-diagnostics`

- [ ] **Step 6: Verify integration tests**

Run:

```bash
rtk pytest tests/test_centerline_pipeline.py tests/test_lineart_pipeline.py -q
```

Expected: all pipeline integration tests pass.

- [ ] **Step 7: Commit integration work**

Run:

```bash
rtk proxy git add mathequations/centerline_pipeline.py mathequations/lineart_pipeline.py mathequations/__main__.py tests/test_centerline_pipeline.py tests/test_lineart_pipeline.py
rtk proxy git commit -m "feat: add centerline v2 pipeline"
```

---

### Task 8: Real Nan Smoke Test And Parameter Tuning

**Files:**
- Modify only files whose behavior is directly implicated by V2 diagnostics: `mathequations/lineart_preprocess.py`, `mathequations/centerline_bridge.py`, `mathequations/centerline_graph.py`, `mathequations/curve_fit.py`, `mathequations/curve_render.py`, and matching tests.
- Output: `output/nan_lineart_v2_1200/`

- [ ] **Step 1: Run focused Centerline V2 tests**

Run:

```bash
rtk pytest tests/test_centerline_preprocess.py tests/test_centerline_graph.py tests/test_centerline_bridge.py tests/test_centerline_pipeline.py tests/test_curve_fit.py tests/test_curve_render.py tests/test_lineart_pipeline.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run full tests**

Run:

```bash
rtk pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run Nan V2 smoke test**

Run:

```bash
rtk python -m mathequations lineart \
  --input "/Users/arthur/Library/Containers/com.apple.photolibraryd/Data/Pictures/Photos Library.photoslibrary/resources/derivatives/9/935EE685-F7B0-4E11-A9B6-F6B43EE82B03_1_102_o.jpeg" \
  --out output/nan_lineart_v2_1200 \
  --target 1200 \
  --fit-mode mixed \
  --trace-mode centerline-v2 \
  --preprocess-scale 4 \
  --render-scale 4 \
  --max-bridge-gap 16 \
  --bridge-angle-threshold 45 \
  --local-threshold sauvola \
  --keep-diagnostics
```

Expected:

- command exits successfully.
- output includes nonzero equation count.
- output includes nonzero accepted bridge count.
- output includes final chain count lower than raw branch count.
- `function_preview.png` and `function_preview_highres.png` exist.

- [ ] **Step 4: Inspect preview images**

Open or view:

```text
/Users/arthur/mathequations/output/nan_lineart_v2_1200/function_preview.png
/Users/arthur/mathequations/output/nan_lineart_v2_1200/bridged_strokes_preview.png
/Users/arthur/mathequations/output/nan_lineart_v2_1200/endpoint_overlay.png
```

Acceptance bar:

- The character is recognizable.
- Hair, face, hand, dress, and ornament lines are present.
- Lines are materially more continuous than V1.
- Output is not dominated by speckles.
- The preview does not look low-resolution.

- [ ] **Step 5: Inspect metrics**

Run:

```bash
rtk python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("output/nan_lineart_v2_1200/segments.json").read_text())
print(payload["metadata"])
types = {}
for segment in payload["segments"]:
    types[segment["type"]] = types.get(segment["type"], 0) + 1
print(types)
PY
```

Expected:

- metadata includes `trace_mode="centerline-v2"`.
- accepted bridge count is greater than zero.
- final stroke chain count is lower than raw branch count.
- segment types include at least one curve type: `bezier_cubic`, `quadratic`, or `parametric_polyline`.

- [ ] **Step 6: Tune parameters only with evidence**

If preview is still broken, inspect diagnostic files before changing code.

Allowed tuning:

- local threshold parameters.
- max bridge gap.
- bridge angle threshold.
- min chain length.
- component filtering thresholds.
- curve fit error thresholds.

Do not paper over bad reconstruction by simply raising `--target`.

- [ ] **Step 7: Commit verified tuning**

Run:

```bash
rtk proxy git add mathequations tests README.md
rtk proxy git commit -m "fix: tune centerline v2 nan tracing"
```

Only commit this if code, tests, or docs changed during tuning.

---

### Task 9: Documentation And Final Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Document:

- V2 command.
- V1 compatibility mode.
- meaning of `--trace-mode centerline-v2`.
- high-resolution outputs.
- diagnostics outputs.
- warning that higher `--target` does not fix broken stroke reconstruction.

Keep this concise and practical.

- [ ] **Step 2: Run full tests after README change**

Run:

```bash
rtk pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Re-run Nan smoke after docs-only change if code changed after last smoke**

Run the command from Task 8 Step 3 if any code changed after the last successful smoke.

Expected: same acceptance criteria as Task 8.

- [ ] **Step 4: Check git status**

Run:

```bash
rtk proxy git status --short
```

Expected: only intended files are modified or untracked.

- [ ] **Step 5: Commit docs and final cleanup**

Run:

```bash
rtk proxy git add README.md
rtk proxy git commit -m "docs: document centerline v2 pipeline"
```

Only commit README if it changed in this task.

---

## Self-Review Checklist

- Spec coverage:
  - High-resolution preprocessing: Task 1.
  - Local thresholding and speckle filtering: Task 1.
  - Conservative thinning and skeleton graph: Task 2.
  - Endpoint bridge scoring: Task 3.
  - Junction pairing and chain reconstruction: Task 4.
  - Chain-level curve fitting and parametric polyline fallback: Task 5.
  - High-resolution rendering and diagnostics: Task 6.
  - CLI and pipeline dispatch: Task 7.
  - Nan real-image acceptance: Task 8.
  - README documentation: Task 9.
- Type consistency:
  - V2 raw graph records stay separate from V1 `StrokePath`.
  - Public pixel-space points use `(x, y)`.
  - Cartesian conversion happens before equation fitting.
  - V1 `skeleton-v1` behavior remains available while V2 is introduced.
- Scope:
  - No Potrace primary implementation.
  - No PolyVector field implementation.
  - No mandatory AI image cleanup.
  - No blind target-count increase as a quality fix.
