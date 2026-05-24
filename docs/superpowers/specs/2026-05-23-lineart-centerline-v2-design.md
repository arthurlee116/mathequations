# Lineart Centerline V2 Design

## Summary

The first `lineart` pipeline proved the wrong thing: basic thresholding plus skeleton tracing can produce a recognizable silhouette, but it cannot produce high-quality continuous line art. The output breaks into short fragments because the source sketch contains faint, interrupted pencil strokes, and the current graph tracer treats every physical gap as a real stroke break.

Centerline V2 replaces that brittle path with a stroke reconstruction pipeline. It should recover long centerline strokes, bridge small gaps when the geometry says they belong together, preserve junctions, fit smooth parametric curves, and render high-resolution previews suitable for judging actual quality.

This is the chosen next direction. External tracing tools can be useful as a baseline, but they are not the main architecture.

## Problem With V1

The current implementation does this:

```text
image -> cleanup -> line mask -> skeleton -> graph paths -> curve fitting
```

That is too naive for pencil line art.

Observed Nan output:

- `815` traced strokes from one character sketch.
- Most strokes are tiny fragments.
- Hair, face, hand, dress, and ornaments are visible only as broken dashes.
- Increasing `--target` cannot fix this because the breakage happens before fitting.
- The preview uses source-size rendering and one-pixel lines, so it looks low-resolution even when the equation data exists.

The root flaw is treating skeleton connectivity as truth. In a pencil sketch, gaps are often scanner/threshold artifacts, not real drawing boundaries.

## Goals

- Produce continuous single-line paths, not dashed skeleton fragments.
- Preserve the centerline nature of strokes, avoiding doubled outline traces.
- Improve preview resolution with supersampled rendering.
- Use local deterministic processing as the default path.
- Export Desmos-compatible linear, vertical, quadratic, cubic Bezier, and fallback parametric polyline equations.
- Keep equation count controllable around `800`, `1200`, and `3000`.
- Emit diagnostics that make failures obvious: mask preview, skeleton preview, endpoint overlay, bridged-stroke preview, final curve preview.

## Non-Goals

- Do not build a full academic PolyVector field implementation in this pass.
- Do not depend on GPT Image or any AI image-editing step for correctness.
- Do not use Potrace as the primary pipeline; it traces filled regions and tends to create outline paths rather than stroke centerlines.
- Do not promise perfect semantic understanding of clothing, hair, face, or ornaments.
- Do not hide bad output behind higher equation counts.

## Chosen Architecture

Use a Centerline V2 pipeline:

```text
image
-> high-resolution normalization
-> local line mask extraction
-> conservative thinning
-> skeleton graph
-> endpoint and junction analysis
-> gap bridging
-> stroke chain reconstruction
-> spline / Bezier fitting
-> high-resolution rendering
-> Desmos export
```

The key difference from V1 is the middle:

```text
skeleton graph -> endpoint/junction reasoning -> reconstructed stroke chains
```

V1 traces whatever connected pixels exist. V2 should infer when nearby fragments are part of the same drawn stroke.

## Stage 1: High-Resolution Preprocessing

Input should be processed at `4x` scale by default.

Steps:

1. Load original image.
2. Snap near-white paper background toward white.
3. Upscale with Lanczos or cubic interpolation.
4. Convert to grayscale.
5. Apply contrast normalization, preferably CLAHE.
6. Apply edge-preserving denoise such as bilateral filtering or non-local means.
7. Save `clean_input_highres.png`.

Why this matters:

- Thin pencil lines become several pixels wide before thresholding.
- Local thresholding has more signal to work with.
- Skeletonization becomes less likely to destroy faint one-pixel marks.
- Preview rendering can be downsampled cleanly for anti-aliasing.

## Stage 2: Local Line Mask Extraction

Use local thresholding instead of one global cutoff.

Preferred implementation:

- If available, use `opencv-contrib-python` and `cv2.ximgproc.niBlackThreshold`.
- Support Sauvola, Wolf, or Niblack-style threshold modes when the OpenCV build exposes them.
- Keep a scikit-image fallback using `skimage.filters.threshold_sauvola`.
- Keep V1 fixed/adaptive modes as debug fallbacks, not the default.

The default V2 mode should be:

```text
local threshold -> component filtering -> optional close-small-gaps -> mask
```

Important: avoid destructive opening. The V1 3x3 open erased valid one-pixel pencil lines. V2 should remove isolated dots by connected-component statistics, not by shaving all thin strokes.

Output:

- `line_mask_highres.png`
- `line_mask.png`
- `mask_diagnostics.json`

Diagnostics should include foreground density, component count, median component size, and removed speckle count.

## Stage 3: Conservative Thinning

Convert mask to centerlines after high-resolution cleanup.

Options:

- Prefer OpenCV `ximgproc.thinning` when `opencv-contrib-python` is installed.
- Fall back to `skimage.morphology.skeletonize`.

The thinning output is still not final stroke data. It is only evidence for graph reconstruction.

Output:

- `skeleton_highres.png`
- `skeleton.png`

## Stage 4: Skeleton Graph With Geometry

Represent skeleton pixels as a graph with geometric annotations.

For each node/path:

- pixel coordinates
- degree
- endpoint / junction / regular classification
- local tangent estimate
- local stroke width estimate from distance transform
- nearby mask confidence

Trace raw graph branches between endpoints and junctions, but do not export them yet.

V2 should introduce a separate data model:

```python
RawBranch
Endpoint
Junction
BridgeCandidate
StrokeChain
```

This prevents the code from confusing "raw skeleton fragments" with "final strokes."

## Stage 5: Endpoint Gap Bridging

This is the heart of V2.

Find candidate pairs of endpoints and score them:

```text
score =
  distance_cost
  + tangent_mismatch_cost
  + curvature_cost
  - mask_evidence_bonus
  - continuation_bonus
```

Candidate rules:

- Max gap defaults to `8-20` high-res pixels, configurable.
- Endpoint tangents must roughly face each other or form a smooth continuation.
- A faint grayscale or mask trail between endpoints should increase confidence.
- Do not bridge across unrelated nearby details such as eye/face/ornament intersections.
- Avoid many-to-one bridges unless a junction rule explicitly allows it.

Bridge selection:

- Build candidate list.
- Sort by score.
- Greedily accept non-conflicting bridges below threshold.
- Later, replace with min-cost matching if greedy causes visible mistakes.

Outputs:

- `endpoint_overlay.png`
- `bridge_candidates.json`
- `bridged_strokes_preview.png`

This directly attacks the broken-line problem.

## Stage 6: Junction Pairing

At junctions, V1 cuts everything. V2 should pair incident branches when the tangent says they continue through the junction.

For each junction:

1. Estimate tangent for each incident branch near the junction.
2. Pair branches with the straightest continuation.
3. Preserve true forks when three or more branches genuinely meet.
4. Mark ambiguous junctions in diagnostics instead of forcing a bad connection.

This matters for hair, clothing folds, fingers, and ornaments, where lines touch or nearly touch.

## Stage 7: Stroke Chain Reconstruction

Build final `StrokeChain` objects from:

- raw branches
- accepted endpoint bridges
- accepted junction pairings
- closed loops

Each chain should be ordered, continuous, and long enough to fit as one or more smooth curves.

Small fragments:

- Drop isolated fragments below a configurable pixel length unless they have high darkness/confidence.
- Keep tiny but important facial details if they pass a contrast/confidence threshold.

The output should have far fewer and longer strokes than V1. A useful target for the Nan sketch is not `815` tiny strokes; it should be closer to dozens or low hundreds of meaningful chains.

## Stage 8: Spline And Bezier Fitting

V1 fits each small piece independently. V2 should fit whole reconstructed chains, then split only when needed.

Process:

1. Resample each chain by arc length.
2. Smooth lightly with a curvature-preserving filter.
3. Detect corners and curvature spikes.
4. Split at strong corners, real junctions, or high fit error.
5. Fit candidate equations:
   - linear
   - vertical
   - quadratic
   - cubic Bezier
   - parametric polyline fallback
6. Prefer cubic Bezier for long graceful curves.
7. Prefer linear segments for genuinely straight strokes.

Scoring:

```text
score = fit_error + complexity_penalty + discontinuity_penalty
```

The preview should optimize visual continuity first, equation cleverness second. For Desmos, a continuous cubic Bezier is much better than ten tiny broken line segments.

## Stage 9: High-Resolution Rendering

Render final output at higher resolution than the source canvas.

Defaults:

- `--render-scale 4`
- draw at high-res
- downsample with area interpolation for preview
- optionally keep `function_preview_highres.png`

Outputs:

- `stroke_preview.png`
- `bridged_strokes_preview.png`
- `function_preview.png`
- `function_preview_highres.png`

This fixes the "low resolution as hell" problem. The math may still be approximate, but the inspection image should not be pixel-starved.

## External Tool Baseline

Add an optional baseline mode later, not first:

```bash
autotrace input.png -centerline -output-file output.svg
```

AutoTrace is useful because it supports centerline tracing. It can give a fast comparison target.

Potrace is less suitable as the main route because it traces black bitmap regions into smooth vector outlines. That is excellent for logos and filled shapes, but line art often becomes doubled outline paths rather than centerline strokes.

Baseline output can be:

- `autotrace_centerline.svg`
- `autotrace_preview.png`
- `autotrace_segments.json`

Do not block Centerline V2 on external tools being installed.

## Optional AI Image Cleanup

The user approved using image tools if helpful. For this design, AI image cleanup is optional and diagnostic, not required.

Allowed use:

- Create a cleaner black-line version of the same sketch.
- Preserve pose, character, object, signature, outfit, and composition.
- Save it separately as `ai_cleaned_input.png`.
- Run Centerline V2 on both original and cleaned image for comparison.

Rejected use:

- Do not silently redraw missing details.
- Do not make AI cleanup the only path that works.
- Do not use AI cleanup as proof that the deterministic pipeline succeeded.

## CLI Additions

Keep the existing command and add V2 options:

```bash
rtk python -m mathequations lineart \
  --input path/to/sketch.jpeg \
  --out output/nan_lineart_v2 \
  --target 1200 \
  --fit-mode mixed \
  --trace-mode centerline-v2 \
  --preprocess-scale 4 \
  --render-scale 4 \
  --max-bridge-gap 16
```

New options:

- `--trace-mode`: `skeleton-v1`, `centerline-v2`, later `autotrace-baseline`.
- `--preprocess-scale`: default `4`.
- `--render-scale`: default `4`.
- `--max-bridge-gap`: default around `16` high-res pixels.
- `--bridge-angle-threshold`: default around `45` degrees.
- `--local-threshold`: `sauvola`, `niblack`, `adaptive`, `fixed`.
- `--keep-diagnostics`: write overlays and JSON diagnostics.

Default should become `centerline-v2` once it beats V1 on the Nan smoke test.

## Output Contract

V2 should write everything V1 writes plus:

```text
clean_input_highres.png
line_mask_highres.png
skeleton_highres.png
endpoint_overlay.png
bridge_candidates.json
bridged_strokes_preview.png
function_preview_highres.png
trace_diagnostics.json
```

`segments.json` metadata should include:

- trace mode
- preprocess scale
- render scale
- raw branch count
- endpoint count
- accepted bridge count
- final stroke chain count
- dropped fragment count
- equation count

## Testing Strategy

Add synthetic tests that prove the new behavior, not just that files exist.

Required tests:

- Local threshold keeps faint one-pixel and two-pixel strokes.
- High-res preprocessing preserves a pale gray line that V1 would fragment.
- Endpoint bridge reconnects two aligned broken line segments.
- Endpoint bridge refuses two nearby but opposing unrelated segments.
- Junction pairing continues a straight line through a T-like or crossing-like junction when appropriate.
- Stroke chain reconstruction reduces raw fragment count.
- Bezier fitting emits one smooth segment for a known arc.
- High-res renderer writes a larger preview and a downsampled preview.
- Nan smoke test metrics improve: fewer tiny strokes and visually continuous output.

Metric targets for Nan should start conservative:

- final stroke chains less than raw skeleton branches.
- median chain length at least `3x` V1 median branch length.
- accepted bridges greater than zero.
- final preview includes recognizable face, hand, hair, dress, and ornament lines.

## Risks

- Over-bridging can connect unrelated facial or ornament details. Diagnostics and angle/distance thresholds are mandatory.
- Under-bridging leaves dashes. The acceptance test must inspect both metrics and images.
- Local thresholding can turn paper texture into noise. Component filtering and density metrics are required.
- OpenCV contrib may not be installed. Keep a scikit-image fallback.
- Desmos can get slow with too many expressions. Target budgets still matter.

## Recommendation

Build Centerline V2 first. Add AutoTrace centerline as a later baseline if needed. Do not make Potrace the primary path for this use case, and do not jump to PolyVector research implementation yet.

This gives the best balance:

- much better continuity than V1
- still local and auditable
- still produces centerline equations
- practical enough to implement inside the current repo
