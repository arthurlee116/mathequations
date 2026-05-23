# Koala Thick Lineart Design

## Goal

Convert the provided koala image into real Desmos-usable function expressions that preserve visible stroke thickness by stacking adjacent offset function lines. The output should cover black outline/details, yellow rays, and the pink mouth.

Source image:

```text
/Users/arthur/Library/Containers/com.apple.photolibraryd/Data/Pictures/Photos Library.photoslibrary/originals/A/A6B5711B-1FA5-4827-A183-AAD5DCF15EE5.jpeg
```

## Context

The existing `lineart --trace-mode centerline-v2` path successfully extracts the visible strokes as a mask, but it collapses thick strokes into one-pixel skeleton centerlines before fitting equations. A baseline run on the koala image produced:

```text
output/koala_centerline_baseline/
equation_count: 72
stroke_count: 75
raw_branch_count: 124
accepted_bridge_count: 0
```

The mask is good enough to see the original thick black and yellow marks. The final function preview is too skinny because thickness is not represented as math. Increasing `--target` is the wrong fix; the missing feature is explicit thick-stroke representation.

## Approved Direction

Use stacked offset functions.

The pipeline should keep the current centerline tracing and curve fitting ideas, then estimate stroke width from the source mask and emit neighboring function copies across the stroke normal. Desmos `lineWidth` may still be used as display metadata, but the primary thickness mechanism must be actual expressions.

Approved scope:

- Include black outline and facial details.
- Include yellow rays.
- Include pink mouth.
- Prioritize real Desmos function expressions over a preview-only trick.
- Accept more equations if that is the price of true stacked thickness.

## Architecture

Add a new thick-stroke export path rather than overloading the current `lineart` behavior. The command should be:

```bash
rtk python -m mathequations thick-lineart \
  --input /path/to/source.jpeg \
  --out output/koala_thick \
  --target 1800 \
  --offset-step 1 \
  --max-offsets 9 \
  --keep-diagnostics
```

Pipeline steps:

1. Load and normalize the source image.
2. Extract color-aware masks for black, yellow, and pink.
3. Clean each mask and write diagnostics.
4. Skeletonize each mask into centerlines.
5. Trace centerline chains using the existing centerline graph machinery.
6. Estimate local stroke radius from each original color mask using distance transform.
7. Fit centerline chains into the existing mixed segment types.
8. Expand fitted centerline segments into stacked offset segment copies.
9. Export equations, Desmos latex, expression JSON, `segments.json`, and previews from the offset segments.

## Components

### `mathequations/color_masks.py`

Responsible for extracting named color masks:

- `black`: outline, eyes, nose, mouth border.
- `yellow`: surrounding ray marks.
- `pink`: mouth fill.

Use HSV or Lab thresholding with connected-component cleanup. This image is simple enough that ML would be silly.

### `mathequations/stroke_width.py`

Responsible for stroke width estimation. It should run `cv2.distanceTransform` on a binary color mask and sample radius values at centerline points.

Rules:

- Clamp radius to a configurable range.
- Default minimum radius: `1` pixel.
- Default maximum radius: `10` pixels.
- Store sampled radius diagnostics in metadata.

### `mathequations/offset_segments.py`

Responsible for expanding fitted segments into stacked neighboring expressions.

Behavior:

- Linear segments offset by shifting intercepts along the normal.
- Vertical segments offset by shifting `x = c`.
- Quadratic segments may be sampled and exported as parametric polyline offsets when direct offsetting would lie.
- Cubic Bezier and parametric polyline segments should use sampled normals and export offset polylines.
- Every emitted segment keeps `source_segment_id`, `offset_index`, `offset_distance`, `stroke_radius`, and color metadata.

### `mathequations/thick_lineart_pipeline.py`

Responsible for orchestration and artifact writing. It should follow existing pipeline patterns: dataclass result, `_write_text`, JSON payload, previews, and CLI dispatch.

Expected artifacts:

```text
clean_input.png
black_mask.png
yellow_mask.png
pink_mask.png
black_skeleton.png
yellow_skeleton.png
pink_skeleton.png
stroke_width_diagnostics.json
stacked_preview.png
function_preview.png
function_preview_highres.png
equations.txt
desmos_latex.txt
desmos_expressions.json
desmos.html
segments.json
selected_equations.txt
```

## Data Flow

For each color:

```text
source image
-> color mask
-> cleaned mask
-> skeleton
-> stroke chains
-> distance-map radius samples
-> fitted centerline segments
-> stacked offset segment copies
-> Desmos expressions and previews
```

Segment metadata should look like:

```json
{
  "segment_id": 128,
  "source_segment_id": 17,
  "stroke_id": 4,
  "color_name": "black",
  "color": "#000000",
  "offset_index": -3,
  "offset_distance": -0.048,
  "stroke_radius": 0.112,
  "type": "linear"
}
```

This makes failures debuggable. If the mouth is wrong, we can isolate whether the problem is masking, tracing, width estimation, fitting, or offsetting.

## Error Handling

Black is required. If the black mask is missing or produces no strokes, fail with a clear `ValueError`.

Yellow and pink are optional. If either mask is missing or produces no strokes, continue and record a warning in metadata.

Width estimates must be clamped to avoid one noisy pixel producing absurd stacks. If offsetting a segment as its original type would be inaccurate, degrade to `parametric_polyline`.

## Testing

Add focused synthetic tests first:

- Color mask extraction separates black, yellow, and pink strokes.
- Distance transform returns larger radius on thicker synthetic strokes.
- Linear offset emits parallel lines with correct metadata.
- Vertical offset shifts `x = c` correctly.
- Curved offset can fall back to parametric polyline with color and source metadata.
- Pipeline writes all expected artifacts.
- Desmos export includes colored stacked expressions.

Then run the real koala image as a diagnostic smoke run. The smoke run should verify that masks are nonblank and that output expression count is substantially higher than the 72-expression skinny baseline.

Verification commands:

```bash
rtk pytest tests/test_color_masks.py tests/test_stroke_width.py tests/test_offset_segments.py tests/test_thick_lineart_pipeline.py -q
rtk pytest -q
rtk python -m mathequations thick-lineart --input '/Users/arthur/Library/Containers/com.apple.photolibraryd/Data/Pictures/Photos Library.photoslibrary/originals/A/A6B5711B-1FA5-4827-A183-AAD5DCF15EE5.jpeg' --out output/koala_thick --target 1800 --offset-step 1 --max-offsets 9 --keep-diagnostics
```

## Acceptance Criteria

- The new CLI subcommand runs on the koala source image.
- `segments.json` contains black, yellow, and pink offset segments.
- Thickness comes from multiple exported expressions, not only Desmos `lineWidth`.
- Preview output is visibly thicker than the centerline baseline.
- The resulting Desmos expressions are colored.
- Existing lineart and centerline tests still pass.

## Out Of Scope

- General-purpose full-color image vectorization.
- Pixel-perfect reproduction.
- Manual tracing.
- Replacing the existing `lineart` command.
- ML-based segmentation.

## Spec Self-Review

- Placeholder scan: no placeholder requirements remain.
- Consistency check: architecture, components, data flow, and acceptance criteria all describe the same `thick-lineart` path.
- Scope check: this is one bounded pipeline extension, not multiple independent subsystems.
- Ambiguity check: black is required, yellow and pink are optional-but-attempted, and real stacked expressions are the primary thickness mechanism.
