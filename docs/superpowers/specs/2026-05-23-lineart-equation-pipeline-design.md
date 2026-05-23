# Lineart Equation Pipeline Design

## Summary

Build a new `lineart` pipeline for converting clean black-and-white or pencil-style line art into Desmos-compatible equations. The first target is the supplied Nan line sketch: a delicate anime-style drawing with many open strokes, pale pencil lines, hair arcs, eyes, fingers, costume ornaments, and skirt curves.

The correct goal is not to fill regions. It is to recover the drawing as strokes, then express those strokes as a controlled mix of linear equations, vertical lines, polynomial curves, and parametric curves.

## Goals

- Convert line art into equation-based strokes that visually resemble the source drawing.
- Preserve the feel of hand-drawn linework: long hair curves, eye curves, fingers, dress folds, and ornamental curls should remain graceful instead of becoming jagged polygon chains.
- Support parameter equations as a first-class output format.
- Keep the process modular enough to tune thresholding, stroke count, simplification, and curve fitting.
- Produce preview images before Desmos export so the result can be judged visually.
- Reuse existing coordinate mapping, output conventions, tests, and Desmos export patterns where they still fit.

## Non-Goals

- Do not add color fill for this version.
- Do not try to identify semantic body parts by name.
- Do not require GPT Image generation to make the pipeline work.
- Do not chase pixel-perfect reproduction at the expense of readable math.
- Do not force every curve into `y = f(x)`. That would be a bad bargain for hair, eyes, fingers, and loop-like ornament strokes.

## Recommended Approach

Create a new `lineart` pipeline rather than modifying the generic emblem pipeline or Lucas color pipeline.

The existing generic pipeline treats foreground pixels as closed contours. That works for solid emblems, but line art becomes the outline of each pencil stroke, which produces heavy doubled shapes. The Lucas color pipeline is useful as a reference for previews, JSON exports, and Desmos files, but its color-region logic is unnecessary here.

The new pipeline should treat the image as a collection of open strokes:

```text
image
-> grayscale cleanup
-> line mask
-> skeleton
-> stroke graph
-> open stroke paths
-> curve segmentation
-> equation fitting
-> preview
-> Desmos and JSON export
```

## CLI

Add a subcommand:

```bash
rtk python -m mathequations lineart \
  --input /path/to/image.jpeg \
  --out output/nan_lineart \
  --target 1200 \
  --fit-mode mixed
```

Initial options:

- `--input`: source image path.
- `--out`: output directory.
- `--target`: approximate equation or curve segment budget.
- `--scale-width`: Cartesian width for the drawing bounding box, default `20.0`.
- `--threshold-mode`: one of `auto`, `fixed`, or `adaptive`, default `auto`.
- `--line-thickness`: preview line thickness, default `1`.
- `--fit-mode`: one of `linear`, `quadratic`, or `mixed`, default `mixed`.
- `--cleaned-input`: optional pre-cleaned image path, used when GPT Image or another tool has produced a clearer line version.

Do not make GPT Image cleanup the default. If used in an explicit cleanup run, it should be a preprocessing step that writes a separate image so the pipeline remains auditable.

## Output Contract

Write:

```text
clean_input.png
line_mask.png
skeleton.png
stroke_preview.png
function_preview.png
equations.txt
desmos_latex.txt
desmos_expressions.json
segments.json
selected_equations.txt
```

`segments.json` should include:

- metadata: image width, image height, scale, target, fit mode, equation count, stroke count.
- strokes: one record per traced stroke, with stroke ID, point count, length, bounding box, and optional quality metrics.
- segments: one record per equation segment.

Each segment should include:

- `segment_id`
- `stroke_id`
- `type`
- `equation`
- `latex`
- `start`
- `end`
- `restriction`
- `source_points`
- fitting parameters
- fit error

## Equation Types

Support these types in the first serious version:

- `linear`: `y = mx + b {x_min <= x <= x_max}`
- `vertical`: `x = c {y_min <= y <= y_max}`
- `quadratic`: `y = ax^2 + bx + c {x_min <= x <= x_max}`
- `parametric_polyline`: `(x(t), y(t))` over a restricted `t` range, used as a safe fallback for hard strokes.
- `bezier_cubic`: cubic Bezier parameter equations:

```text
x(t) = (1-t)^3 x0 + 3(1-t)^2 t x1 + 3(1-t)t^2 x2 + t^3 x3
y(t) = (1-t)^3 y0 + 3(1-t)^2 t y1 + 3(1-t)t^2 y2 + t^3 y3
0 <= t <= 1
```

Excluded from the first implementation:

- `sine`: only for actual wave-like strokes such as signatures or repeated ornament lines.
- `ellipse_arc`: useful for eyes or circular ornaments if the fitting quality beats Bezier.

## Preprocessing

Line-art preprocessing should handle pale pencil marks without turning paper texture into noise.

Steps:

1. Load the image as BGR and grayscale.
2. Normalize near-white background.
3. Use contrast enhancement such as CLAHE for faint lines.
4. Generate candidate masks with fixed and adaptive thresholding.
5. Select or combine masks based on foreground density and connected-component sanity checks.
6. Remove isolated speckles while preserving thin strokes.
7. Save `line_mask.png`.

This stage should expose tunable parameters, because line sketches vary wildly. The Nan sketch has useful line pixels even near very high grayscale values, so a strict dark-line threshold drops too much detail.

## Skeleton And Stroke Tracing

The line mask should be reduced to a centerline skeleton, then converted into graph paths.

Preferred implementation:

- Use `scikit-image` for `skeletonize` if available.
- Add `scikit-image` as a dependency if needed.
- Represent skeleton pixels as graph nodes with 8-neighbor connectivity.
- Classify nodes by degree:
  - degree 1: endpoints
  - degree 2: normal path nodes
  - degree 3 or more: junctions
- Trace paths between endpoints and junctions.
- Also trace closed loops with no endpoints.
- Merge very short dangling fragments only when they are below a small length threshold.

The output should be open stroke paths, not closed filled contours.

## Curve Segmentation

Each traced stroke can be too long or too bendy for one equation. Split strokes into manageable pieces.

Use these signals:

- maximum arc length per segment
- curvature spikes
- junction boundaries
- fit error
- target equation budget

The budget allocator should prioritize visually important strokes:

- longer strokes get more budget
- high-curvature strokes get more budget
- tiny noise fragments get little or no budget

## Curve Fitting

Fit candidate functions for each stroke piece:

1. Try linear or vertical if the segment is nearly straight.
2. Try quadratic when `x` is monotonic and the residual is low.
3. Try cubic Bezier for smooth non-monotonic or high-value visual curves.
4. Fall back to short linear or parametric polyline segments when fitting would distort the drawing.

Use a simple scoring rule:

```text
score = fit_error + complexity_penalty + continuity_penalty
```

Select the simplest curve that stays below the error threshold. This matters: a beautiful school project should not use complicated functions where a line does the job.

## Rendering

Extend preview rendering beyond straight segments.

For each segment:

- sample its equation into points
- map Cartesian points back to image pixels
- draw anti-aliased polylines on a white canvas

Render two previews:

- `stroke_preview.png`: traced skeleton strokes before fitting.
- `function_preview.png`: final equation output.

The preview should be the main quality gate. If it does not look like the drawing, the export is not done.

## Desmos Export

Write:

- `equations.txt`: readable equations.
- `desmos_latex.txt`: one LaTeX expression per line.
- `desmos_expressions.json`: Desmos API expression objects.
- `selected_equations.txt`: a short curated list for school explanation.

For Bezier and parametric curves, use Desmos-compatible parametric notation. If a single expression format becomes awkward, store both `x_latex` and `y_latex` in JSON while emitting a tested Desmos text form in `desmos_latex.txt`.

## GPT Image Cleanup

GPT Image may be useful as an optional assistant, not as the source of truth.

Acceptable use:

- Generate a cleaner black-line version from the source sketch.
- Preserve pose, expression, outfit, object, and composition.
- Use the cleaned image as `--cleaned-input`.
- Save both original and cleaned inputs in the output folder for comparison.

Risk:

- It may redraw or simplify the character.
- It may change details, which weakens the "automated image-to-equation" proof.

Recommendation: first implement deterministic cleanup. Use GPT Image only if the source sketch is too faint for reliable extraction.

## Testing Strategy

Add focused tests with synthetic images so private artwork is not required.

Tests should cover:

- thresholding keeps faint gray strokes on white background.
- skeleton extraction turns thick lines into thin centerlines.
- graph tracing returns open paths for line segments and loops for circles.
- line fitting emits linear and vertical equations.
- quadratic fitting beats linear fitting on a parabolic stroke.
- Bezier or parametric fitting handles non-monotonic curves.
- renderer can preview mixed equation types.
- CLI writes all output files.

Run full verification with:

```bash
rtk pytest -q
```

Then run a real-image smoke test on the Nan image and inspect `function_preview.png`.

## Implementation Boundaries

New modules:

- `mathequations/lineart_preprocess.py`
- `mathequations/skeleton_graph.py`
- `mathequations/curve_fit.py`
- `mathequations/curve_equations.py`
- `mathequations/lineart_pipeline.py`

Modify:

- `mathequations/__main__.py` to add the `lineart` command.
- `mathequations/render.py` or a new mixed renderer to support curve sampling.
- `README.md` after the implementation exists.

Do not rewrite the existing generic, Lucas raster, or Lucas vector pipelines. Reuse their helpers where practical and leave their behavior stable.

## Success Criteria

The first accepted prototype should:

- produce recognizable Nan line art from the supplied sketch.
- avoid doubled pencil-stroke outlines.
- include meaningful hair, face, hand, dress, and ornament strokes.
- export mixed Desmos-compatible equations.
- render a preview that is recognizably close to the source.
- keep equation count controllable around 800, 1200, and 3000.
- include a selected equation list that a student can explain.
