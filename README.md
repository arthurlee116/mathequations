# mathequations

Turn clean image artwork into coordinate-plane math art: contours become short,
restricted linear equations, with JSON and Desmos-friendly exports.

The core idea is deliberately simple. An input image is reduced to foreground
masks, masks become contours, contours are simplified or resampled to a target
point budget, and each neighboring pair of points becomes either:

```text
y = mx + b {x_min <= x <= x_max}
```

or, for vertical segments:

```text
x = c {y_min <= y <= y_max}
```

That gives a pile of bounded line equations that can recreate the original
shape in a graphing tool.

## What Is In Here

- `mathequations/pipeline.py` is the generic image-to-equations pipeline.
- `mathequations/equations.py` formats contour segments as restricted linear equations.
- `mathequations/geometry.py` handles contour simplification, resampling, length-based point allocation, and pixel-to-Cartesian mapping.
- `mathequations/image_processing.py` isolates foreground pixels and loads/normalizes images.
- `mathequations/render.py` renders previews from generated segments.
- `mathequations/lucas_pipeline.py` is the image-specific Lucas equation pipeline.
- `mathequations/lucas_vector_pipeline.py` rebuilds Lucas as semantic vector shapes first, then exports those shapes as equations.
- `tests/` covers geometry, equation formatting, generic conversion, Lucas extraction, and vector export behavior.

Generated files go under `output/` and are intentionally ignored by Git.

## Setup

Install the runtime and test dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generic Pipeline

Use this when the source image is already a clean emblem or simple colored
foreground on a mostly white background.

```bash
python -m mathequations \
  --input path/to/image.png \
  --target 200 \
  --out output/run_200
```

Outputs include:

- `mask.png`: extracted foreground mask
- `preview.png`: rendered preview of the generated equations
- `equations.txt`: plain restricted linear equations
- `selected_equations.txt`: small sample for quick inspection
- `segments.json`: structured segment metadata

`--target` is approximate. The code allocates points by contour length, then
simplifies or resamples so larger contours get more equations.

## Lucas Pipeline

Lucas needed special treatment because the plain contour pipeline is too blunt:
it can trace a silhouette, but it does not understand character layers, facial
features, fill regions, or the difference between an outline and an interior
detail. The Lucas path adds that missing structure.

```bash
python -m mathequations lucas \
  --input path/to/lucas.png \
  --out output/lucas_1500 \
  --outline-target 1500 \
  --region-target 1200 \
  --gray-levels 6
```

What it does differently:

- normalizes the white background before processing
- extracts a main foreground silhouette from the character
- separates dark line/detail masks from filled regions
- detects larger color/feature regions, including purple, yellow, and dark facial/detail areas
- quantizes the character into grayscale fill regions
- gives outline/detail contours and fill-boundary contours separate equation budgets
- writes Desmos exports: `desmos_expressions.json`, `desmos_latex.txt`, `desmos_equations.txt`, and `desmos_lucas.html`
- renders both `outline_preview.png` and `filled_preview.png`

The important bit: Lucas is not treated as one anonymous blob. The pipeline
knows enough about the character image to preserve recognizable structure.

## Lucas Vector Pipeline

There is also a cleaner Lucas route that reconstructs semantic vector artwork
before converting it to equations:

```bash
python -m mathequations lucas-vector \
  --input path/to/lucas.png \
  --out output/lucas_vector
```

This path builds named shapes such as:

- `silhouette`
- `green_body`
- `dark_blue_clothing`
- `purple_eye_regions`
- `black_eye_lids`
- `purple_regions`
- `metal_gray`
- `yellow_accents`
- `white_details`
- `black_details`

It writes:

- `lucas_clean.svg`
- `lucas_vector_preview.png`
- `lucas_function_render.png`
- `lucas_vector_segments.json`
- `desmos_vector_expressions.json`
- `desmos_vector_latex.txt`
- `desmos_vector_equations.txt`

This is the better path when the goal is a readable, layered Lucas asset rather
than a raw high-equation trace. It is less "scan every contour and pray" and
more "recover the character as shapes, then graph the shapes."

## Lineart Pipeline

Use this for pencil sketches, anime line art, and other drawings where the visible object is made of strokes rather than filled regions.

```bash
python -m mathequations lineart \
  --input path/to/sketch.jpeg \
  --out output/nan_lineart_1200 \
  --target 1200 \
  --fit-mode mixed
```

The default `--trace-mode skeleton-v1` path extracts faint lines, skeletonizes them into stroke centerlines, traces open paths, fits a mix of straight, quadratic, and parametric cubic Bezier equations, and writes:

- `line_mask.png`
- `skeleton.png`
- `stroke_preview.png`
- `function_preview.png`
- `equations.txt`
- `desmos_latex.txt`
- `desmos_expressions.json`
- `segments.json`
- `selected_equations.txt`

Parameter equations are intentional here. They preserve hair, eyes, fingers, and ornament curves much better than forcing every stroke into `y = f(x)`.

### Centerline V2

Use Centerline V2 when V1 turns pencil strokes into broken dashes:

```bash
python -m mathequations lineart \
  --input path/to/sketch.jpeg \
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

V2 processes the image at higher resolution, uses local thresholding, builds a skeleton graph, scores endpoint bridges, pairs likely junction continuations, reconstructs longer stroke chains, and renders supersampled previews.

Extra V2 outputs include:

- `clean_input_highres.png`
- `line_mask_highres.png`
- `skeleton_highres.png`
- `endpoint_overlay.png`
- `bridge_candidates.json`
- `bridged_strokes_preview.png`
- `function_preview_highres.png`
- `trace_diagnostics.json`

`segments.json` metadata records `trace_mode`, preprocess/render scales, raw branch count, endpoint count, accepted bridge count, final chain count, dropped fragment count, and equation count.

Higher `--target` does not fix broken stroke reconstruction. If output is dashed, inspect the V2 diagnostics and tune thresholding or bridge parameters.

## Thick Lineart Pipeline

Use this for koala-style colored line art where stroke thickness must be represented by real Desmos expressions instead of only display line width.

```bash
python -m mathequations thick-lineart \
  --input path/to/koala.jpeg \
  --out output/koala_thick \
  --target 1800 \
  --offset-step 1 \
  --max-offsets 9 \
  --keep-diagnostics
```

This path extracts required black, yellow, and pink masks, traces centerlines, estimates stroke radius from each mask, and emits stacked offset expressions. `--offset-step` is measured in source-image pixels, then converted into Cartesian distance with the export scale. `--max-offsets` is the total odd stack count per source segment, including the centerline `offset_index=0`.

Outputs include:

- `black_mask.png`, `yellow_mask.png`, `pink_mask.png`
- `black_skeleton.png`, `yellow_skeleton.png`, `pink_skeleton.png`
- `stroke_width_diagnostics.json`
- `function_preview.png`
- `function_preview_highres.png`
- `stacked_preview.png`
- `segments.json`
- `equations.txt`
- `desmos_latex.txt`
- `desmos_expressions.json`
- `desmos.html`

## Tests

```bash
pytest -q
```

The tests use synthetic images, so they do not depend on private source artwork.
