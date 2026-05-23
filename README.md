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

This repo currently has no package metadata file, so install the few runtime
pieces directly:

```bash
python -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy pytest
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

## Tests

```bash
pytest -q
```

The tests use synthetic images, so they do not depend on private source artwork.
