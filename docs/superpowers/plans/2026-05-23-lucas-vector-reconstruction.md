# Lucas Vector Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean Lucas SVG/vector reconstruction before exporting Desmos equations.

**Architecture:** Add a small vector artwork model that stores semantic shapes as filled/stroked paths in source-image coordinates. Build a Lucas-specific extractor that creates design layers from foreground and color masks, exports SVG/PNG previews, and derives Desmos line expressions only from those clean paths.

**Tech Stack:** Python, OpenCV, NumPy, existing `mathequations` package and unittest/pytest.

---

### Task 1: Vector Artwork Model

**Files:**
- Create: `mathequations/vector_model.py`
- Test: `tests/test_vector_model.py`

- [ ] Write tests for SVG path generation and shape storage.
- [ ] Run `rtk pytest tests/test_vector_model.py -q` and confirm the module is missing.
- [ ] Implement `VectorShape`, `VectorArtwork`, and SVG path serialization.
- [ ] Re-run the test and confirm it passes.

### Task 2: SVG Export And Raster Preview

**Files:**
- Create: `mathequations/vector_render.py`
- Test: `tests/test_vector_render.py`

- [ ] Write tests that render one filled square and one stroked open line.
- [ ] Run `rtk pytest tests/test_vector_render.py -q` and confirm the renderer is missing.
- [ ] Implement SVG writing and OpenCV preview rendering from vector shapes.
- [ ] Re-run the test and confirm it passes.

### Task 3: Lucas Semantic Vector Pipeline

**Files:**
- Create: `mathequations/lucas_vector_pipeline.py`
- Modify: `mathequations/__main__.py`
- Test: `tests/test_lucas_vector_pipeline.py`

- [ ] Write tests for semantic layer extraction on a synthetic Lucas-like image.
- [ ] Run the test and confirm the pipeline module or function is missing.
- [ ] Implement semantic masks for silhouette, green body, purple eyes/ears, yellow accents, white details, black internal features, and metal-gray details.
- [ ] Export `lucas_clean.svg`, `lucas_vector_preview.png`, `lucas_vector_segments.json`, and Desmos expression files.
- [ ] Add a `lucas-vector` CLI subcommand.
- [ ] Re-run focused and full tests.

### Task 4: Real Lucas Verification

**Files:**
- Output: `output/lucas_vector/`

- [ ] Run `rtk python -m mathequations lucas-vector --input /Users/arthur/Downloads/lucas.png --out output/lucas_vector`.
- [ ] Inspect `lucas_vector_preview.png` and `lucas_clean.svg`.
- [ ] Confirm the new Desmos paths come from semantic vector shapes, not gray quantization boundaries.
