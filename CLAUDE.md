# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run tests:**
```bash
python tests/test_calc.py
python tests/test_synth.py
python tests/test_alignment.py
```

**Run the interactive UI:**
```bash
jupyter nbconvert --to script Calc.ipynb --stdout | python
```

**Dependencies:** `opencv-python`, `numpy` (no requirements.txt — install manually).

## Architecture

**All source code lives in a single Jupyter notebook cell** — `Calc.ipynb`, cell index 0. There are no `.py` source files. Tests load this code by parsing the notebook JSON and `exec()`-ing the source string directly.

### Pipeline (in order)

1. `_register_to` — aligns a shot photo to the baseline using the aiming mark's 4 extreme points (`estimateAffinePartial2D` similarity transform). Falls back to pure translation if the mark can't be found.
2. `_to_gray_norm` — CLAHE normalisation (LAB L-channel) + bilateral filter → normalised greyscale.
3. `estimate_group_center_from_diff_avg_robust` — abs-diff against baseline → Otsu threshold → morphological clean → contour filter → area-weighted group centre. Falls back to intensity-weighted centroid of the dominant blob if no individual holes pass the filter.
4. `estimate_px_per_cm_from_grid` — autocorrelation on row/column edge profiles to find the grid period in pixels.
5. `pixels_to_clicks` — pixel offset → cm → click count + direction.

### `ZeroingSession` (main stateful class)

Keeps a rolling baseline that advances after each `process_shot()` call, so only new holes are detected each round.

Key methods:
- `set_clean_target(img_bgr)` — sets the initial baseline, crops a center template for future alignment.
- `process_shot(img_bgr, show_debug, show_operator)` — runs the full pipeline; returns `{gc, aim, wind, elev, shot_warped, overlay, operator}`.

Constructor parameter:
- `max_dist_cm=None` — if set, holes further than this distance (in cm) from the aim point are excluded from detection. Converted to pixels using the per-shot `px_per_cm` estimate.

### `find_target_center`

Two modes: dark-blob heuristic (used on clean target at setup) and normalised template matching (used on all subsequent shots using the cropped template from `set_clean_target`).

### Tests

- `test_calc.py` — 76 tests against real target photos in `images/`.
- `test_synth.py` — 30 pipeline tests on synthetic images in `images/synth/`.
- `test_alignment.py` — alignment stress tests (rotation/scale variants are expected to fail on synthetic images — this is documented behaviour, not a bug).
- Tests use a custom minimal pass/fail harness (not pytest). All tests run headless (`show_debug=False, show_operator=False`).

### UI

Tkinter-based: `_run_setup_dialog()` collects distance, clicks/cm, and clean target path, then `_run_session_panel()` drives per-round file picking. OpenCV windows are used for the operator view and debug panel.
