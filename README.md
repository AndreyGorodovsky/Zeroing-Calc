# Zeroing Calc

A computer-vision tool that analyses photos of a shooting target and tells you
exactly how many scope clicks to apply — and in which direction — to zero your
rifle.

## How it works

The pipeline runs in five stages every time you photograph a new target:

```
Photo of target
      │
      ▼
1. Registration  ──  align the new photo to the baseline using the diamond
      │
      ▼
2. Preprocessing  ──  CLAHE normalisation + bilateral filter → grayscale
      │
      ▼
3. Hole detection  ──  abs-diff against baseline → threshold → contour filter
      │
      ▼
4. Calibration  ──  autocorrelation on the grid lines → px/cm
      │
      ▼
5. Click calculation  ──  pixel offset → cm → clicks + direction
```

---

### Stage 1 — Registration (`_register_to`)

The shooter holds the camera slightly differently each time, so the two photos
are never pixel-perfect aligned.

- Finds the black diamond aiming mark in both images (`find_diamond_center`).
- Builds a pure-translation homography (`H`) that shifts the new photo so the
  diamonds overlap exactly.
- Applies `warpPerspective` with a **white** border fill (white matches the
  target background and prevents the border from being mistaken for a hole).

### Stage 2 — Preprocessing (`_to_gray_norm`)

- Converts to LAB colour space and applies CLAHE to the L channel to normalise
  uneven lighting across photos.
- Converts back to BGR then to grayscale.
- Applies a bilateral filter to smooth noise while preserving sharp hole edges.

### Stage 3 — Hole detection (`estimate_group_center_from_diff_avg_robust`)

New holes are found by comparing the current photo to the previous baseline:

1. `cv.absdiff` — pixels that changed between shots stand out.
2. Gaussian blur + Otsu threshold → binary mask.
3. Morphological open/close to remove noise and seal small gaps.
4. `findContours` + shape filters (area, circularity ≥ 0.15, solidity > 0.7)
   → list of candidate holes.
5. Each hole is represented as `(cx, cy, area, circularity)`.
6. Group centre is the **area-weighted average** of all accepted holes.
7. Fallback: if no individual contours pass the filter (e.g. holes are merged
   into one blob), the largest connected component's intensity-weighted centroid
   is used instead.
8. Optional `max_dist_px` gate — holes further than N pixels from the aim point
   can be rejected as outliers (disabled by default).

A helper `_contour_filter` is also used during the debug visualisation path
to annotate individual holes on screen.

### Stage 4 — Scale calibration (`estimate_px_per_cm_from_grid`)

The target has a 1 cm grid printed on it. The function measures how many pixels
span one grid square so that pixel offsets can be converted to real distances:

1. Canny edge detection on the normalised grayscale image.
2. Collapse edges to a 1-D horizontal and vertical profile by summing rows/cols.
3. Autocorrelation of each profile (`_period_from_profile`) — the dominant peak
   lag equals the grid period in pixels.
4. Average of horizontal and vertical periods → `px_per_cm`.

### Stage 5 — Click calculation (`pixels_to_clicks` / `_format_clicks`)

```
dx_px = group_center_x - aim_x      # +: group is to the RIGHT
dy_px = group_center_y - aim_y      # +: group is BELOW

dx_cm = dx_px / px_per_cm
dy_cm = dy_px / px_per_cm

wind  = dx_cm / click_value_cm      # > 0 → turn LEFT
elev  = dy_cm / click_value_cm      # > 0 → turn UP
```

Default `click_value_cm = 0.5` (each click moves point of impact 0.5 cm at the
shooting distance). `_format_clicks` converts the signed floats into a
human-readable string: `"6.8 clicks LEFT | 15.1 clicks UP"`.

---

## Session workflow (`ZeroingSession`)

`ZeroingSession` manages a moving baseline so each round is compared only
against the previous state of the target (not the very first clean target).
This means it detects only the *new* holes from each shot.

```python
sess = ZeroingSession(distance_m=50.0, grid_cm=1.0, click_value_cm=0.5)

sess.set_clean_target(cv.imread("clean_target.png"))  # baseline = clean target

res1 = sess.process_shot(cv.imread("shot1.png"))      # baseline → shot1
res2 = sess.process_shot(cv.imread("shot2.png"))      # baseline → shot2
```

`process_shot` returns a dict:

| Key | Contents |
|-----|----------|
| `gc` | `(x, y)` group centre in pixels, or `None` |
| `aim` | `(x, y)` diamond centre in pixels |
| `wind` | windage correction in clicks (+ = LEFT) |
| `elev` | elevation correction in clicks (+ = UP) |
| `shot_warped` | registered shot image |
| `overlay` | debug annotated image |
| `operator` | clean operator view with correction arrow and click banner |

After each round the baseline is automatically advanced to the current shot
so subsequent rounds only highlight fresh holes.

---

## Files

| File | Purpose |
|------|---------|
| `Calc.ipynb` | Main notebook — all source code + demo runs |
| `test_calc.py` | 76 unit tests covering every function + real image |
| `test_synth.py` | 30 pipeline tests on synthetic target images |
| `generate_test_images.py` | Generates the synthetic images in `synth/` |
| `Real.jpeg` | Real target photo used in tests |
| `target_empty.png` | Clean target (baseline for demo) |
| `t1–t7.png` | Sample shot images used during development |
| `synth/` | 12 generated synthetic test images |

## Running the tests

```bash
cd C:\Users\Andrey\Desktop\Calc
python test_calc.py
python test_synth.py
```

106 tests, 0 failures.

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
