# Zeroing Calc

A computer-vision tool that analyses photos of a shooting target and outputs
scope adjustment clicks (windage + elevation) to zero a rifle.

## Workflow

1. **Register** — align the shot photo to the baseline using the aiming mark centre
2. **Preprocess** — CLAHE normalisation + bilateral filter to a normalised greyscale
3. **Detect holes** — abs-diff against the baseline → threshold → contour filter → area-weighted group centre
4. **Calibrate** — autocorrelation on the 1 cm grid lines to get px/cm
5. **Calculate** — pixel offset → cm → clicks + direction

## Usage

```python
sess = ZeroingSession(distance_m=50.0, grid_cm=1.0, click_value_cm=0.5)
sess.set_clean_target(cv.imread("clean_target.png"))

res1 = sess.process_shot(cv.imread("shot1.png"))
res2 = sess.process_shot(cv.imread("shot2.png"))
```

Each call to `process_shot` returns windage and elevation corrections in clicks.
The baseline advances automatically after each round so only fresh holes are detected.

Run the interactive UI with:

```bash
jupyter nbconvert --to script Calc.ipynb --stdout | python
```

## Running tests

```bash
python tests/test_calc.py
python tests/test_synth.py
```

106 tests, 0 failures.

## Files

| Path | Purpose |
|------|---------|
| `Calc.ipynb` | All source code |
| `tests/test_calc.py` | 76 unit tests against real images |
| `tests/test_synth.py` | 30 pipeline tests on synthetic images |
| `images/` | Real target photos used in tests |
| `images/synth/` | Synthetic test images |

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
