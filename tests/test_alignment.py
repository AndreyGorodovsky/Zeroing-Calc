"""
Alignment stress tests — rotation, scale, and combined misalignment.

_register_to uses similarity-transform estimation from the aiming-mark's
4 extreme points (estimateAffinePartial2D).  This correctly handles any
translation, rotation, and scale in the REGISTRATION step.

However, these tests use perfectly-periodic synthetic images (40 px grid,
80 px rings).  Even a tiny residual registration error (~0.3% scale,
~0.2 deg rotation) from pixel-quantisation of the diamond vertices creates
systematic diff noise across the whole image, which Otsu's threshold cannot
separate from the bullet-hole signal.  In real camera photos the background
is noisy/textured, so the same residuals are invisible and the bullet hole
dominates the diff.

Expected behaviour:
  - Baseline (no misalignment):  PASSES — registration is identity, diff is clean.
  - Rotation / scale variants:   Detection-limited failure on these synthetic
    images; NOT a failure of the registration transform itself.
"""

import sys, json, types, math
import numpy as np
import cv2 as cv
from pathlib import Path

# ── load notebook ─────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
with open(REPO / "Calc.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
source = "".join(nb["cells"][0]["source"])
mod = types.ModuleType("calc")
exec(compile(source, "Calc.ipynb[cell3]", "exec"), mod.__dict__)
ZeroingSession = mod.ZeroingSession

# ── image constants (match existing synth images) ─────────────────────────────
W, H      = 900, 1200
CX, CY    = W // 2, H // 2   # diamond centre = image centre
HOLE_R    = 9
DIAMOND_R = 36
HOLE_DY   = 80                # hole 80 px directly below aim (2 cm at 40 px/cm)
TRUE_X    = CX
TRUE_Y    = CY + HOLE_DY
PX_TOL    = 12                # same tolerance as test_synth.py

passed = failed = 0

def ok(name):
    global passed; passed += 1
    print(f"  PASS  {name}")

def fail(name, reason):
    global failed; failed += 1
    print(f"  FAIL  {name}: {reason}")

def check(name, condition, reason=""):
    if condition: ok(name)
    else: fail(name, reason)


# ── image helpers ─────────────────────────────────────────────────────────────

def make_blank_target():
    img = np.full((H, W, 3), 255, np.uint8)
    for x in range(0, W, 40):
        cv.line(img, (x, 0), (x, H), (210, 210, 210), 1)
    for y in range(0, H, 40):
        cv.line(img, (0, y), (W, y), (210, 210, 210), 1)
    for r in range(80, max(W, H), 80):
        cv.circle(img, (CX, CY), r, (180, 180, 180), 1)
    pts = np.array([
        [CX,             CY - DIAMOND_R],
        [CX + DIAMOND_R, CY],
        [CX,             CY + DIAMOND_R],
        [CX - DIAMOND_R, CY],
    ], np.int32)
    cv.fillPoly(img, [pts], (0, 0, 0))
    return img


def add_hole(img, x, y, r=HOLE_R):
    out = img.copy()
    cv.circle(out, (x, y), r + 2, (80, 80, 80), -1)
    cv.circle(out, (x, y), r,     (20, 20, 20), -1)
    return out


def run_pipeline(clean, shot):
    sess = ZeroingSession(distance_m=50.0, grid_cm=1.0, click_value_cm=0.5)
    sess.set_clean_target(clean)
    return sess.process_shot(shot, show_debug=False, show_operator=False)


def warp(img, angle_deg, scale):
    M = cv.getRotationMatrix2D((CX, CY), angle_deg, scale)
    return cv.warpAffine(img, M, (W, H), borderValue=(255, 255, 255))


# ── build reference images once ───────────────────────────────────────────────
clean    = make_blank_target()
shot_ref = add_hole(clean.copy(), TRUE_X, TRUE_Y)


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Baseline (no misalignment) ===")
# ─────────────────────────────────────────────────────────────────────────────
res = run_pipeline(clean, shot_ref)
gc  = res["gc"]
check("baseline: hole detected", gc is not None)
if gc:
    d = math.hypot(gc[0] - TRUE_X, gc[1] - TRUE_Y)
    check(f"baseline: gc within {PX_TOL}px of ({TRUE_X},{TRUE_Y})",
          d <= PX_TOL, f"dist={d:.1f}px")
    print(f"         gc=({gc[0]:.1f},{gc[1]:.1f})  dist={d:.1f}px")


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Rotation only ===")
# ─────────────────────────────────────────────────────────────────────────────
# Rotating around (CX,CY) keeps the diamond pixel-perfect at (CX,CY).
# Pure-translation registration applies no shift, so the image stays rotated.
# Error = HOLE_DY * sin(angle).  Predicted failures: >=10 deg.
for angle in [3, 5, 10, 15]:
    shot_t = warp(shot_ref, angle, 1.0)
    res    = run_pipeline(clean, shot_t)
    gc     = res["gc"]
    label  = f"rot_{angle:02d}deg"
    check(f"{label}: hole detected", gc is not None)
    if gc is None:
        continue
    d = math.hypot(gc[0] - TRUE_X, gc[1] - TRUE_Y)
    check(f"{label}: gc within {PX_TOL}px of ({TRUE_X},{TRUE_Y})",
          d <= PX_TOL, f"got ({gc[0]:.1f},{gc[1]:.1f}), dist={d:.1f}px")
    predicted_err = HOLE_DY * math.sin(math.radians(angle))
    print(f"         gc=({gc[0]:.1f},{gc[1]:.1f})  dist={d:.1f}px  "
          f"(predicted windage error ~{predicted_err:.1f}px)")


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Scale only ===")
# ─────────────────────────────────────────────────────────────────────────────
# Scale around (CX,CY) keeps the diamond at (CX,CY).
# The hole moves to (CX, CY + HOLE_DY*scale).
# px/cm also scales by the same factor, so click output self-corrects.
# Position error = HOLE_DY * |scale-1|.  Expected: all pass at PX_TOL=12.
for scale in [0.90, 0.95, 1.05, 1.10]:
    shot_t = warp(shot_ref, 0, scale)
    res    = run_pipeline(clean, shot_t)
    gc     = res["gc"]
    label  = f"scale_{scale:.2f}x"
    check(f"{label}: hole detected", gc is not None)
    if gc is None:
        continue
    d = math.hypot(gc[0] - TRUE_X, gc[1] - TRUE_Y)
    check(f"{label}: gc within {PX_TOL}px of ({TRUE_X},{TRUE_Y})",
          d <= PX_TOL, f"got ({gc[0]:.1f},{gc[1]:.1f}), dist={d:.1f}px")
    print(f"         gc=({gc[0]:.1f},{gc[1]:.1f})  dist={d:.1f}px")


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Combined rotation + scale ===")
# ─────────────────────────────────────────────────────────────────────────────
for angle, scale in [(5, 0.95), (10, 1.05), (15, 0.90)]:
    shot_t = warp(shot_ref, angle, scale)
    res    = run_pipeline(clean, shot_t)
    gc     = res["gc"]
    label  = f"rot_{angle:02d}deg_scale_{scale:.2f}x"
    check(f"{label}: hole detected", gc is not None)
    if gc is None:
        continue
    d = math.hypot(gc[0] - TRUE_X, gc[1] - TRUE_Y)
    check(f"{label}: gc within {PX_TOL}px of ({TRUE_X},{TRUE_Y})",
          d <= PX_TOL, f"got ({gc[0]:.1f},{gc[1]:.1f}), dist={d:.1f}px")
    print(f"         gc=({gc[0]:.1f},{gc[1]:.1f})  dist={d:.1f}px")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*52}")
print(f"  Results:  {passed} passed   {failed} failed   ({passed+failed} total)")
print(f"{'='*52}\n")
sys.exit(0 if failed == 0 else 1)
