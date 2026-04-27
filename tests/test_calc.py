"""
Test suite for Calc.ipynb zeroing logic.
Runs headless (no GUI windows). Uses the real images from the calc folder.
"""

import sys
import math
import numpy as np
import cv2 as cv
from pathlib import Path

# == load the source code from the notebook cell ==============================
import json, types

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "Calc.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

source = "".join(nb["cells"][0]["source"])
mod = types.ModuleType("calc")
exec(compile(source, "Calc.ipynb[cell3]", "exec"), mod.__dict__)

# Pull everything we need into local scope
_to_gray_norm                          = mod._to_gray_norm
find_diamond_center                    = mod.find_diamond_center
_register_to                           = mod._register_to
_contour_filter                        = mod._contour_filter
estimate_group_center_from_diff_avg_robust = mod.estimate_group_center_from_diff_avg_robust
estimate_px_per_cm_from_grid           = mod.estimate_px_per_cm_from_grid
_period_from_profile                   = mod._period_from_profile
pixels_to_clicks                       = mod.pixels_to_clicks
_format_clicks                         = mod._format_clicks
ZeroingSession                         = mod.ZeroingSession

BASE = str(REPO / "images") + "/"

# == tiny test harness =========================================================
passed = 0
failed = 0

def ok(name):
    global passed
    passed += 1
    print(f"  PASS  {name}")

def fail(name, reason):
    global failed
    failed += 1
    print(f"  FAIL  {name}: {reason}")

def check(name, condition, reason="assertion failed"):
    if condition:
        ok(name)
    else:
        fail(name, reason)

# == load images ===============================================================
print("\n=== Loading images ===")
imgs = {}
for fname in ["target_empty.png", "t1.png", "t2.png", "t3.png",
              "t4.png", "t4_two_clusters.png", "t5.png", "t6.png",
              "t7.png", "outlier_test_clean.png", "Real.jpeg"]:
    img = cv.imread(BASE + fname)
    imgs[fname] = img
    status = "OK" if img is not None else "MISSING"
    print(f"  {status:7s}  {fname}  {img.shape if img is not None else ''}")

clean   = imgs["target_empty.png"]
shot1   = imgs["t4.png"]
shot2   = imgs["t4_two_clusters.png"]
real    = imgs["Real.jpeg"]


# =============================================================================
print("\n=== 1. _to_gray_norm ===")
# =============================================================================

gray = _to_gray_norm(clean)
check("returns 2-D array",    gray.ndim == 2)
check("same H/W as input",    gray.shape == clean.shape[:2])
check("dtype is uint8",       gray.dtype == np.uint8)
check("not all-zero",         gray.max() > 0)
check("not all-same value",   gray.std() > 1.0,
      f"std={gray.std():.2f} = image looks flat")


# =============================================================================
print("\n=== 2. find_diamond_center ===")
# =============================================================================

cx, cy = find_diamond_center(clean)
h, w = clean.shape[:2]
check("returns two ints",     isinstance(cx, int) and isinstance(cy, int))
check("cx inside image",      0 <= cx < w,  f"cx={cx}, w={w}")
check("cy inside image",      0 <= cy < h,  f"cy={cy}, h={h}")

# Diamond should be roughly in the centre-ish of the target (not a corner)
margin = 0.15
check("cx not on left edge",  cx > w * margin,   f"cx={cx}")
check("cx not on right edge", cx < w * (1-margin), f"cx={cx}")
check("cy not on top edge",   cy > h * margin,   f"cy={cy}")
check("cy not on btm edge",   cy < h * (1-margin), f"cy={cy}")

# Consistency: calling twice on same image gives same result
cx2, cy2 = find_diamond_center(clean)
check("deterministic",        cx == cx2 and cy == cy2)

# Shot image should find the diamond too
if shot1 is not None:
    sx, sy = find_diamond_center(shot1)
    check("works on shot image",  0 <= sx < shot1.shape[1] and 0 <= sy < shot1.shape[0])


# =============================================================================
print("\n=== 3. _register_to ===")
# =============================================================================

if shot1 is not None:
    warped, H = _register_to(clean, shot1)

    check("warped same shape as ref",  warped.shape == clean.shape)
    check("H is 3x3",                  H.shape == (3, 3))
    check("H top-left is 1",           abs(H[0, 0] - 1) < 1e-5)
    check("H is pure translation",     abs(H[0, 1]) < 1e-5 and abs(H[1, 0]) < 1e-5,
          "rotation component found = unexpected")

    # After alignment the diamond centers should be very close
    ref_cx, ref_cy = find_diamond_center(clean)
    w_cx,  w_cy   = find_diamond_center(warped)
    dist = math.hypot(w_cx - ref_cx, w_cy - ref_cy)
    check("diamond aligned to within 3 px", dist <= 3,
          f"residual={dist:.1f} px")

    # Registering a clean image to itself should give near-zero shift
    same, H_id = _register_to(clean, clean)
    check("self-register: H is identity-like",
          abs(H_id[0, 2]) < 2 and abs(H_id[1, 2]) < 2,
          f"dx={H_id[0,2]:.1f} dy={H_id[1,2]:.1f}")


# =============================================================================
print("\n=== 4. _contour_filter ===")
# =============================================================================

# Synthetic image: one perfect circle (should pass) + one tiny dot (should be rejected)
syn = np.zeros((200, 200), np.uint8)
cv.circle(syn, (100, 100), 15, 255, -1)   # good circle ~706 px area
cv.circle(syn, (20, 20),    3, 255, -1)   # tiny dot ~28 px area = below min_area=150

result = _contour_filter(syn, min_area=150, max_area=8000, circ_min=0.2)
check("detects the good circle",       len(result) == 1,
      f"got {len(result)} contours")
if result:
    rx, ry, area, circ = result[0]
    check("center near (100,100)",     math.hypot(rx-100, ry-100) < 5,
          f"center=({rx},{ry})")
    check("circularity is high",       circ > 0.7, f"circ={circ:.2f}")

# All-black image = should return nothing
empty_result = _contour_filter(np.zeros((100, 100), np.uint8))
check("empty image -> no contours",    len(empty_result) == 0)


# =============================================================================
print("\n=== 5. estimate_group_center_from_diff_avg_robust ===")
# =============================================================================

# --- 5a: synthetic test = one known hole ---
ref_syn = np.full((300, 300), 200, np.uint8)
shot_syn = ref_syn.copy()
cv.circle(shot_syn, (150, 120), 10, 50, -1)   # dark hole at (150, 120)

gcx, gcy, r_eq, mask, diff, holes = estimate_group_center_from_diff_avg_robust(
    ref_syn, shot_syn, blur_ks=3, open_ks=3, close_ks=3,
    min_area=50, max_area=5000, circ_min=0.2)

check("detects synthetic hole",       gcx is not None, "returned None")
if gcx is not None:
    dist = math.hypot(gcx - 150, gcy - 120)
    check("center within 5 px of truth", dist < 5,
          f"got ({gcx:.1f},{gcy:.1f}), expected (150,120), dist={dist:.1f}")
    check("r_eq is positive",          r_eq > 0)
    check("mask is 2-D uint8",         mask.ndim == 2 and mask.dtype == np.uint8)

# --- 5b: two holes = group center should be between them ---
ref2 = np.full((300, 400), 200, np.uint8)
shot2_syn = ref2.copy()
cv.circle(shot2_syn, (100, 150), 10, 50, -1)   # left hole
cv.circle(shot2_syn, (300, 150), 10, 50, -1)   # right hole

gcx2, gcy2, _, _, _, holes2 = estimate_group_center_from_diff_avg_robust(
    ref2, shot2_syn, blur_ks=3, open_ks=3, close_ks=3,
    min_area=50, max_area=5000, circ_min=0.1)

check("two-hole: group center detected",    gcx2 is not None)
if gcx2 is not None:
    check("two-hole: center X between holes", 100 < gcx2 < 300,
          f"gcx={gcx2:.1f}")
    check("two-hole: center Y near 150",      abs(gcy2 - 150) < 10,
          f"gcy={gcy2:.1f}")
    check("two-hole: found 2 holes",          len(holes2) == 2,
          f"found {len(holes2)}")

# --- 5c: identical images = should return nothing ---
gcx3, gcy3, _, _, _, _ = estimate_group_center_from_diff_avg_robust(
    ref_syn, ref_syn)
check("identical images -> no detection",  gcx3 is None,
      f"got gc=({gcx3},{gcy3})")

# --- 5d: real images ---
if clean is not None and shot1 is not None:
    clean_g = _to_gray_norm(clean)
    shot1_w, _ = _register_to(clean, shot1)
    shot1_g = _to_gray_norm(shot1_w)
    gcx_r, gcy_r, r_r, _, _, holes_r = estimate_group_center_from_diff_avg_robust(
        clean_g, shot1_g, blur_ks=5, open_ks=3, close_ks=3,
        min_area=80, max_area=6000, circ_min=0.15)
    check("real image: group center found", gcx_r is not None,
          "returned None = no holes detected in t4.png vs empty target")
    if gcx_r is not None:
        check("real image: at least 1 hole",  len(holes_r) >= 1,
              f"holes={len(holes_r)}")
        print(f"         real group center: ({gcx_r:.1f}, {gcy_r:.1f})  holes={len(holes_r)}")


# =============================================================================
print("\n=== 6. estimate_px_per_cm_from_grid ===")
# =============================================================================

if clean is not None:
    gray_clean = _to_gray_norm(clean)
    px = estimate_px_per_cm_from_grid(gray_clean, grid_size_cm=1.0)
    check("returns a positive float",  isinstance(px, float) and px > 0,
          f"px={px}")
    check("plausible px/cm (5=300)",   5 < px < 300,
          f"px_per_cm={px:.1f} = outside expected range")
    print(f"         estimated px/cm = {px:.1f}")

    # Consistency: calling twice gives same result
    px2 = estimate_px_per_cm_from_grid(gray_clean, grid_size_cm=1.0)
    check("deterministic",             abs(px - px2) < 0.01)

    # Scaling: 2 cm grid should give half the px/cm
    px_2cm = estimate_px_per_cm_from_grid(gray_clean, grid_size_cm=2.0)
    check("grid_cm scales result",     abs(px_2cm - px / 2) < 2.0,
          f"1cm->{px:.1f}, 2cm->{px_2cm:.1f}, expected ~{px/2:.1f}")


# =============================================================================
print("\n=== 7. pixels_to_clicks ===")
# =============================================================================

# Known values: 40 px/cm, 0.5 cm/click
# 40 px right -> 1 cm right -> 2 clicks wind (LEFT correction)
# 40 px down  -> 1 cm down  -> 2 clicks elev (UP correction)
w, e = pixels_to_clicks(40, 40, px_per_cm=40, distance_m=50, click_value_cm=0.5)
check("40px right @ 40px/cm -> 2.0 wind", abs(w - 2.0) < 1e-9, f"wind={w}")
check("40px down  @ 40px/cm -> 2.0 elev", abs(e - 2.0) < 1e-9, f"elev={e}")

# Zero offset -> zero clicks
w0, e0 = pixels_to_clicks(0, 0, px_per_cm=40, distance_m=50, click_value_cm=0.5)
check("zero offset -> zero clicks",       w0 == 0.0 and e0 == 0.0)

# Negative offsets (group to left/above aim)
wn, en = pixels_to_clicks(-20, -20, px_per_cm=40, distance_m=50, click_value_cm=0.5)
check("negative dx -> negative wind",     wn < 0, f"wind={wn}")
check("negative dy -> negative elev",     en < 0, f"elev={en}")

# Proportionality: doubling px_per_cm halves the clicks
w2, e2 = pixels_to_clicks(40, 40, px_per_cm=80, distance_m=50, click_value_cm=0.5)
check("double px/cm -> half clicks",      abs(w2 - 1.0) < 1e-9, f"wind={w2}")


# =============================================================================
print("\n=== 8. _format_clicks ===")
# =============================================================================

dx, dy, wx, ey = _format_clicks(2.0, 3.0)
check("positive wind -> LEFT",   dx == "LEFT",  f"got '{dx}'")
check("positive elev -> UP",     dy == "UP",    f"got '{dy}'")
check("magnitudes are abs",     wx == 2.0 and ey == 3.0)

dx2, dy2, wx2, ey2 = _format_clicks(-1.5, -2.5)
check("negative wind -> RIGHT",  dx2 == "RIGHT", f"got '{dx2}'")
check("negative elev -> DOWN",   dy2 == "DOWN",  f"got '{dy2}'")
check("magnitudes correct",      wx2 == 1.5 and ey2 == 2.5)

dx3, dy3, _, _ = _format_clicks(0.0, 0.0)
check("zero wind -> LEFT (edge case accepted)", dx3 in ("LEFT", "RIGHT"))


# =============================================================================
print("\n=== 9. ZeroingSession = full pipeline ===")
# =============================================================================

if clean is not None and shot1 is not None:
    sess = ZeroingSession(distance_m=50.0, grid_cm=1.0, click_value_cm=0.5)

    # set_clean_target
    sess.set_clean_target(clean)
    check("baseline set after set_clean_target",
          sess.baseline_bgr is not None and sess.baseline_gray is not None)
    check("round_index reset to 0", sess.round_index == 0)

    # calling before set_clean_target should raise
    sess2 = ZeroingSession()
    try:
        sess2.process_shot(shot1, show_debug=False, show_operator=False)
        fail("process_shot before baseline: should raise RuntimeError", "no exception raised")
    except RuntimeError:
        ok("process_shot before baseline raises RuntimeError")

    # process_shot = round 1
    res1 = sess.process_shot(shot1, show_debug=False, show_operator=False)

    check("returns a dict",           isinstance(res1, dict))
    check("has 'aim' key",            "aim" in res1)
    check("has 'wind' key",           "wind" in res1)
    check("has 'elev' key",           "elev" in res1)
    check("has 'gc' key",             "gc" in res1)
    check("has 'shot_warped' key",    "shot_warped" in res1)
    check("has 'overlay' key",        "overlay" in res1)
    check("has 'operator' key",       "operator" in res1)

    check("round_index incremented",  sess.round_index == 1)
    check("aim is a 2-tuple",         isinstance(res1["aim"], tuple) and len(res1["aim"]) == 2)
    check("wind is float",            isinstance(res1["wind"], float))
    check("elev is float",            isinstance(res1["elev"], float))

    check("shot_warped same shape as clean",
          res1["shot_warped"].shape == clean.shape)

    if res1["gc"] is not None:
        gcx_s, gcy_s = res1["gc"]
        check("gc values are finite",
              math.isfinite(gcx_s) and math.isfinite(gcy_s))
        print(f"         round 1 -> wind={res1['wind']:+.2f}  elev={res1['elev']:+.2f}  "
              f"gc=({gcx_s:.1f},{gcy_s:.1f})")
    else:
        print("         round 1 -> no group center detected")

    # baseline should have rolled forward
    check("baseline updated after shot",
          not np.array_equal(sess.baseline_bgr, clean))

    # process_shot = round 2 (two-cluster image)
    if shot2 is not None:
        res2 = sess.process_shot(shot2, show_debug=False, show_operator=False)
        check("round 2 completes",        isinstance(res2, dict))
        check("round_index = 2",          sess.round_index == 2)
        if res2["gc"] is not None:
            print(f"         round 2 -> wind={res2['wind']:+.2f}  elev={res2['elev']:+.2f}  "
                  f"gc=({res2['gc'][0]:.1f},{res2['gc'][1]:.1f})")

    # None image should raise
    try:
        sess.process_shot(None, show_debug=False, show_operator=False)
        fail("process_shot(None): should raise ValueError", "no exception raised")
    except ValueError:
        ok("process_shot(None) raises ValueError")


# =============================================================================
print("\n=== 10. Real.jpeg ===")
# =============================================================================

if real is not None:
    gray_r = _to_gray_norm(real)
    cx_r, cy_r = find_diamond_center(real)
    px_r = estimate_px_per_cm_from_grid(gray_r, grid_size_cm=1.0)

    check("Real.jpeg: gray conversion OK",   gray_r is not None and gray_r.max() > 0)
    check("Real.jpeg: diamond found",        0 <= cx_r < real.shape[1] and
                                             0 <= cy_r < real.shape[0])
    check("Real.jpeg: px/cm plausible",      5 < px_r < 300, f"px_per_cm={px_r:.1f}")
    print(f"         Real.jpeg -> diamond=({cx_r},{cy_r})  px/cm={px_r:.1f}")


# =============================================================================
print(f"\n{'='*52}")
print(f"  Results:  {passed} passed   {failed} failed   "
      f"({passed+failed} total)")
print(f"{'='*52}\n")

# clean up temp file
import os
try:
    os.remove(REPO / "extracted.py")
except Exception:
    pass

sys.exit(0 if failed == 0 else 1)
