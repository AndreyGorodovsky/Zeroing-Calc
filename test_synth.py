"""
Test the zeroing pipeline against synthetic images with known ground truth.

Strategy:
  - Verify GROUP CENTER POSITION in pixels — this is what the vision code controls.
  - Click values depend on px_per_cm calibration, which works on real photos (40px/cm)
    but returns 80px/cm on synthetic images due to their perfectly uniform grid.
    Click math is tested separately in test_calc.py with exact known values.
  - Registration accuracy is tested by verifying the aim pixel is correctly located
    after alignment.
"""

import sys, json, types, math
import numpy as np
import cv2 as cv

# ── load notebook cell 3 ─────────────────────────────────────────────────────
with open("C:/Users/Andrey/desktop/calc/Calc.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
source = "".join(nb["cells"][2]["source"])
mod = types.ModuleType("calc")
exec(compile(source, "Calc.ipynb[cell3]", "exec"), mod.__dict__)
ZeroingSession      = mod.ZeroingSession
_register_to        = mod._register_to
_to_gray_norm       = mod._to_gray_norm
find_diamond_center = mod.find_diamond_center

SYNTH = "C:/Users/Andrey/desktop/calc/images/synth/"
CX, CY = 450, 600   # diamond centre in synthetic images

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


def run(clean_path, shot_path):
    """Run a full session; return (wind, elev, gc, aim) without any GUI."""
    clean = cv.imread(clean_path)
    shot  = cv.imread(shot_path)
    assert clean is not None and shot is not None
    sess = ZeroingSession(distance_m=50.0, grid_cm=1.0, click_value_cm=0.5)
    sess.set_clean_target(clean)
    res = sess.process_shot(shot, show_debug=False, show_operator=False)
    return res["wind"], res["elev"], res["gc"], res["aim"]


CLEAN = SYNTH + "synth_empty.png"
PX_TOL = 12   # pixel tolerance for group centre (hole radius=9 + shadow=2 + rounding)


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Single-hole group centre position ===")
# ─────────────────────────────────────────────────────────────────────────────
# Hole placed at a known pixel offset from the diamond centre.
# We verify the detected gc is within PX_TOL of the true position.

single_cases = [
    # fname,                      dx,  dy
    ("synth_shot_below.png",       0, +80),
    ("synth_shot_above.png",       0, -80),
    ("synth_shot_right.png",     +80,   0),
    ("synth_shot_left.png",      -80,   0),
    ("synth_shot_diag.png",      +40, +40),
    ("synth_shot_centre.png",      0,   0),
]

for fname, dx, dy in single_cases:
    w, e, gc, aim = run(CLEAN, SYNTH + fname)
    label = fname.replace("synth_", "").replace(".png", "")
    true_x, true_y = CX + dx, CY + dy

    check(f"{label}: hole detected", gc is not None)
    if gc is None:
        continue

    dist = math.hypot(gc[0] - true_x, gc[1] - true_y)
    check(f"{label}: gc within {PX_TOL}px of true position ({true_x},{true_y})",
          dist <= PX_TOL,
          f"got ({gc[0]:.1f},{gc[1]:.1f}), dist={dist:.1f}px")

    # Sign of correction should always be correct even if magnitude varies
    if dx != 0:
        check(f"{label}: wind sign correct",
              (w > 0) == (dx > 0),
              f"dx={dx}, wind={w:+.2f}")
    if dy != 0:
        check(f"{label}: elev sign correct",
              (e > 0) == (dy > 0),
              f"dy={dy}, elev={e:+.2f}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Multi-hole group centre position ===")
# ─────────────────────────────────────────────────────────────────────────────

# 3 holes — true group centre is the simple average of hole centres (equal radii)
hole_pos_3 = [(CX+40, CY+60), (CX-20, CY+80), (CX+10, CY+100)]
true3_x = sum(p[0] for p in hole_pos_3) / 3
true3_y = sum(p[1] for p in hole_pos_3) / 3

w3, e3, gc3, aim3 = run(CLEAN, SYNTH + "synth_group_3.png")
check("group_3: holes detected",   gc3 is not None)
if gc3:
    dist3 = math.hypot(gc3[0] - true3_x, gc3[1] - true3_y)
    check(f"group_3: gc within {PX_TOL}px of true ({true3_x:.0f},{true3_y:.0f})",
          dist3 <= PX_TOL, f"got ({gc3[0]:.1f},{gc3[1]:.1f}), dist={dist3:.1f}px")
    print(f"         gc=({gc3[0]:.1f},{gc3[1]:.1f})  true=({true3_x:.0f},{true3_y:.0f})  dist={dist3:.1f}px")

# 5 holes
hole_pos_5 = [(CX+60,CY+40),(CX-40,CY+60),(CX+20,CY-40),(CX-60,CY+20),(CX+10,CY+80)]
true5_x = sum(p[0] for p in hole_pos_5) / 5
true5_y = sum(p[1] for p in hole_pos_5) / 5

w5, e5, gc5, aim5 = run(CLEAN, SYNTH + "synth_group_5.png")
check("group_5: holes detected",   gc5 is not None)
if gc5:
    dist5 = math.hypot(gc5[0] - true5_x, gc5[1] - true5_y)
    check(f"group_5: gc within {PX_TOL}px of true ({true5_x:.0f},{true5_y:.0f})",
          dist5 <= PX_TOL, f"got ({gc5[0]:.1f},{gc5[1]:.1f}), dist={dist5:.1f}px")
    print(f"         gc=({gc5[0]:.1f},{gc5[1]:.1f})  true=({true5_x:.0f},{true5_y:.0f})  dist={dist5:.1f}px")


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Registration (shifted image) ===")
# ─────────────────────────────────────────────────────────────────────────────
# The image was shifted 15px right, 10px down before the hole was added.
# After _register_to(), the aim pixel should land near (CX, CY)
# and the gc should be at the same position as the unshifted version.

w_s, e_s, gc_s, aim_s = run(CLEAN, SYNTH + "synth_shot_shifted.png")
check("shifted: hole detected",    gc_s is not None)

# Aim point after registration should be near the diamond centre
if aim_s:
    aim_dist = math.hypot(aim_s[0] - CX, aim_s[1] - CY)
    check("shifted: aim aligned to within 3px after registration",
          aim_dist <= 3, f"aim=({aim_s[0]:.1f},{aim_s[1]:.1f}), dist={aim_dist:.1f}px")

# gc should match the unshifted "below" case: true position = (CX, CY+80)
if gc_s:
    shifted_dist = math.hypot(gc_s[0] - CX, gc_s[1] - (CY + 80))
    check(f"shifted: gc within {PX_TOL}px of true position ({CX},{CY+80})",
          shifted_dist <= PX_TOL,
          f"got ({gc_s[0]:.1f},{gc_s[1]:.1f}), dist={shifted_dist:.1f}px")
    print(f"         aim=({aim_s[0]:.1f},{aim_s[1]:.1f})  gc=({gc_s[0]:.1f},{gc_s[1]:.1f})")


# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Edge cases ===")
# ─────────────────────────────────────────────────────────────────────────────

# Two symmetric holes -> group centre should land on the aim point
w_op, e_op, gc_op, aim_op = run(CLEAN, SYNTH + "synth_opposite.png")
check("opposite: holes detected", gc_op is not None)
if gc_op:
    dist_op = math.hypot(gc_op[0] - CX, gc_op[1] - CY)
    check("opposite: gc near aim (symmetric holes cancel out)",
          dist_op <= PX_TOL, f"dist={dist_op:.1f}px from aim")

# Outlier: 3 tight holes near aim + 1 far outlier.
# max_dist_px=None by design -> outlier IS included in the group centre.
# We verify the outlier pulls the gc significantly away from the tight cluster,
# confirming the code behaves as documented (no gating = no outlier rejection).
w_out, e_out, gc_out, aim_out = run(CLEAN, SYNTH + "synth_outlier.png")
OUTLIER_POS = (CX + 300, CY + 300)
CLUSTER_CENTER = (CX + 2, CY + 10)  # approximate centre of tight cluster

check("outlier: detected something", gc_out is not None)
if gc_out:
    dist_from_cluster = math.hypot(gc_out[0] - CLUSTER_CENTER[0],
                                   gc_out[1] - CLUSTER_CENTER[1])
    dist_from_outlier = math.hypot(gc_out[0] - OUTLIER_POS[0],
                                   gc_out[1] - OUTLIER_POS[1])
    # gc should be somewhere between cluster and outlier (outlier included)
    check("outlier: gc pulled away from tight cluster (outlier included as designed)",
          dist_from_cluster > 50,
          f"gc too close to cluster ({dist_from_cluster:.0f}px) — outlier may be rejected")
    check("outlier: gc not fully at outlier position",
          dist_from_outlier > 50,
          f"gc collapsed to outlier position")
    print(f"         gc=({gc_out[0]:.0f},{gc_out[1]:.0f})  "
          f"dist_from_cluster={dist_from_cluster:.0f}px  dist_from_outlier={dist_from_outlier:.0f}px")
    print(f"         NOTE: max_dist_px=None by design — outlier is intentionally included.")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*52}")
print(f"  Results:  {passed} passed   {failed} failed   ({passed+failed} total)")
print(f"{'='*52}\n")
sys.exit(0 if failed == 0 else 1)
