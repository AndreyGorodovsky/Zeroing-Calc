"""
Generate synthetic zeroing target images for testing.

Each image is 900x1200 px at 40 px/cm (matches the real test images).
The target has:
  - White background
  - 1cm grid lines (every 40 px)
  - Concentric scoring rings
  - A filled black diamond as the aiming reticle

Bullet holes are dark filled circles drawn at known pixel positions,
so the expected click output can be calculated exactly.
"""

import cv2 as cv
import numpy as np
import os
from pathlib import Path

OUT = str(Path(__file__).resolve().parent.parent / "images" / "synth") + "/"
os.makedirs(OUT, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
W, H       = 900, 1200       # image size (matches real images)
PX_PER_CM  = 40              # grid pitch
GRID_CM    = 1.0             # one grid square = 1 cm
CX, CY     = W // 2, H // 2  # image centre = diamond centre
HOLE_R     = 9               # bullet hole radius in pixels
DIAMOND_R  = 36              # half-width of diamond


# ── drawing helpers ───────────────────────────────────────────────────────────

def make_blank_target():
    """Return a clean target image: white bg + grid + rings + diamond."""
    img = np.full((H, W, 3), 255, np.uint8)

    # --- grid lines ---
    for x in range(0, W, PX_PER_CM):
        cv.line(img, (x, 0), (x, H), (210, 210, 210), 1)
    for y in range(0, H, PX_PER_CM):
        cv.line(img, (0, y), (W, y), (210, 210, 210), 1)

    # --- concentric scoring rings (every 2 cm = 80 px) ---
    for r in range(80, max(W, H), 80):
        cv.circle(img, (CX, CY), r, (180, 180, 180), 1)

    # --- thick centre cross hair ---
    cv.line(img, (CX - 20, CY), (CX + 20, CY), (150, 150, 150), 1)
    cv.line(img, (CX, CY - 20), (CX, CY + 20), (150, 150, 150), 1)

    # --- diamond (filled black rotated square) ---
    pts = np.array([
        [CX,              CY - DIAMOND_R],   # top
        [CX + DIAMOND_R,  CY],               # right
        [CX,              CY + DIAMOND_R],   # bottom
        [CX - DIAMOND_R,  CY],               # left
    ], np.int32)
    cv.fillPoly(img, [pts], (0, 0, 0))

    return img


def add_hole(img, x, y, r=HOLE_R):
    """Draw a bullet hole (dark circle with slight grey shadow) at (x, y)."""
    out = img.copy()
    cv.circle(out, (x, y), r + 2, (80, 80, 80), -1)   # shadow
    cv.circle(out, (x, y), r,     (20, 20, 20), -1)   # hole
    return out


def save(img, name):
    path = OUT + name
    cv.imwrite(path, img)
    print(f"  saved  {name}")


# ── generate images ───────────────────────────────────────────────────────────

clean = make_blank_target()
save(clean, "synth_empty.png")

# ── single-hole shots ─────────────────────────────────────────────────────────
# Each image has one hole at a known offset from the diamond centre.
# Expected clicks = offset_px / PX_PER_CM / click_value_cm
# click_value_cm = 0.5, so clicks = offset_px / 40 / 0.5 = offset_px / 20

cases = [
    # name,                dx,  dy,   description
    ("synth_shot_below.png",   0,  80,  "hole 2cm below  -> 0 wind, +4 elev UP"),
    ("synth_shot_above.png",   0, -80, "hole 2cm above  -> 0 wind, -4 elev DOWN"),
    ("synth_shot_right.png",  80,   0, "hole 2cm right  -> +4 wind LEFT, 0 elev"),
    ("synth_shot_left.png",  -80,   0, "hole 2cm left   -> -4 wind RIGHT, 0 elev"),
    ("synth_shot_diag.png",   40,  40, "hole 1cm right+down -> +2 wind LEFT, +2 elev UP"),
    ("synth_shot_centre.png",  0,   0, "hole on aim     -> 0 wind, 0 elev (perfect zero)"),
]

print("\nSingle-hole images:")
for fname, dx, dy, desc in cases:
    img = add_hole(clean.copy(), CX + dx, CY + dy)
    save(img, fname)
    expected_wind = dx / 20
    expected_elev = dy / 20
    print(f"         {desc}")
    print(f"         expected: wind={expected_wind:+.1f}  elev={expected_elev:+.1f}")


# ── multi-hole group ──────────────────────────────────────────────────────────
# Three holes — group centre is the area-weighted average of their positions.
print("\nMulti-hole images:")

hole_positions_3 = [(CX + 40, CY + 60), (CX - 20, CY + 80), (CX + 10, CY + 100)]
img3 = clean.copy()
for hx, hy in hole_positions_3:
    img3 = add_hole(img3, hx, hy)
save(img3, "synth_group_3.png")

# All holes same radius -> equal weight -> simple average
gcx3 = sum(p[0] for p in hole_positions_3) / 3
gcy3 = sum(p[1] for p in hole_positions_3) / 3
dx3, dy3 = gcx3 - CX, gcy3 - CY
print(f"  synth_group_3.png  holes at {hole_positions_3}")
print(f"         group centre: ({gcx3:.1f},{gcy3:.1f})  offset: ({dx3:+.1f},{dy3:+.1f}) px")
print(f"         expected: wind={dx3/20:+.2f}  elev={dy3/20:+.2f}")


# Five holes spread wider
hole_positions_5 = [
    (CX + 60, CY + 40),
    (CX - 40, CY + 60),
    (CX + 20, CY - 40),
    (CX - 60, CY + 20),
    (CX + 10, CY + 80),
]
img5 = clean.copy()
for hx, hy in hole_positions_5:
    img5 = add_hole(img5, hx, hy)
save(img5, "synth_group_5.png")

gcx5 = sum(p[0] for p in hole_positions_5) / 5
gcy5 = sum(p[1] for p in hole_positions_5) / 5
dx5, dy5 = gcx5 - CX, gcy5 - CY
print(f"  synth_group_5.png  5 holes")
print(f"         group centre: ({gcx5:.1f},{gcy5:.1f})  offset: ({dx5:+.1f},{dy5:+.1f}) px")
print(f"         expected: wind={dx5/20:+.2f}  elev={dy5/20:+.2f}")


# ── registration test (shifted image) ────────────────────────────────────────
# Same as synth_shot_below but the whole image is shifted 15px right, 10px down.
# After _register_to() the click output should be the same as the unshifted version.
print("\nRegistration test image:")
SHIFT_X, SHIFT_Y = 15, 10
M = np.float32([[1, 0, SHIFT_X], [0, 1, SHIFT_Y]])
shifted = cv.warpAffine(clean, M, (W, H), borderValue=(255, 255, 255))
shifted = add_hole(shifted, CX + SHIFT_X, CY + 80 + SHIFT_Y)
save(shifted, "synth_shot_shifted.png")
print(f"  synth_shot_shifted.png  (same hole as below but image shifted {SHIFT_X},{SHIFT_Y} px)")
print(f"         expected after registration: wind=+0.0  elev=+4.0")


# ── edge cases ────────────────────────────────────────────────────────────────
print("\nEdge case images:")

# Two holes at exact opposite sides — group centre should be diamond centre
img_opposite = clean.copy()
img_opposite = add_hole(img_opposite, CX + 80, CY)
img_opposite = add_hole(img_opposite, CX - 80, CY)
save(img_opposite, "synth_opposite.png")
print("  synth_opposite.png  2 symmetric holes -> expected wind=0, elev=0")

# Outlier: one tight cluster + one far outlier
img_outlier = clean.copy()
for ox, oy in [(CX+5, CY+10), (CX-5, CY+15), (CX+8, CY+5)]:  # tight cluster near centre
    img_outlier = add_hole(img_outlier, ox, oy)
img_outlier = add_hole(img_outlier, CX + 300, CY + 300)  # far outlier
save(img_outlier, "synth_outlier.png")
print("  synth_outlier.png  3 tight holes + 1 far outlier")
print("         (tests whether outlier skews the group centre)")


print(f"\nAll images saved to {OUT}")
print("\nExpected click summary (wind/elev at 50m, 0.5cm/click, 40px/cm):")
print("  +wind = LEFT,  -wind = RIGHT")
print("  +elev = UP,    -elev = DOWN")
