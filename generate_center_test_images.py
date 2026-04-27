"""
Generate synthetic zeroing target images for different center/reticle shapes.

Shapes covered:
  1. hollow_circle  – ring outline
  2. cross          – thick crosshair lines
  3. filled_dot     – solid circle
  4. bullseye       – dot + two concentric rings
  5. open_cross     – crosshair with a clear gap at the centre

For every shape we produce:
  {shape}_clean.png     – target with no holes (used as baseline)
  {shape}_shot_a.png    – single hole 2 cm right, 1.5 cm below  (known offset)
  {shape}_shot_b.png    – 3-hole group scattered around +1 cm right, +1 cm down

Expected clicks (50 m, 0.5 cm/click, 40 px/cm):
  shot_a  single hole dx=+80 px, dy=+60 px
          wind = 80/40/0.5 = +4.0 LEFT   elev = 60/40/0.5 = +3.0 UP
  shot_b  group centre dx=+40 px, dy=+46.7 px  (average of the 3 holes below)
          wind ≈ +2.0 LEFT               elev ≈ +2.3 UP
"""

import cv2 as cv
import numpy as np
import os

OUT = "C:/Users/Andrey/Desktop/Calc/images/synth_centers/"
os.makedirs(OUT, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
W, H      = 900, 1200
PX_PER_CM = 40
CX, CY    = W // 2, H // 2
HOLE_R    = 9
MARK_R    = 36   # radius / half-width of the aiming mark


# ── background template (grid + rings, NO centre mark) ────────────────────────
def _make_background():
    img = np.full((H, W, 3), 255, np.uint8)
    for x in range(0, W, PX_PER_CM):
        cv.line(img, (x, 0), (x, H), (210, 210, 210), 1)
    for y in range(0, H, PX_PER_CM):
        cv.line(img, (0, y), (W, y), (210, 210, 210), 1)
    for r in range(80, max(W, H), 80):
        cv.circle(img, (CX, CY), r, (180, 180, 180), 1)
    return img


# ── centre-mark drawing functions ─────────────────────────────────────────────

def _draw_hollow_circle(img):
    """Ring outline – 6 px thick."""
    cv.circle(img, (CX, CY), MARK_R, (0, 0, 0), 6)


def _draw_cross(img):
    """Bold crosshair lines, no gap at centre."""
    t = 5          # line thickness
    arm = MARK_R + 10
    cv.line(img, (CX - arm, CY), (CX + arm, CY), (0, 0, 0), t)
    cv.line(img, (CX, CY - arm), (CX, CY + arm), (0, 0, 0), t)


def _draw_filled_dot(img):
    """Solid filled circle."""
    cv.circle(img, (CX, CY), MARK_R, (0, 0, 0), -1)


def _draw_bullseye(img):
    """Filled inner dot + two concentric ring outlines."""
    cv.circle(img, (CX, CY), MARK_R,      (0, 0, 0), 4)   # outer ring
    cv.circle(img, (CX, CY), MARK_R // 2, (0, 0, 0), 4)   # inner ring
    cv.circle(img, (CX, CY), 6,           (0, 0, 0), -1)  # centre dot


def _draw_open_cross(img):
    """Crosshair with a clear gap of 12 px around the centre."""
    t   = 5
    gap = 12
    arm = MARK_R + 10
    # horizontal arms (left and right of gap)
    cv.line(img, (CX - arm, CY), (CX - gap, CY), (0, 0, 0), t)
    cv.line(img, (CX + gap, CY), (CX + arm, CY), (0, 0, 0), t)
    # vertical arms (above and below gap)
    cv.line(img, (CX, CY - arm), (CX, CY - gap), (0, 0, 0), t)
    cv.line(img, (CX, CY + gap), (CX, CY + arm), (0, 0, 0), t)


SHAPES = {
    "hollow_circle": _draw_hollow_circle,
    "cross":         _draw_cross,
    "filled_dot":    _draw_filled_dot,
    "bullseye":      _draw_bullseye,
    "open_cross":    _draw_open_cross,
}


# ── bullet hole helper ────────────────────────────────────────────────────────
def _add_hole(img, x, y, r=HOLE_R):
    out = img.copy()
    cv.circle(out, (x, y), r + 2, (80, 80, 80), -1)   # shadow
    cv.circle(out, (x, y), r,     (20, 20, 20), -1)   # hole
    return out


def _save(img, name):
    path = OUT + name
    cv.imwrite(path, img)
    print(f"  saved  {name}")


# ── shot-B hole positions (fixed for all shapes) ─────────────────────────────
# three holes whose simple average lands at (+40 px, +46.7 px) from centre
HOLES_B = [
    (CX + 60, CY + 20),   # +60, +20
    (CX + 20, CY + 60),   # +20, +60
    (CX + 40, CY + 60),   # +40, +60
]
GCX_B = sum(p[0] for p in HOLES_B) / 3
GCY_B = sum(p[1] for p in HOLES_B) / 3
DX_B  = GCX_B - CX
DY_B  = GCY_B - CY

# shot-A hole (single)
HOLE_A = (CX + 80, CY + 60)   # +2 cm right, +1.5 cm below

# shot-B round-2: new cluster added on top of shot_b image.
# Placed upper-left quadrant so it is visually separate from the shot_b group.
# Group centre: -2.5 cm left (-100 px), -2 cm above (-80 px)
# Expected new-round clicks (relative to aim):
#   wind = -100/40/0.5 = -5.0  (RIGHT)
#   elev = -80/40/0.5  = -4.0  (DOWN)
HOLES_C = [
    (CX - 120, CY - 70),   # -120, -70
    (CX - 80,  CY - 90),   # -80,  -90
    (CX - 100, CY - 80),   # -100, -80
]
GCX_C = sum(p[0] for p in HOLES_C) / 3
GCY_C = sum(p[1] for p in HOLES_C) / 3
DX_C  = GCX_C - CX
DY_C  = GCY_C - CY


# ── generate images ───────────────────────────────────────────────────────────
bg = _make_background()

print("Generating centre-shape test images ...\n")

for shape_name, draw_fn in SHAPES.items():
    print(f"  [{shape_name}]")

    # --- clean target ---
    clean = bg.copy()
    draw_fn(clean)
    _save(clean, f"{shape_name}_clean.png")

    # --- shot A: single hole (+2 cm right, +1.5 cm below) ---
    img_a = _add_hole(clean.copy(), *HOLE_A)
    _save(img_a, f"{shape_name}_shot_a.png")
    wind_a = (HOLE_A[0] - CX) / PX_PER_CM / 0.5
    elev_a = (HOLE_A[1] - CY) / PX_PER_CM / 0.5
    print(f"         shot_a  hole at +{HOLE_A[0]-CX}px, +{HOLE_A[1]-CY}px  "
          f"->  wind={wind_a:+.1f}  elev={elev_a:+.1f}")

    # --- shot B: 3-hole group (round 1) ---
    img_b = clean.copy()
    for hx, hy in HOLES_B:
        img_b = _add_hole(img_b, hx, hy)
    _save(img_b, f"{shape_name}_shot_b.png")
    wind_b = DX_B / PX_PER_CM / 0.5
    elev_b = DY_B / PX_PER_CM / 0.5
    print(f"         shot_b  group centre at ({DX_B:+.1f}px, {DY_B:+.1f}px)  "
          f"->  wind={wind_b:+.2f}  elev={elev_b:+.2f}")

    # --- shot B round 2: shot_b holes + new cluster (session test) ---
    # The session will diff this against shot_b, so only HOLES_C should appear.
    img_b_r2 = img_b.copy()
    for hx, hy in HOLES_C:
        img_b_r2 = _add_hole(img_b_r2, hx, hy)
    _save(img_b_r2, f"{shape_name}_shot_b_r2.png")
    wind_c = DX_C / PX_PER_CM / 0.5
    elev_c = DY_C / PX_PER_CM / 0.5
    print(f"         shot_b_r2  new cluster at ({DX_C:+.1f}px, {DY_C:+.1f}px)  "
          f"->  wind={wind_c:+.2f}  elev={elev_c:+.2f}  (session diff vs shot_b)")

    print()

print(f"All images saved to {OUT}")
print("\nExpected click summary (50 m, 0.5 cm/click, 40 px/cm):")
print("  shot_a:   single hole  wind=+4.0 LEFT   elev=+3.0 UP")
print(f"  shot_b:   group       wind={DX_B/PX_PER_CM/0.5:+.2f} LEFT   elev={DY_B/PX_PER_CM/0.5:+.2f} UP")
print(f"  shot_b_r2 new cluster wind={DX_C/PX_PER_CM/0.5:+.2f} RIGHT  elev={DY_C/PX_PER_CM/0.5:+.2f} DOWN  (diff vs shot_b)")
