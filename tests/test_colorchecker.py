"""
Brzi test: detect_and_measure na jednoj slici.

Pokretanje:
    source env/bin/activate
    python test_colorchecker.py <putanja_do_slike>

    npr. python test_colorchecker.py data/moja_slika.jpeg
"""

import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="ColorChecker test")
parser.add_argument("image_path", help="Path to image")
args = parser.parse_args()

image_path = args.image_path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s  %(levelname)s  %(message)s",
)

print("Using image:", image_path)

img_path = Path(sys.argv[1])
if not img_path.exists():
    print(f"GREŠKA: datoteka ne postoji: {img_path}")
    sys.exit(1)

# ── Učitaj sliku ──────────────────────────────────────────────────────────────
print(f"\n[1] Učitavam: {img_path}")
bgr = cv.imread(str(img_path))

# HEIF/HEVC (iPhone format) — OpenCV ne podržava, konvertiraj via sips
if bgr is None:
    import subprocess, tempfile

    tmp = Path(tempfile.mktemp(suffix=".jpeg"))
    print("    HEIF format detektiran — konvertiram via sips ...")
    ret = subprocess.run(
        ["sips", "-s", "format", "jpeg", str(img_path), "--out", str(tmp)],
        capture_output=True,
    )
    if ret.returncode != 0 or not tmp.exists():
        print(f"GREŠKA: sips konverzija nije uspjela:\n{ret.stderr.decode()}")
        sys.exit(1)
    bgr = cv.imread(str(tmp))
    tmp.unlink(missing_ok=True)

if bgr is None:
    print("GREŠKA: OpenCV nije mogao učitati sliku ni nakon konverzije.")
    sys.exit(1)

img_srgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
print(f"    Rezolucija: {img_srgb.shape[1]} × {img_srgb.shape[0]} px")

# ── Detekcija ─────────────────────────────────────────────────────────────────
print("\n[2] Pokrećem detect_and_measure (log=True) ...")
from src.colorchecker import detect_and_measure, ColorCheckerResult

output_dir = Path("debug_output")
result = detect_and_measure(
    img_srgb, show_steps=True, log=True, save_steps=True, output_dir=output_dir
)

# ── Rezultati ──────────────────────────────────c───────────────────────────────
print("\n" + "=" * 60)
print(f"detected        : {result.detected}")
print(f"failure_reason  : {result.failure_reason}")
print(f"checker_type    : {result.checker_type}")
print(f"n_swatches      : {result.n_swatches}")

if result.detected:
    print(f"\nswatch_colors_xyz  shape : {result.swatch_colors_xyz.shape}")
    print(f"reference_colors_xyz shape: {result.reference_colors_xyz.shape}")
    print(f"\nΔE00 statistike:")
    print(f"  mean   = {result.delta_e_stats['mean']:.2f}")
    print(f"  median = {result.delta_e_stats['median']:.2f}")
    print(f"  p95    = {result.delta_e_stats['p95']:.2f}")

    # Prikaz prvih 5 swatcheva: izmjereni vs referentni Lab
    import colour

    D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
    meas_lab = colour.XYZ_to_Lab(result.swatch_colors_xyz, illuminant=D50)
    ref_lab = colour.XYZ_to_Lab(result.reference_colors_xyz, illuminant=D50)
    print(f"\nPrvih 5 swatcheva (Lab D50):")
    print(f"  {'#':>2}  {'Izmjereno (L a b)':>24}  {'Referentno (L a b)':>24}")
    for i in range(min(5, len(meas_lab))):
        ml = meas_lab[i]
        rl = ref_lab[i]
        print(
            f"  {i+1:>2}  {ml[0]:6.1f} {ml[1]:6.1f} {ml[2]:6.1f}          "
            f"{rl[0]:6.1f} {rl[1]:6.1f} {rl[2]:6.1f}"
        )
else:
    print("\n  ColorChecker NIJE pronađen.")
    print("  Mogući razlozi:")
    print("  - Karta nije vidljiva u cijelosti")
    print("  - Premalo kontrasta (premračna prostorija?)")
    print("  - Karta zauzima premalo/previše kadra")

# ── Debug slike ───────────────────────────────────────────────────────────────
print(f"\n[3] Vizualizacije (ako je detekcija uspjela):")
for fname in [
    "viz_01_contours.jpg",
    "viz_02_left_panel.jpg",
    "viz_03_right_panel.jpg",
    "viz_04_swatch_grid.jpg",
    "viz_05_comparison.jpg",
]:
    p = output_dir / fname
    if p.exists():
        print(f"  ✓ {p}")
    else:
        print(f"  ✗ {p}  (nije generirano)")

print("=" * 60)
