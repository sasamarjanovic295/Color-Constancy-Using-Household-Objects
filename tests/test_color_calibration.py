"""
Test color calibration on a single image.

Detects the SCK300 ColorChecker, applies a correction matrix, then
re-detects on the corrected image so you can compare before vs after.

Output structure:
    <output>/<image_stem>/colorchecker_before/   ← detection on original
    <output>/<image_stem>/colorchecker_after/    ← detection on corrected

Usage:
    python test_color_calibration.py <image_path>
    python test_color_calibration.py <image_path> --method poly2
    python test_color_calibration.py <image_path> --output results
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

from src.colorchecker import detect_and_measure
from src.color_calibration import (
    METHODS,
    adapt_d50_to_d65,
    apply_calibration,
    calibrate,
    srgb_to_xyz,
    xyz_to_srgb,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_color_calibration")

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="ColorChecker detection + calibration: before / after comparison",
)
parser.add_argument("image_path", help="Path to the input image")
parser.add_argument(
    "--method",
    choices=METHODS,
    default="affine",
    help="Correction method (default: affine)",
)
parser.add_argument(
    "--output",
    default="output",
    help="Root output directory (default: output)",
)
args = parser.parse_args()

# ── Load image ────────────────────────────────────────────────────────────────

img_path = Path(args.image_path)
if not img_path.exists():
    print(f"GREŠKA: datoteka ne postoji: {img_path}")
    sys.exit(1)

print(f"\n[1] Učitavam: {img_path}")
bgr = cv.imread(str(img_path))

# HEIF/HEVC (iPhone format) — OpenCV ne podržava, konvertiraj via sips
if bgr is None:
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

# ── Output dirs ───────────────────────────────────────────────────────────────

stem = img_path.stem
base_dir = Path(args.output) / stem
before_dir = base_dir / "colorchecker_before"
after_dir = base_dir / "colorchecker_after"

# ── BEFORE: detect on original ────────────────────────────────────────────────

print("\n[2] Detekcija ColorCheckera na ORIGINALNOJ slici ...")
result_before = detect_and_measure(
    img_srgb, log=True, save_steps=True, output_dir=before_dir,
)

if not result_before.detected:
    print(f"\n✗ ColorChecker NIJE pronađen: {result_before.failure_reason}")
    print("  Kalibracija nije moguća bez detekcije.")
    sys.exit(1)

print(f"    ΔE00 before — mean={result_before.delta_e_stats['mean']:.2f}  "
      f"median={result_before.delta_e_stats['median']:.2f}  "
      f"p95={result_before.delta_e_stats['p95']:.2f}")

# ── Apply correction ─────────────────────────────────────────────────────────
#
# Pipeline: sRGB → XYZ(D65) → calibrate → apply → XYZ(D65) → sRGB
# Reference swatch values are XYZ(D50) from darktable — adapt to D65 first.

method = args.method
print(f"\n[3] Primjenjujem korekciju: {method} ...")

image_xyz_d65 = srgb_to_xyz(img_srgb)  # sRGB gamma → XYZ(D65)

is_reference_based = method in ("linear", "affine", "poly2", "poly3_cheung")

if is_reference_based:
    measured_xyz_d65 = result_before.swatch_colors_xyz        # (48,3) XYZ D65
    reference_xyz_d65 = adapt_d50_to_d65(                     # (48,3) XYZ D50 → D65
        result_before.reference_colors_xyz
    )
    params = calibrate(method, measured_xyz_d65=measured_xyz_d65, reference_xyz_d65=reference_xyz_d65)
else:
    # gray_world, shades_of_gray — no reference swatches needed
    params = calibrate(method, image_xyz_d65=image_xyz_d65)

corrected_xyz_d65 = apply_calibration(image_xyz_d65, params)  # XYZ(D65) → XYZ(D65)
corrected_srgb = xyz_to_srgb(corrected_xyz_d65)              # XYZ(D65) → sRGB gamma

# Save corrected full image
base_dir.mkdir(parents=True, exist_ok=True)
corrected_bgr = cv.cvtColor(
    (np.clip(corrected_srgb, 0, 1) * 255 + 0.5).astype(np.uint8),
    cv.COLOR_RGB2BGR,
)
cv.imwrite(str(base_dir / f"corrected_{method}.jpg"), corrected_bgr)
print(f"    Korigirana slika: {base_dir / f'corrected_{method}.jpg'}")

# ── AFTER: detect on corrected ────────────────────────────────────────────────

print("\n[4] Detekcija ColorCheckera na KORIGIRANOJ slici ...")
result_after = detect_and_measure(
    corrected_srgb, log=True, save_steps=True, output_dir=after_dir,
)

if not result_after.detected:
    print(f"\n✗ ColorChecker NIJE pronađen na korigiranoj slici: "
          f"{result_after.failure_reason}")
    print("  Vizualizacije 'before' su sačuvane, ali 'after' nije moguć.")
    sys.exit(1)

print(f"    ΔE00 after  — mean={result_after.delta_e_stats['mean']:.2f}  "
      f"median={result_after.delta_e_stats['median']:.2f}  "
      f"p95={result_after.delta_e_stats['p95']:.2f}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"  Slika:    {img_path.name}")
print(f"  Metoda:   {method}")
print(f"  Output:   {base_dir}/")
print(f"  ΔE00 BEFORE — mean={result_before.delta_e_stats['mean']:.2f}  "
      f"median={result_before.delta_e_stats['median']:.2f}  "
      f"p95={result_before.delta_e_stats['p95']:.2f}")
print(f"  ΔE00 AFTER  — mean={result_after.delta_e_stats['mean']:.2f}  "
      f"median={result_after.delta_e_stats['median']:.2f}  "
      f"p95={result_after.delta_e_stats['p95']:.2f}")

improvement = result_before.delta_e_stats["mean"] - result_after.delta_e_stats["mean"]
print(f"  Poboljšanje (mean ΔE00): {improvement:+.2f}")
print("=" * 60)

print(f"\n  colorchecker_before/  → {before_dir}")
print(f"  colorchecker_after/   → {after_dir}")
print(f"  corrected_{method}.jpg")
