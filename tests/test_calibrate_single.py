"""
Smoke test for calibrate_single — runs on a real image from CLI.

Usage:
    python test_calibrate_single.py <image_path>
    python test_calibrate_single.py <image_path> --methods affine poly2
    python test_calibrate_single.py <image_path> --save
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

from calibrate_single import calibrate_image
from src.evaluation import write_results_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

parser = argparse.ArgumentParser(description="Smoke test for calibrate_image")
parser.add_argument("image_path", help="Path to image")
parser.add_argument("--methods", nargs="*", default=None,
                    help="Methods to test (default: all)")
parser.add_argument("--save", action="store_true", help="Save debug steps")
parser.add_argument("--output", default="output/calibrate_single_test")
args = parser.parse_args()

# ── Load image ────────────────────────────────────────────────────────────────

img_path = Path(args.image_path)
if not img_path.exists():
    print(f"GREŠKA: {img_path} ne postoji")
    sys.exit(1)

bgr = cv.imread(str(img_path))
if bgr is None:
    tmp = Path(tempfile.mktemp(suffix=".jpeg"))
    subprocess.run(
        ["sips", "-s", "format", "jpeg", str(img_path), "--out", str(tmp)],
        capture_output=True,
    )
    bgr = cv.imread(str(tmp))
    tmp.unlink(missing_ok=True)

if bgr is None:
    print("GREŠKA: ne mogu učitati sliku")
    sys.exit(1)

image_srgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
print(f"Slika: {img_path.name} ({image_srgb.shape[1]}×{image_srgb.shape[0]})")

# ── Run pipeline ──────────────────────────────────────────────────────────────

methods = args.methods if args.methods else "all"
out_dir = Path(args.output) / img_path.stem

results = calibrate_image(
    image_srgb,
    methods=methods,
    image_path=str(img_path),
    lighting_id="L1",
    person_id=1,
    hand="left",
    split="train",
    show_steps=False,
    save_steps=args.save,
    log=True,
    output_dir=out_dir if args.save else None,
)

# ── Print results ─────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"{'Method':<18s} {'Ref':>4s} {'OK':>3s} {'ΔE before':>10s} {'ΔE after':>10s} "
      f"{'Δ':>7s} {'ITA bef':>8s} {'ITA aft':>8s}")
print(f"{'-'*70}")

for r in results:
    de_b = f"{r.cc_de_mean_before:.2f}" if r.cc_de_mean_before is not None else "—"
    de_a = f"{r.cc_de_mean_after:.2f}" if r.cc_de_mean_after is not None else "—"
    if r.cc_de_mean_before is not None and r.cc_de_mean_after is not None:
        delta = f"{r.cc_de_mean_before - r.cc_de_mean_after:+.2f}"
    else:
        delta = "—"
    ita_b = f"{r.ITA_before:.1f}°" if r.ITA_before is not None else "—"
    ita_a = f"{r.ITA_after:.1f}°" if r.ITA_after is not None else "—"
    ref = r.reference_object[:2].upper() if r.reference_object != "none" else "—"
    ok = "✓" if r.success else "✗"

    print(f"{r.correction_method:<18s} {ref:>4s} {ok:>3s} {de_b:>10s} {de_a:>10s} "
          f"{delta:>7s} {ita_b:>8s} {ita_a:>8s}")

    if not r.success:
        print(f"  └─ {r.failure_reason}")

print(f"{'='*70}")
print(f"Ukupno: {len(results)} rezultata")

# ── Save CSV ──────────────────────────────────────────────────────────────────

csv_path = out_dir / "results.csv"
write_results_csv(results, csv_path)
print(f"CSV: {csv_path}")
