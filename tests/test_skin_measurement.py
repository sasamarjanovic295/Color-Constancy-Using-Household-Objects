"""
Test skin_measurement on images — runs segmentation v2 then measurement.

Usage:
    python test_skin_measurement.py data/train/IMG_2900.jpeg
    python test_skin_measurement.py data/train/IMG_2900.jpeg data/eval/IMG_2629.jpeg --no-view
"""

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser(description="Skin measurement test")
parser.add_argument("image_paths", nargs="+", help="Image path(s)")
parser.add_argument("--no-view", action="store_true")
parser.add_argument("--erode", type=int, default=5, help="Erosion kernel size (0=skip)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(name)s  %(levelname)s  %(message)s")

from src.hand_segmentation_v2 import detect_and_segment  # noqa: E402
from src.skin_measurement import measure_skin_tone  # noqa: E402


def load_image(path: Path) -> np.ndarray | None:
    bgr = cv.imread(str(path))
    if bgr is None:
        tmp = Path(tempfile.mktemp(suffix=".jpeg"))
        ret = subprocess.run(
            ["sips", "-s", "format", "jpeg", str(path), "--out", str(tmp)],
            capture_output=True,
        )
        if ret.returncode == 0 and tmp.exists():
            bgr = cv.imread(str(tmp))
        tmp.unlink(missing_ok=True)
    if bgr is None:
        return None
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0


paths = [Path(p) for p in args.image_paths]
summary = []

for i, img_path in enumerate(paths, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(paths)}] {img_path.name}")
    print(f"{'='*60}")

    if not img_path.exists():
        print("  GREŠKA: ne postoji")
        summary.append((img_path.stem, "NOT FOUND", ""))
        continue

    img_srgb = load_image(img_path)
    if img_srgb is None:
        print("  GREŠKA: ne mogu učitati")
        summary.append((img_path.stem, "LOAD FAIL", ""))
        continue

    # Step 1: Segmentation
    seg = detect_and_segment(img_srgb, detect_landmarks=True, log=False, save_steps=False)
    if not seg.detected:
        print(f"  ✗ Segmentation failed: {seg.failure_reason}")
        summary.append((img_path.stem, "SEG FAIL", seg.failure_reason or ""))
        continue

    pct_seg = seg.skin_area_px / (img_srgb.shape[0] * img_srgb.shape[1]) * 100
    hand_info = f" ({seg.handedness})" if seg.handedness else ""
    print(f"  Segmentation: {pct_seg:.1f}% skin{hand_info}")

    # Step 2: Measurement
    out_dir = Path("debug_output") / "skin_measurement" / img_path.stem
    result = measure_skin_tone(
        img_srgb, seg.mask,
        erode_size=args.erode,
        show_steps=not args.no_view,
        save_steps=True,
        log=True,
        output_dir=out_dir,
    )

    if result is None:
        print("  ✗ Measurement failed")
        summary.append((img_path.stem, "MEAS FAIL", ""))
        continue

    print(f"  ✓ ITA={result.ITA:.1f}°  {result.chardon_category} (FP {result.fitzpatrick_type})")
    print(f"    L*={result.L_median:.1f}  a*={result.a_median:.1f}  b*={result.b_median:.1f}")
    print(f"    Pixels: {result.n_pixels_filtered:,} / {result.n_pixels_total:,}"
          f" ({result.filter_ratio*100:.0f}% kept)")
    print(f"    → {out_dir}/")
    summary.append((
        img_path.stem, "OK",
        f"ITA={result.ITA:.1f}° {result.chardon_category}"
    ))

if len(paths) > 1:
    print(f"\n{'='*60}")
    print(f"REZULTATI")
    print(f"{'='*60}")
    for stem, status, detail in summary:
        print(f"  {stem:30s}  {status:6s}  {detail}")

if not args.no_view:
    print("\nPritisnite tipku za zatvaranje ...")
    cv.waitKey(0)
    cv.destroyAllWindows()
