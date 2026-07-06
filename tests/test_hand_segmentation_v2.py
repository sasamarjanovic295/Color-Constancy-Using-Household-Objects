"""
Test hand_segmentation_v2 (MediaPipe multiclass body-skin) on one or more images.

Usage:
    python test_hand_segmentation_v2.py data/train/IMG_2900.jpeg
    python test_hand_segmentation_v2.py data/train/IMG_2900.jpeg data/eval/IMG_2629.jpeg --no-view
"""

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser(description="Hand segmentation v2 test")
parser.add_argument("image_paths", nargs="+", help="Image path(s)")
parser.add_argument("--no-view", action="store_true")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Skin confidence threshold (default: 0.5)")
parser.add_argument("--no-landmarks", action="store_true",
                    help="Skip MediaPipe Hands landmark detection")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(name)s  %(levelname)s  %(message)s")

from src.hand_segmentation_v2 import detect_and_segment  # noqa: E402


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
n_total = len(paths)
n_ok = 0
n_fail = 0
summary = []

for i, img_path in enumerate(paths, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{n_total}] {img_path.name}")
    print(f"{'='*60}")

    if not img_path.exists():
        print("  GREŠKA: ne postoji")
        n_fail += 1
        summary.append((img_path.stem, "NOT FOUND", "—"))
        continue

    img_srgb = load_image(img_path)
    if img_srgb is None:
        print("  GREŠKA: ne mogu učitati")
        n_fail += 1
        summary.append((img_path.stem, "LOAD FAIL", "—"))
        continue

    print(f"  Rezolucija: {img_srgb.shape[1]} × {img_srgb.shape[0]} px")

    out_dir = Path("debug_output") / "hand_segmentation_v2" / img_path.stem

    result = detect_and_segment(
        img_srgb,
        detect_landmarks=not args.no_landmarks,
        skin_threshold=args.threshold,
        show_steps=not args.no_view,
        save_steps=True,
        log=True,
        output_dir=out_dir,
    )

    if result.detected:
        pct = result.skin_area_px / (img_srgb.shape[0] * img_srgb.shape[1]) * 100
        lm_info = ""
        if result.landmarks is not None:
            lm_info = f"  {result.handedness} conf={result.confidence:.3f}"
        print(f"  ✓ skin={pct:.1f}%{lm_info}")
        print(f"    → {out_dir}/")
        n_ok += 1
        summary.append((img_path.stem, "OK", f"{pct:.1f}%"))
    else:
        print(f"  ✗ {result.failure_reason}")
        n_fail += 1
        summary.append((img_path.stem, "FAIL", result.failure_reason or ""))

if n_total > 1:
    print(f"\n{'='*60}")
    print(f"REZULTATI: {n_ok}/{n_total} OK, {n_fail} FAIL")
    print(f"{'='*60}")
    for stem, status, detail in summary:
        print(f"  {stem:30s}  {status:6s}  {detail}")

if not args.no_view and n_ok > 0:
    print("\nPritisnite tipku za zatvaranje ...")
    cv.waitKey(0)
    cv.destroyAllWindows()
