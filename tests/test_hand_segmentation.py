"""
Brzi test: detect_and_segment na jednoj ili više slika.

Pokretanje:
    source env/bin/activate
    python test_hand_segmentation.py data/train/IMG_2900.jpeg
    python test_hand_segmentation.py data/eval/IMG_2629.jpeg data/train/IMG_2900.jpeg
    python test_hand_segmentation.py data/eval/*.jpeg --no-view --mode skin

Output slike se čuvaju u debug_output/hand_segmentation/<ime_slike>/.
"""

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

# ── Argumenti ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hand segmentation test")
parser.add_argument("image_paths", nargs="+", help="Path(s) to image(s)")
parser.add_argument("--no-view", action="store_true", help="Skip OpenCV display")
parser.add_argument(
    "--mode",
    choices=["hand", "skin"],
    default="skin",
    help="Segmentation mode (default: skin)",
)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s  %(levelname)s  %(message)s",
)

from _old_srcipts.hand_segmentation import detect_and_segment  # noqa: E402


def load_image(img_path: Path) -> np.ndarray | None:
    """Load image, converting HEIF via sips if needed."""
    bgr = cv.imread(str(img_path))
    if bgr is None:
        tmp = Path(tempfile.mktemp(suffix=".jpeg"))
        ret = subprocess.run(
            ["sips", "-s", "format", "jpeg", str(img_path), "--out", str(tmp)],
            capture_output=True,
        )
        if ret.returncode == 0 and tmp.exists():
            bgr = cv.imread(str(tmp))
        tmp.unlink(missing_ok=True)
    if bgr is None:
        return None
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0


# ── Obrada slika ─────────────────────────────────────────────────────────────
paths = [Path(p) for p in args.image_paths]
n_total = len(paths)
n_ok = 0
n_fail = 0
results_summary = []

for i, img_path in enumerate(paths, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{n_total}] {img_path.name}")
    print(f"{'='*60}")

    if not img_path.exists():
        print(f"  GREŠKA: datoteka ne postoji")
        n_fail += 1
        results_summary.append((img_path.stem, "FILE NOT FOUND", "—"))
        continue

    img_rgb = load_image(img_path)
    if img_rgb is None:
        print(f"  GREŠKA: ne mogu učitati sliku")
        n_fail += 1
        results_summary.append((img_path.stem, "LOAD FAILED", "—"))
        continue

    print(f"  Rezolucija: {img_rgb.shape[1]} × {img_rgb.shape[0]} px")

    output_dir = Path("debug_output") / "hand_segmentation" / img_path.stem

    result = detect_and_segment(
        img_rgb,
        mode=args.mode,
        view=not args.no_view,
        log=False,
        save=True,
        output_dir=output_dir,
    )

    if result.detected:
        total_px = img_rgb.shape[0] * img_rgb.shape[1]
        pct = result.hand_area_px / total_px * 100
        print(f"  ✓ {result.handedness} hand  conf={result.confidence:.3f}"
              f"  area={pct:.1f}%  mode={result.mode}")
        print(f"    → {output_dir}/")
        n_ok += 1
        results_summary.append((img_path.stem, "OK", f"{pct:.1f}%"))
    else:
        print(f"  ✗ {result.failure_reason}")
        n_fail += 1
        results_summary.append((img_path.stem, "FAIL", result.failure_reason))

    if not args.no_view and result.detected:
        cv.waitKey(1)

# ── Sumarni rezultat ─────────────────────────────────────────────────────────
if n_total > 1:
    print(f"\n{'='*60}")
    print(f"REZULTATI: {n_ok}/{n_total} OK, {n_fail} FAIL")
    print(f"{'='*60}")
    for stem, status, detail in results_summary:
        print(f"  {stem:30s}  {status:6s}  {detail}")

if not args.no_view and n_ok > 0:
    print("\nPritisnite bilo koju tipku za zatvaranje prozora ...")
    cv.waitKey(0)
    cv.destroyAllWindows()
