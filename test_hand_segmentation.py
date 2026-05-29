"""
Brzi test: detect_and_segment na jednoj slici.

Pokretanje:
    source env/bin/activate
    python test_hand_segmentation.py <putanja_do_slike>

    npr. python test_hand_segmentation.py data/train/IMG_hand.jpeg

Opcije:
    --no-view       Ne prikazuj OpenCV prozore
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

# ── Argumenti ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hand segmentation test")
parser.add_argument("image_path", help="Path to image")
parser.add_argument("--no-view", action="store_true", help="Skip OpenCV display")
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s  %(levelname)s  %(message)s",
)

# ── Učitaj sliku ─────────────────────────────────────────────────────────────
img_path = Path(args.image_path)
if not img_path.exists():
    print(f"GREŠKA: datoteka ne postoji: {img_path}")
    sys.exit(1)

print(f"\n[1] Učitavam: {img_path}")
bgr = cv.imread(str(img_path))

# HEIF/HEVC (iPhone format)
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

img_rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
print(f"    Rezolucija: {img_rgb.shape[1]} × {img_rgb.shape[0]} px")

# ── Detekcija i segmentacija ─────────────────────────────────────────────────
print("\n[2] Pokrećem detect_and_segment (log=True) ...")
from src.hand_segmentation import detect_and_segment, HandSegmentationResult

output_dir = Path("debug_output")
result = detect_and_segment(
    img_rgb,
    view=not args.no_view,
    log=True,
    save=True,
    output_dir=output_dir,
)

# ── Rezultati ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"detected        : {result.detected}")
print(f"failure_reason  : {result.failure_reason}")
print(f"n_hands         : {result.n_hands}")
print(f"handedness      : {result.handedness}")
print(f"confidence      : {result.confidence}")
print(f"hand_area_px    : {result.hand_area_px}")

if result.detected:
    total_px = img_rgb.shape[0] * img_rgb.shape[1]
    print(f"hand_area_%     : {result.hand_area_px / total_px * 100:.2f}%")
    print(f"landmarks shape : {result.landmarks.shape}")
    print(f"mask shape      : {result.mask.shape}")
    print(f"mask dtype      : {result.mask.dtype}")
    print(f"mask values     : min={result.mask.min()}, max={result.mask.max()}")

    # Ispis landmark koordinata
    print(f"\nLandmark koordinate (px):")
    names = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]
    for i, (name, pt) in enumerate(zip(names, result.landmarks)):
        marker = " *" if i in [4, 8, 12, 16, 20] else ""
        print(f"  {i:2d} {name:15s}  ({pt[0]:7.1f}, {pt[1]:7.1f}){marker}")
else:
    print("\n  Ruka NIJE pronađena.")
    print("  Mogući razlozi:")
    print("  - Ruka nije vidljiva")
    print("  - Premračna slika")
    print("  - Ruka zauzima premalo kadra")

# ── Debug slike ──────────────────────────────────────────────────────────────
print(f"\n[3] Vizualizacije:")
for fname in [
    "hand_viz_01_landmarks.jpg",
    "hand_viz_02_M1_skeleton.jpg",
    "hand_viz_03_M2_dilated.jpg",
    "hand_viz_04_skin_sample.jpg",
    "hand_viz_05_Mab.jpg",
    "hand_viz_06_M3.jpg",
    "hand_viz_07_output_mask.jpg",
    "hand_viz_08_overlay.jpg",
]:
    p = output_dir / fname
    if p.exists():
        print(f"  ✓ {p}")
    else:
        print(f"  ✗ {p}  (nije generirano)")

# ── Čekaj na tipku ako je view uključen ──────────────────────────────────────
if not args.no_view and result.detected:
    print("\n[4] Pritisnite bilo koju tipku za zatvaranje prozora ...")
    cv.waitKey(0)
    cv.destroyAllWindows()

print("=" * 60)
