"""
Test MediaPipe multiclass segmenter — body-skin mask extraction.

Classes: 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others

Usage:
    python test_body_skin_segmentation.py data/train/IMG_2900.jpeg
    python test_body_skin_segmentation.py data/train/IMG_2900.jpeg data/eval/IMG_2629.jpeg
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    ImageSegmenter,
    ImageSegmenterOptions,
)

# ── Config ───────────────────────────────────────────────────────────────────
_MODEL_PATH = Path("models/selfie_multiclass_256x256.tflite")
_CLASS_NAMES = ["background", "hair", "body-skin", "face-skin", "clothes", "others"]
_SKIN_INDEX = 2      # body-skin
_FACE_INDEX = 3      # face-skin

# Overlay colours (BGR)
_COLORS = {
    "body-skin": (0, 200, 0),    # green
    "face-skin": (0, 200, 200),  # yellow
    "all-skin":  (0, 200, 0),    # green
}

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Body-skin segmentation test")
parser.add_argument("image_paths", nargs="+", help="Image path(s)")
parser.add_argument("--no-view", action="store_true")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Confidence threshold for skin class (default: 0.5)")
args = parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────
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
    return bgr


def overlay_mask(bgr: np.ndarray, mask: np.ndarray,
                 color: tuple = (0, 200, 0), alpha: float = 0.4) -> np.ndarray:
    vis = bgr.copy()
    overlay = vis.copy()
    overlay[mask > 0] = color
    cv.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(vis, contours, -1, color, 2)
    return vis


# ── Segmenter ────────────────────────────────────────────────────────────────
print(f"Loading model: {_MODEL_PATH}")
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
    output_category_mask=False,
    output_confidence_masks=True,
)
segmenter = ImageSegmenter.create_from_options(options)

# ── Process images ───────────────────────────────────────────────────────────
paths = [Path(p) for p in args.image_paths]

for i, img_path in enumerate(paths, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(paths)}] {img_path.name}")
    print(f"{'='*60}")

    if not img_path.exists():
        print("  GREŠKA: ne postoji")
        continue

    bgr = load_image(img_path)
    if bgr is None:
        print("  GREŠKA: ne mogu učitati")
        continue

    h, w = bgr.shape[:2]
    print(f"  Rezolucija: {w} × {h} px")

    # Convert to MediaPipe Image (RGB)
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Segment
    result = segmenter.segment(mp_image)
    confidence_masks = result.confidence_masks

    # Extract body-skin and face-skin confidence maps
    body_skin_conf = confidence_masks[_SKIN_INDEX].numpy_view().squeeze()  # (H, W)
    face_skin_conf = confidence_masks[_FACE_INDEX].numpy_view().squeeze()

    # Binary masks at threshold
    body_skin_mask = (body_skin_conf >= args.threshold).astype(np.uint8) * 255
    face_skin_mask = (face_skin_conf >= args.threshold).astype(np.uint8) * 255
    all_skin_mask = ((body_skin_conf >= args.threshold)
                     | (face_skin_conf >= args.threshold)).astype(np.uint8) * 255

    # Stats
    body_px = np.count_nonzero(body_skin_mask)
    face_px = np.count_nonzero(face_skin_mask)
    total_px = h * w
    print(f"  body-skin: {body_px:,} px ({body_px/total_px*100:.1f}%)")
    print(f"  face-skin: {face_px:,} px ({face_px/total_px*100:.1f}%)")

    # Argmax class map for full viz
    all_confs = np.stack([m.numpy_view().squeeze() for m in confidence_masks], axis=-1)
    class_map = np.argmax(all_confs, axis=-1)  # (H, W)

    # Per-class stats
    for ci, name in enumerate(_CLASS_NAMES):
        count = np.count_nonzero(class_map == ci)
        if count > 0:
            print(f"    [{ci}] {name:12s}: {count/total_px*100:5.1f}%")

    # ── Save outputs ─────────────────────────────────────────────────────
    out_dir = Path("debug_output") / "body_skin_segmentation" / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Body-skin mask (binary)
    cv.imwrite(str(out_dir / "01_body_skin_mask.jpg"), body_skin_mask)

    # 2. Body-skin confidence heatmap
    heatmap = (body_skin_conf * 255).astype(np.uint8)
    heatmap_color = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    cv.imwrite(str(out_dir / "02_body_skin_heatmap.jpg"), heatmap_color)

    # 3. Body-skin overlay
    overlay_body = overlay_mask(bgr, body_skin_mask, _COLORS["body-skin"])
    cv.imwrite(str(out_dir / "03_body_skin_overlay.jpg"), overlay_body)

    # 4. All-skin overlay (body + face)
    overlay_all = overlay_mask(bgr, all_skin_mask, _COLORS["all-skin"])
    cv.imwrite(str(out_dir / "04_all_skin_overlay.jpg"), overlay_all)

    # 5. Full class map visualization
    class_colors = np.array([
        [0, 0, 0],        # background — black
        [139, 69, 19],    # hair — brown
        [0, 200, 0],      # body-skin — green
        [0, 200, 200],    # face-skin — yellow
        [200, 0, 0],      # clothes — blue
        [200, 0, 200],    # others — magenta
    ], dtype=np.uint8)
    class_viz = class_colors[class_map]
    class_viz_bgr = cv.cvtColor(class_viz, cv.COLOR_RGB2BGR)
    cv.imwrite(str(out_dir / "05_class_map.jpg"), class_viz_bgr)

    print(f"  → {out_dir}/")

    if not args.no_view:
        cv.imshow(f"Body Skin — {img_path.name}", overlay_body)
        cv.waitKey(1)

segmenter.close()

if not args.no_view:
    print("\nPritisnite tipku za zatvaranje ...")
    cv.waitKey(0)
    cv.destroyAllWindows()
