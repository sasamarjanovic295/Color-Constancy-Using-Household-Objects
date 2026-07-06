"""
Validation script for banknote detection.

Runs the SIFT + RANSAC banknote detection pipeline on real images from
the dataset, logging every step and saving visualisation artefacts to
``debug_output/banknote_detection/<image_stem>/<side>/``.

Usage::

    # Run on all annotated images (default)
    python test_banknote_detection.py --log

    # Single image with display
    python test_banknote_detection.py --image IMG_3898.HEIC --log --show-steps

    # Override side mapping (annotations say "front", refs use "single"/"double")
    python test_banknote_detection.py --side-map front=single back=double

    # Try both sides and pick best
    python test_banknote_detection.py --try-both --log
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from src.banknote_detection import (
    BanknoteResult,
    detect_and_measure_banknote,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_banknote_detection")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_REF_DIR = Path("data/ref")
_DEBUG_OUTPUT_DIR = Path("debug_output") / "banknote_detection"

# Annotation side → reference side fallback mapping.
# Annotations use "front"/"back"; reference images use "single"/"double".
_DEFAULT_SIDE_MAP: dict[str, str] = {
    "front": "single",
    "back": "double",
}


# ---------------------------------------------------------------------------
# Image loading (HEIF-aware, adapted from calibrate_dataset.py)
# ---------------------------------------------------------------------------


def load_image_srgb(path: Path) -> np.ndarray | None:
    """Load an image as float32 sRGB [0, 1], shape (H, W, 3).

    Handles HEIC/HEIF files via macOS ``sips`` conversion.
    Returns ``None`` on failure.
    """
    bgr = cv.imread(str(path))

    # HEIF fallback (macOS only)
    if bgr is None and path.suffix.lower() in (".heic", ".heif"):
        tmp = Path(tempfile.mktemp(suffix=".jpeg"))
        try:
            ret = subprocess.run(
                ["sips", "-s", "format", "jpeg", str(path), "--out", str(tmp)],
                capture_output=True,
                timeout=30,
            )
            if ret.returncode == 0 and tmp.exists():
                bgr = cv.imread(str(tmp))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        finally:
            tmp.unlink(missing_ok=True)

    if bgr is None:
        return None

    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------


def _log_result(
    image_name: str,
    result: BanknoteResult,
    elapsed: float,
) -> None:
    """Print a structured summary of one detection run."""
    sep = "─" * 60
    logger.info(sep)
    logger.info("Image: %s", image_name)
    logger.info("  Denomination: %s€, Side: %s", result.denomination, result.side)
    logger.info("  Detected: %s", result.detected)

    if not result.detected:
        logger.info("  Failure: %s", result.failure_reason)
        logger.info("  Elapsed: %.2fs", elapsed)
        logger.info(sep)
        return

    det = result.detection_result
    if det is not None:
        logger.info("  SIFT keypoints: scene=%d, ref=%d",
                     det.n_keypoints_scene, det.n_keypoints_ref)
        logger.info("  Matches: raw=%d → Lowe=%d → inliers=%d (%.1f%%)",
                     det.n_matches_raw, det.n_matches_lowe,
                     det.n_inliers, 100.0 * det.inlier_ratio)
        logger.info("  Reproj RMSE: %.2f px", det.reproj_rmse)
        logger.info("  Confidence: %.3f", det.confidence)
        if det.warped_scene is not None:
            logger.info("  Warped dims: %dx%d",
                         det.warped_scene.shape[1], det.warped_scene.shape[0])

    logger.info("  Paired samples: %d", result.n_samples)
    logger.info("  Elapsed: %.2fs", elapsed)
    logger.info(sep)


def _log_comparison(
    image_name: str,
    results: dict[str, BanknoteResult],
) -> None:
    """Compare results from multiple sides for the same image."""
    logger.info("  Comparison for %s:", image_name)
    for side, r in results.items():
        if r.detected:
            logger.info(
                "    %s: OK  inliers=%d  RMSE=%.2f  conf=%.3f  samples=%d",
                side, r.n_inliers, r.reproj_rmse, r.confidence, r.n_samples,
            )
        else:
            logger.info("    %s: FAIL  (%s)", side, r.failure_reason)


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------


def run_detection_test(
    image_name: str,
    image_srgb: np.ndarray,
    denomination: int,
    sides: list[str],
    ref_dir: Path,
    *,
    cell_size: int = 32,
    show_steps: bool = False,
    log_pipeline: bool = False,
) -> dict[str, BanknoteResult]:
    """Run detection for one image against one or more reference sides.

    Visualisations are always saved to ``debug_output/banknote_detection/``.

    Returns a dict mapping side name → BanknoteResult.
    """
    stem = Path(image_name).stem
    results: dict[str, BanknoteResult] = {}

    for side in sides:
        side_out = _DEBUG_OUTPUT_DIR / stem / side

        t0 = time.monotonic()
        result = detect_and_measure_banknote(
            image_srgb,
            denomination=denomination,
            side=side,
            ref_dir=ref_dir,
            cell_size=cell_size,
            show_steps=show_steps,
            save_steps=True,
            log=log_pipeline,
            output_dir=side_out,
        )
        elapsed = time.monotonic() - t0

        _log_result(f"{image_name} ({side})", result, elapsed)
        results[side] = result

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate banknote detection on dataset images",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="*",
        default=None,
        help="Image filename(s) to test (default: all annotated images)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Dataset root containing annotations.json and images/",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=_DEFAULT_REF_DIR,
        help="Reference banknote directory",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=32,
        help="Grid cell size for sampling (default: 32)",
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Display each step in OpenCV windows",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable per-step pipeline logging",
    )
    parser.add_argument(
        "--try-both",
        action="store_true",
        help="Try both single and double sides, pick best match",
    )
    parser.add_argument(
        "--side-map",
        nargs="*",
        metavar="FROM=TO",
        default=None,
        help="Side mapping overrides, e.g. front=single back=double",
    )
    args = parser.parse_args()

    # ── Parse side mapping ────────────────────────────────────────────

    side_map = dict(_DEFAULT_SIDE_MAP)
    if args.side_map:
        for mapping in args.side_map:
            if "=" not in mapping:
                logger.error("Invalid side mapping: %r (expected FROM=TO)", mapping)
                sys.exit(1)
            k, v = mapping.split("=", 1)
            side_map[k] = v

    # ── Load annotations ──────────────────────────────────────────────

    ann_path = args.data_dir / "annotations.json"
    if not ann_path.exists():
        logger.error("Annotations not found: %s", ann_path)
        sys.exit(1)

    with open(ann_path) as f:
        data = json.load(f)
    annotations: dict = data.get("images", data)

    images_dir = args.data_dir / "images"

    # ── Select images to test ─────────────────────────────────────────

    if args.image:
        image_names = args.image
    else:
        image_names = list(annotations.keys())

    if not image_names:
        logger.error("No images to test")
        sys.exit(1)

    # ── Run tests ─────────────────────────────────────────────────────

    logger.info("=" * 60)
    logger.info("BANKNOTE DETECTION VALIDATION")
    logger.info("  Images: %d", len(image_names))
    logger.info("  Ref dir: %s", args.ref_dir)
    logger.info("  Output:  %s", _DEBUG_OUTPUT_DIR)
    logger.info("  Cell size: %d", args.cell_size)
    logger.info("  Try both sides: %s", args.try_both)
    logger.info("  Side mapping: %s", side_map)
    logger.info("=" * 60)

    n_success = 0
    n_fail = 0
    all_results: list[dict] = []

    for image_name in image_names:
        ann = annotations.get(image_name)
        if ann is None:
            logger.warning("No annotation for %s — skipping", image_name)
            n_fail += 1
            continue

        denomination = ann.get("denomination")
        ann_side = ann.get("side", "")

        if denomination is None:
            logger.warning("No denomination in annotation for %s — skipping", image_name)
            n_fail += 1
            continue

        # ── Determine which sides to try ──────────────────────────
        if args.try_both:
            sides = ["single", "double"]
        else:
            mapped_side = side_map.get(ann_side, ann_side)
            if mapped_side not in ("single", "double"):
                logger.warning(
                    "Unknown side %r (mapped from %r) for %s — trying both",
                    mapped_side, ann_side, image_name,
                )
                sides = ["single", "double"]
            else:
                sides = [mapped_side]

        # ── Load image ────────────────────────────────────────────
        image_path = images_dir / image_name
        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            n_fail += 1
            continue

        logger.info("Loading %s ...", image_name)
        image_srgb = load_image_srgb(image_path)
        if image_srgb is None:
            logger.error("Failed to decode: %s", image_path)
            n_fail += 1
            continue

        logger.info("  Loaded: %dx%d", image_srgb.shape[1], image_srgb.shape[0])

        results = run_detection_test(
            image_name,
            image_srgb,
            denomination,
            sides,
            ref_dir=args.ref_dir,
            cell_size=args.cell_size,
            show_steps=args.show_steps,
            log_pipeline=args.log,
        )

        # ── Pick best result (when trying both sides) ─────────────
        if len(results) > 1:
            _log_comparison(image_name, results)

        best_side = None
        best_conf = -1.0
        for side, r in results.items():
            if r.detected and (r.confidence or 0.0) > best_conf:
                best_conf = r.confidence or 0.0
                best_side = side

        if best_side is not None:
            n_success += 1
            best = results[best_side]
            logger.info(
                "  BEST: %s  (conf=%.3f, inliers=%d, samples=%d)",
                best_side, best.confidence, best.n_inliers, best.n_samples,
            )
        else:
            n_fail += 1
            logger.info("  NO MATCH for %s with any side", image_name)

        all_results.append({
            "image": image_name,
            "denomination": denomination,
            "annotation_side": ann_side,
            "best_side": best_side,
            "best_confidence": best_conf if best_side else None,
            "sides_tried": {
                s: {
                    "detected": r.detected,
                    "confidence": r.confidence,
                    "n_inliers": r.n_inliers,
                    "n_samples": r.n_samples,
                    "failure_reason": r.failure_reason,
                    "reproj_rmse": r.reproj_rmse,
                }
                for s, r in results.items()
            },
        })

    # ── Summary ───────────────────────────────────────────────────────

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Total:   %d", len(image_names))
    logger.info("  Success: %d", n_success)
    logger.info("  Failed:  %d", n_fail)
    if n_success > 0:
        avg_conf = np.mean([
            r["best_confidence"] for r in all_results
            if r["best_confidence"] is not None
        ])
        logger.info("  Avg confidence: %.3f", avg_conf)
    logger.info("=" * 60)

    # ── Save summary JSON ─────────────────────────────────────────────

    _DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = _DEBUG_OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Summary written to %s", summary_path)

    # ── Wait for key if showing windows ───────────────────────────────

    if args.show_steps:
        logger.info("Press any key to close windows...")
        cv.waitKey(0)
        cv.destroyAllWindows()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
