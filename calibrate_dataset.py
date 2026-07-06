"""
Batch calibration — iterate a dataset, calibrate every image, write results.

Reads ``annotations.json``, loads each image, runs
:func:`calibrate_single.calibrate_image`, and accumulates results into a
single ``results.csv``.  Supports ``--resume`` to continue an interrupted
run with granular per-method tracking.

Usage::

    # Run all images, all methods
    python calibrate_dataset.py --data-dir data/ --output results/exp_01

    # Only eval split (L2), only affine
    python calibrate_dataset.py --data-dir data/ --output results/exp_02 \\
        --split eval --methods affine

    # Resume — keeps existing method rows, runs only missing ones
    python calibrate_dataset.py --data-dir data/ --output results/exp_01 --resume

    # Do not save final corrected images
    python calibrate_dataset.py --data-dir data/ --output results/exp_01 --no-save-corrected

Output structure::

    <output>/
    ├── results.csv              ← one row per image × method
    ├── run_info.json            ← reproducibility metadata
    └── <image_stem>/            ← per-image outputs (when --save-corrected)
        ├── affine_colorchecker/
        │   └── corrected.jpg
        ├── affine_banknote/
        │   └── corrected.jpg
        └── ...
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

from calibrate_single import calibrate_image
from src.evaluation import ImageResult, read_results_csv, write_results_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("calibrate_dataset")


# ---------------------------------------------------------------------------
# Train / eval split by lighting condition
# ---------------------------------------------------------------------------

HELD_OUT_LIGHTING: str = "L2"
EVAL_LIGHTINGS: frozenset[str] = frozenset({HELD_OUT_LIGHTING})
TRAIN_LIGHTINGS: frozenset[str] = frozenset({"L1", "L3", "L4", "L5", "L6"})

# ---------------------------------------------------------------------------
# Image loading (HEIF-aware)
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

    image_srgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image_srgb


# ---------------------------------------------------------------------------
# Annotations loading
# ---------------------------------------------------------------------------


def load_annotations(path: Path) -> dict:
    """Load annotations.json and return a ``{image_name: annotation}`` dict.

    Supports three schemas:
    - v3 (current): flat list of objects with ``"image"`` key
    - v2: ``{"metadata": ..., "images": {...}}``
    - v1: flat ``{image_name: {...}}`` dict
    """
    with open(path) as f:
        data = json.load(f)

    # v3: list of dicts with "image" field
    if isinstance(data, list):
        return {entry["image"]: entry for entry in data if "image" in entry}

    if "images" in data:
        return data["images"]

    # Legacy format: entire file is the images dict
    return data


def _normalize_banknote_side(raw_side: str | None) -> str | None:
    """Map annotation ``banknote_side`` values to detection API ``side``.

    Annotation uses ``"single_number"`` / ``"double_number"``;
    detection API expects ``"single"`` / ``"double"``.
    """
    if raw_side is None:
        return None
    if "single" in raw_side:
        return "single"
    if "double" in raw_side:
        return "double"
    return raw_side


# ---------------------------------------------------------------------------
# Run metadata (for reproducibility)
# ---------------------------------------------------------------------------


def _build_run_info(args: argparse.Namespace, n_images: int) -> dict:
    """Collect environment and parameter metadata for reproducibility."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output),
        "split_filter": args.split,
        "methods": args.methods,
        "n_images": n_images,
        "resume": args.resume,
        "save_corrected": not args.no_save_corrected,
        "cell_size": args.cell_size,
        "argv": sys.argv,
    }


def _process_one_image(
    image_name: str,
    annotations: dict,
    images_dir: Path,
    ref_dir: Path,
    methods: str | list[str],
    cell_size: int,
    log: bool,
    save_corrected: bool = False,
    output_dir: Path | None = None,
) -> list[ImageResult]:
    """Load + calibrate a single image.  Pickle-safe for multiprocessing."""
    stem = Path(image_name).stem
    ann = annotations[image_name]

    ann_lighting = ann.get("lighting_id")
    ann_person = ann.get("person_id")
    ann_hand = ann.get("hand")
    ann_hand_side = ann.get("hand_side")
    ann_split = "eval" if ann_lighting in EVAL_LIGHTINGS else "train"
    ann_denom_raw = ann.get("denomination")
    ann_denom = int(ann_denom_raw) if ann_denom_raw is not None else None
    ann_banknote_side = ann.get("banknote_side")
    ann_side = _normalize_banknote_side(ann_banknote_side)

    meta = dict(
        image_path=image_name, image_stem=stem,
        lighting_id=ann_lighting, person_id=ann_person,
        hand=ann_hand, hand_side=ann_hand_side, split=ann_split,
        denomination=ann_denom, banknote_side=ann_banknote_side,
    )

    image_path = images_dir / image_name
    if not image_path.exists():
        return [ImageResult(**meta, success=False,
                            failure_reason="load: file not found")]

    image_srgb = load_image_srgb(image_path)
    if image_srgb is None:
        return [ImageResult(**meta, success=False,
                            failure_reason="load: decode failed")]

    try:
        return calibrate_image(
            image_srgb,
            methods=methods,
            image_path=image_name,
            lighting_id=ann_lighting,
            person_id=ann_person,
            hand=ann_hand,
            hand_side=ann_hand_side,
            split=ann_split,
            denomination=ann_denom,
            banknote_side=ann_banknote_side,
            side=ann_side,
            ref_dir=ref_dir,
            cell_size=cell_size,
            show_steps=False,
            save_steps=False,
            save_corrected=save_corrected,
            log=log,
            output_dir=output_dir / stem if output_dir else None,
        )
    except Exception as exc:
        logger.error("EXCEPTION on %s: %s", image_name, exc)
        return [ImageResult(**meta, success=False,
                            failure_reason=f"exception: {exc}")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch calibration — run pipeline on entire dataset",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root (contains annotations.json and images/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results.csv and per-image debug",
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval", "all"],
        default="all",
        help="Process only this split: train (L1,L3-L6), eval (L2), or all (default)",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Correction methods to run (default: all)",
    )
    parser.add_argument(
        "--no-save-corrected",
        action="store_true",
        help="Do not save final corrected images",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in results.csv",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=32,
        help="Banknote grid cell size (default: 32)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable per-image pipeline logging",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--csv-interval",
        type=int,
        default=10,
        help="Write CSV every N images (default: 10)",
    )
    args = parser.parse_args()

    # ── Load annotations ──────────────────────────────────────────────

    ann_path = args.data_dir / "annotations.json"
    if not ann_path.exists():
        logger.error("Annotations not found: %s", ann_path)
        sys.exit(1)

    annotations = load_annotations(ann_path)
    images_dir = args.data_dir / "images"
    ref_dir = args.data_dir / "ref"

    # Resolve methods
    methods = args.methods if args.methods else "all"

    # ── Filter by split ───────────────────────────────────────────────

    image_names: list[str] = []
    for name, ann in annotations.items():
        lighting = ann.get("lighting_id")
        if args.split == "train" and lighting not in TRAIN_LIGHTINGS:
            continue
        if args.split == "eval" and lighting not in EVAL_LIGHTINGS:
            continue
        image_names.append(name)

    if not image_names:
        logger.error("No images match split=%r in %s", args.split, ann_path)
        sys.exit(1)

    logger.info(
        "Dataset: %d images (split=%s) from %s",
        len(image_names), args.split, args.data_dir,
    )

    # ── Resume logic ──────────────────────────────────────────────────

    csv_path = args.output / "results.csv"
    results: list[ImageResult] = []
    done_stems: set[str] = set()
    partial_stems: dict[str, list[str]] = {}  # stem → missing methods

    from src.color_calibration import METHODS as ALL_METHODS
    requested = set(args.methods) if args.methods else set(ALL_METHODS)

    if args.resume and csv_path.exists():
        # ── Check run-level parameter compatibility ───────────────
        run_info_path = args.output / "run_info.json"
        prev_cell_size = None
        if run_info_path.exists():
            prev_info = json.loads(run_info_path.read_text())
            prev_cell_size = prev_info.get("cell_size")

        if prev_cell_size is not None and prev_cell_size != args.cell_size:
            logger.warning(
                "Resume: cell_size changed (%d → %d) — discarding all previous results",
                prev_cell_size, args.cell_size,
            )
            # cell_size affects all banknote rows; can't keep any
        else:
            all_existing = read_results_csv(csv_path)

            # Group rows by image
            from collections import defaultdict
            by_stem: dict[str, list[ImageResult]] = defaultdict(list)
            for r in all_existing:
                by_stem[r.image_stem].append(r)

            for stem, rows in by_stem.items():
                existing_methods = {
                    r.correction_method for r in rows
                } - {"baseline"}
                keep = existing_methods & requested   # overlap → keep
                missing = requested - existing_methods # not yet run → run
                # (existing_methods - requested) → implicitly dropped

                # Retain baseline + rows whose method is in the requested set
                kept_rows = [
                    r for r in rows
                    if r.correction_method == "baseline"
                    or r.correction_method in keep
                ]
                results.extend(kept_rows)

                if not missing:
                    done_stems.add(stem)
                else:
                    partial_stems[stem] = sorted(missing)

            n_kept_rows = len(results)
            n_dropped = len(all_existing) - n_kept_rows
            logger.info(
                "Resume: %d images done, %d partial (%d rows kept, %d dropped)",
                len(done_stems), len(partial_stems), n_kept_rows, n_dropped,
            )

    # ── Save run info ─────────────────────────────────────────────────

    args.output.mkdir(parents=True, exist_ok=True)
    run_info = _build_run_info(args, len(image_names))
    (args.output / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False)
    )

    # ── Shared kwargs for the worker (methods passed per-image) ──────
    worker_kwargs = dict(
        annotations=annotations,
        images_dir=images_dir,
        ref_dir=ref_dir,
        cell_size=args.cell_size,
        log=args.log,
        save_corrected=not args.no_save_corrected,
        output_dir=args.output,
    )

    # ── Build todo list: (image_name, methods_to_run) ────────────────

    n_total = len(image_names)
    n_processed = 0
    n_failed = 0
    t_start = time.monotonic()
    csv_interval = args.csv_interval
    since_last_save = 0

    todo: list[tuple[str, str | list[str]]] = []
    for name in image_names:
        stem = Path(name).stem
        if stem in done_stems:
            continue
        if stem in partial_stems:
            todo.append((name, partial_stems[stem]))
        else:
            todo.append((name, methods))

    n_skipped = n_total - len(todo)
    if n_skipped:
        logger.info("Resume: skipping %d fully-done images", n_skipped)

    workers = max(1, args.workers)

    def _collect(image_name: str, image_results: list[ImageResult],
                 is_partial: bool) -> None:
        nonlocal n_processed, n_failed, since_last_save
        if is_partial:
            # Partial re-run: drop the new baseline row (we kept the old one)
            image_results = [r for r in image_results
                             if r.correction_method != "baseline"]
        results.extend(image_results)
        ok = all(r.success for r in image_results) if image_results else False
        done_count = n_processed + n_failed + 1
        if ok:
            n_processed += 1
        else:
            n_failed += 1
        elapsed = time.monotonic() - t_start
        rate = elapsed / max(n_processed + n_failed, 1)
        eta = rate * (len(todo) - n_processed - n_failed)
        logger.info("[%d/%d] %s — %s (%d rows, %.1fs/img, ETA %.0fs)",
                    done_count + n_skipped, n_total, image_name,
                    "OK" if ok else "FAIL", len(image_results), rate, eta)
        since_last_save += 1
        if since_last_save >= csv_interval:
            write_results_csv(results, csv_path)
            since_last_save = 0

    if workers <= 1:
        for image_name, image_methods in todo:
            is_partial = Path(image_name).stem in partial_stems
            image_results = _process_one_image(
                image_name, methods=image_methods, **worker_kwargs)
            _collect(image_name, image_results, is_partial)
    else:
        logger.info("Parallel mode: %d workers", workers)
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            futures = {
                pool.submit(
                    _process_one_image, name,
                    methods=img_methods, **worker_kwargs,
                ): name
                for name, img_methods in todo
            }
            for fut in as_completed(futures):
                image_name = futures[fut]
                is_partial = Path(image_name).stem in partial_stems
                try:
                    image_results = fut.result()
                except Exception as exc:
                    logger.error("Worker crash on %s: %s", image_name, exc)
                    image_results = [ImageResult(
                        image_path=image_name,
                        image_stem=Path(image_name).stem,
                        success=False,
                        failure_reason=f"worker crash: {exc}",
                    )]
                _collect(image_name, image_results, is_partial)

    # Final CSV write
    write_results_csv(results, csv_path)

    # ── Summary ───────────────────────────────────────────────────────

    elapsed = time.monotonic() - t_start
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("  Total images:  %d", n_total)
    logger.info("  Processed:     %d", n_processed)
    logger.info("  Skipped:       %d (resume)", n_skipped)
    logger.info("  Failed:        %d", n_failed)
    logger.info("  Result rows:   %d", len(results))
    logger.info("  CSV:           %s", csv_path)
    logger.info("  Elapsed:       %.1fs (%.1fs/image)",
                 elapsed, elapsed / max(n_processed, 1))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
