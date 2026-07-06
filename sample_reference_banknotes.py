"""
Generate reference banknote PNGs and sample them.

For each ``_v0_*.jpeg`` source image in ``--ref-dir``:

1. Resize to height=512 with correct aspect ratio from ECB Europa series
   physical dimensions.  Saves as ``<denom>_<side>.png``.
2. Run :func:`sample_and_save` which creates ``samples.json``,
   ``masks.json``, and grid visualisation PNGs inside a per-banknote
   subfolder.

Unstable regions (hologram, serial number, etc.) are loaded from
``unstable_regions.json`` in the same directory and applied as exclusion
masks automatically.

Usage::

    python sample_reference_banknotes.py
    python sample_reference_banknotes.py --ref-dir data/ref
    python sample_reference_banknotes.py --cell-sizes 32 64
    python sample_reference_banknotes.py --height 1024
    python sample_reference_banknotes.py --skip-resize   # re-sample existing PNGs only

Output structure::

    data/ref/
    ├── 10_double.png            ← resized from _v0_10_double.jpeg
    ├── 10_double/
    │   ├── grid_32.png
    │   ├── grid_64.png
    │   ├── grid_128.png
    │   ├── samples.json
    │   └── masks.json
    ├── 10_single.png
    ├── 10_single/
    │   └── ...
    └── unstable_regions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from src.banknote_sampling import draw_unstable_regions, sample_and_save

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger("sample_reference")

# Euro banknote physical dimensions (ES2 Europa series), millimetres.
# Source: ECB "Design elements".
_EURO_DIMENSIONS_MM: dict[int, tuple[float, float]] = {
    5: (120.0, 62.0),
    10: (127.0, 67.0),
    20: (133.0, 72.0),
    50: (140.0, 77.0),
    100: (147.0, 77.0),
}

_DENOMINATIONS = sorted(_EURO_DIMENSIONS_MM)
_SIDES = ("single", "double")


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


def _resize_v0_to_png(
    ref_dir: Path, target_height: int,
) -> list[tuple[str, Path]]:
    """Resize each ``_v0_*.jpeg`` to an AR-correct PNG.

    Returns list of ``(stem, png_path)`` for images that were processed.
    """
    results: list[tuple[str, Path]] = []

    for denom in _DENOMINATIONS:
        w_mm, h_mm = _EURO_DIMENSIONS_MM[denom]
        ar = w_mm / h_mm
        target_w = round(target_height * ar)

        for side in _SIDES:
            stem = f"{denom}_{side}"
            v0_path = ref_dir / f"_v0_{stem}.jpeg"
            png_path = ref_dir / f"{stem}.png"

            if not v0_path.exists():
                logger.warning("Source not found: %s — skipping", v0_path)
                continue

            bgr = cv.imread(str(v0_path))
            if bgr is None:
                logger.warning("Failed to decode: %s — skipping", v0_path)
                continue

            orig_h, orig_w = bgr.shape[:2]

            # INTER_AREA for downscale (sharper), LANCZOS4 for upscale (smoother)
            interp = cv.INTER_AREA if orig_h > target_height else cv.INTER_LANCZOS4

            resized = cv.resize(bgr, (target_w, target_height), interpolation=interp)
            cv.imwrite(str(png_path), resized)

            interp_name = "AREA" if interp == cv.INTER_AREA else "LANCZOS4"
            logger.info(
                "  %s: %dx%d → %dx%d  (AR=%.4f, %s)",
                stem, orig_w, orig_h, target_w, target_height, ar, interp_name,
            )
            results.append((stem, png_path))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reference banknote PNGs and sample them",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=Path("data/ref"),
        help="Directory with _v0_*.jpeg source images (default: data/ref)",
    )
    parser.add_argument(
        "--cell-sizes",
        type=int,
        nargs="*",
        default=[32, 64, 128],
        help="Cell sizes to sample (default: 32 64 128)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Target height in pixels for reference PNGs (default: 512)",
    )
    parser.add_argument(
        "--skip-resize",
        action="store_true",
        help="Skip resize step — only re-sample existing PNGs",
    )
    args = parser.parse_args()

    ref_dir: Path = args.ref_dir

    # ── 1. Resize _v0 originals → PNGs ───────────────────────────────

    if not args.skip_resize:
        logger.info("=== RESIZE: _v0 originals → height=%d PNGs ===", args.height)
        resized = _resize_v0_to_png(ref_dir, args.height)
        logger.info("Resized %d images", len(resized))
    else:
        logger.info("Skipping resize (--skip-resize)")

    # ── 2. Load unstable regions ──────────────────────────────────────

    regions_path = ref_dir / "unstable_regions.json"
    all_regions: dict[str, list[dict]] = {}
    if regions_path.exists():
        with open(regions_path) as f:
            all_regions = json.load(f)
        logger.info("Loaded unstable regions for %d banknotes", len(all_regions))
    else:
        logger.warning("No unstable_regions.json found — no exclusions applied")

    # ── 3. Find reference PNGs ────────────────────────────────────────

    ref_images = sorted(ref_dir.glob("*.png"))
    ref_images = [p for p in ref_images if not p.name.startswith("_v0_")]

    if not ref_images:
        logger.error("No *.png reference images found in %s", ref_dir)
        sys.exit(1)

    logger.info("=== SAMPLE: %d reference images ===", len(ref_images))

    # ── 4. Sample each reference image ────────────────────────────────

    for img_path in ref_images:
        stem = img_path.stem  # e.g. "20_single"
        logger.info("── %s ──", stem)

        bgr = cv.imread(str(img_path))
        if bgr is None:
            logger.warning("  Failed to load: %s", img_path)
            continue
        image_srgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        logger.info("  Size: %d × %d px", image_srgb.shape[1], image_srgb.shape[0])

        output_dir = ref_dir / stem
        regions = all_regions.get(stem)

        # Unstable-regions visualization
        if regions:
            viz_bgr = draw_unstable_regions(image_srgb, regions)
            viz_path = output_dir / "viz_unstable_regions.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(viz_path), viz_bgr)
            logger.info("  Saved unstable regions viz (%d regions)", len(regions))

        results = sample_and_save(
            image_srgb,
            output_dir=output_dir,
            cell_sizes=args.cell_sizes,
            unstable_regions=regions,
        )

        for cs, info in sorted(results.items()):
            logger.info(
                "  %dx%d: %dx%d=%d cells, %d valid, %d excluded",
                cs, cs, info["n_rows"], info["n_cols"],
                info["n_rows"] * info["n_cols"],
                info["n_valid"], info["n_excluded"],
            )

    logger.info("=" * 50)
    logger.info("Done: %d reference images processed", len(ref_images))


if __name__ == "__main__":
    main()
