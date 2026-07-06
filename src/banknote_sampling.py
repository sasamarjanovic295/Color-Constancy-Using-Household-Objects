"""
Banknote grid sampling, masking, visualization, and I/O.

Samples colours from a banknote image using a regular grid of cells.
Each cell yields one representative colour via:

1. **Blur** the image (Gaussian 5×5) to suppress sensor noise.
2. **Divide** into a grid of ``cell_size × cell_size`` cells.
3. Take the **inner window** (central 70 %) of each cell.
4. Compute the **median** colour of pixels inside the inner window.

The module provides a single high-level entry point for callers:

- :func:`sample_and_save` — sample at multiple cell sizes, apply masks,
  write ``samples.json``, ``masks.json``, and per-size grid PNGs into
  an *output_dir*.  Both the reference-sampling script and
  ``calibrate_single`` call this function — the only difference is the
  output path they pass.

Lower-level building blocks are also public for custom use:

- :func:`sample_banknote_grid` — pure grid sampling, returns arrays.
- :func:`make_exclusion_mask` — build boolean mask from (row, col) list.
- :func:`frac_regions_to_cells` — convert fractional unstable-region
  definitions to concrete (row, col) exclusion lists for a given grid.
- :func:`draw_sampling_grid` — render the visualization overlay.

Colour-space convention
-----------------------
All functions operate in **sRGB float32 [0, 1]**.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CELL_SIZE: int = 32
_DEFAULT_INNER_RATIO: float = 0.7
_DEFAULT_BLUR_KSIZE: int = 5
_DEFAULT_SWATCH_SIZE: int = 8
_DEFAULT_CELL_SIZES: list[int] = [32, 64, 128]


# ---------------------------------------------------------------------------
# Grid geometry helpers
# ---------------------------------------------------------------------------


def _grid_dims(h: int, w: int, cell_size: int) -> tuple[int, int]:
    """Return (n_rows, n_cols) for a grid of *cell_size* covering *h × w*."""
    return h // cell_size, w // cell_size


def _cell_bounds(
    row: int, col: int, cell_size: int, h: int, w: int,
) -> tuple[int, int, int, int]:
    """Return (y1, y2, x1, x2) pixel bounds for cell (row, col)."""
    y1 = row * cell_size
    x1 = col * cell_size
    y2 = min(y1 + cell_size, h)
    x2 = min(x1 + cell_size, w)
    return y1, y2, x1, x2


def _inner_bounds(
    y1: int, y2: int, x1: int, x2: int, inner_ratio: float,
) -> tuple[int, int, int, int]:
    """Return (iy1, iy2, ix1, ix2) for the inner window of a cell."""
    bw, bh = x2 - x1, y2 - y1
    margin_x = int(round((1.0 - inner_ratio) * 0.5 * bw))
    margin_y = int(round((1.0 - inner_ratio) * 0.5 * bh))
    return y1 + margin_y, y2 - margin_y, x1 + margin_x, x2 - margin_x


# ---------------------------------------------------------------------------
# Public API — sampling
# ---------------------------------------------------------------------------


def sample_banknote_grid(
    image_srgb: np.ndarray,
    cell_size: int = _DEFAULT_CELL_SIZE,
    inner_ratio: float = _DEFAULT_INNER_RATIO,
    blur_ksize: int = _DEFAULT_BLUR_KSIZE,
    aggregation: str = "median",
) -> tuple[np.ndarray, int, int]:
    """Grid-sample a banknote image.

    Parameters
    ----------
    image_srgb : (H, W, 3) float32 sRGB [0, 1].
    cell_size : side length of each grid cell in pixels.
    inner_ratio : fraction of the cell used for the inner window.
    blur_ksize : Gaussian blur kernel size (0 to skip).
    aggregation : ``"median"`` or ``"mean"``.

    Returns
    -------
    samples : (n_rows * n_cols, 3) float32, row-major.
    n_rows, n_cols : grid dimensions.
    """
    h, w = image_srgb.shape[:2]
    n_rows, n_cols = _grid_dims(h, w, cell_size)

    if n_rows == 0 or n_cols == 0:
        return np.empty((0, 3), dtype=np.float32), 0, 0

    if blur_ksize > 0 and blur_ksize % 2 == 1:
        blurred_srgb = cv.GaussianBlur(image_srgb, (blur_ksize, blur_ksize), 0)
    else:
        blurred_srgb = image_srgb

    use_median = aggregation == "median"
    samples = np.empty((n_rows * n_cols, 3), dtype=np.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            y1, y2, x1, x2 = _cell_bounds(r, c, cell_size, h, w)
            iy1, iy2, ix1, ix2 = _inner_bounds(y1, y2, x1, x2, inner_ratio)
            iy1 = max(iy1, 0)
            iy2 = min(iy2, h)
            ix1 = max(ix1, 0)
            ix2 = min(ix2, w)

            region = blurred_srgb[iy1:iy2, ix1:ix2]
            if region.size == 0:
                samples[r * n_cols + c] = 0.0
                continue

            pixels = region.reshape(-1, 3)
            if use_median:
                samples[r * n_cols + c] = np.median(pixels, axis=0)
            else:
                samples[r * n_cols + c] = np.mean(pixels, axis=0)

    return samples, n_rows, n_cols


# ---------------------------------------------------------------------------
# Public API — exclusion mask
# ---------------------------------------------------------------------------


def make_exclusion_mask(
    n_cells: int,
    n_cols: int,
    excluded_cells: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Create a boolean validity mask for a sampling grid.

    Parameters
    ----------
    n_cells : total cells (n_rows × n_cols).
    n_cols : grid width (for row-major indexing).
    excluded_cells : (row, col) tuples to set False, or None.

    Returns
    -------
    mask : (n_cells,) bool — True = valid.
    """
    mask = np.ones(n_cells, dtype=bool)
    if excluded_cells:
        for r, c in excluded_cells:
            idx = r * n_cols + c
            if 0 <= idx < n_cells:
                mask[idx] = False
    return mask


def frac_regions_to_cells(
    regions: list[dict],
    n_rows: int,
    n_cols: int,
    cell_size: int | None = None,
) -> list[tuple[int, int]]:
    """Convert fractional unstable-region rectangles to (row, col) lists.

    Each region dict has keys ``x1``, ``x2``, ``y1``, ``y2`` as fractions
    of image width/height, plus a ``reason`` string and an optional
    ``min_cell_size`` integer.

    When *cell_size* is provided and a region has ``min_cell_size`` set,
    the region is **skipped** (not excluded) if
    ``cell_size >= min_cell_size``.  This lets small features (serial
    numbers, signatures) be excluded only at fine grid resolutions where
    they could dominate a cell, but included at coarser resolutions
    where the median is robust enough.

    Parameters
    ----------
    regions : list of ``{x1, x2, y1, y2, reason, min_cell_size?}`` dicts.
    n_rows, n_cols : grid dimensions.
    cell_size : current sampling cell size in pixels.  ``None`` means
        all regions are applied unconditionally.

    Returns
    -------
    List of unique (row, col) tuples covered by active regions.
    """
    cells: set[tuple[int, int]] = set()
    for reg in regions:
        threshold = reg.get("min_cell_size")
        if threshold is not None and cell_size is not None and cell_size >= threshold:
            continue
        c1 = max(0, int(reg["x1"] * n_cols))
        c2 = min(n_cols, int(np.ceil(reg["x2"] * n_cols)))
        r1 = max(0, int(reg["y1"] * n_rows))
        r2 = min(n_rows, int(np.ceil(reg["y2"] * n_rows)))
        for r in range(r1, r2):
            for c in range(c1, c2):
                cells.add((r, c))
    return sorted(cells)


# ---------------------------------------------------------------------------
# Public API — visualization
# ---------------------------------------------------------------------------


def draw_sampling_grid(
    image_srgb: np.ndarray,
    samples: np.ndarray,
    n_rows: int,
    n_cols: int,
    valid_mask: np.ndarray | None = None,
    cell_size: int = _DEFAULT_CELL_SIZE,
    inner_ratio: float = _DEFAULT_INNER_RATIO,
    swatch_size: int = _DEFAULT_SWATCH_SIZE,
) -> np.ndarray:
    """Draw the sampling grid with inner windows and median-colour swatches.

    Valid cells get a green border and a small filled swatch in the centre.
    Excluded cells get a red border and a diagonal cross.
    Row/column indices are drawn on margins.

    Returns BGR uint8 image ready for ``cv.imwrite``.
    """
    h, w = image_srgb.shape[:2]

    margin_top = 18
    margin_left = 22
    canvas_h = h + margin_top
    canvas_w = w + margin_left

    img_bgr = (np.clip(image_srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    img_bgr = cv.cvtColor(img_bgr, cv.COLOR_RGB2BGR)

    vis_bgr = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    vis_bgr[margin_top:, margin_left:] = img_bgr

    if valid_mask is None:
        valid_mask = np.ones(n_rows * n_cols, dtype=bool)

    color_grid = (180, 180, 180)
    color_valid = (0, 200, 0)
    color_excluded = (0, 0, 220)
    color_label = (80, 80, 80)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3

    for c in range(n_cols):
        cx = margin_left + c * cell_size + cell_size // 2
        cv.putText(vis_bgr, str(c), (cx - 4, margin_top - 5),
                   font, font_scale, color_label, 1, cv.LINE_AA)

    for r in range(n_rows):
        cy = margin_top + r * cell_size + cell_size // 2
        cv.putText(vis_bgr, str(r), (2, cy + 4),
                   font, font_scale, color_label, 1, cv.LINE_AA)

    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            y1, y2, x1, x2 = _cell_bounds(r, c, cell_size, h, w)
            iy1, iy2, ix1, ix2 = _inner_bounds(y1, y2, x1, x2, inner_ratio)
            y1 += margin_top
            y2 += margin_top
            x1 += margin_left
            x2 += margin_left
            iy1 += margin_top
            iy2 += margin_top
            ix1 += margin_left
            ix2 += margin_left

            is_valid = bool(valid_mask[idx])

            cv.rectangle(vis_bgr, (x1, y1), (x2 - 1, y2 - 1), color_grid, 1)

            if is_valid:
                cv.rectangle(vis_bgr, (ix1, iy1), (ix2 - 1, iy2 - 1),
                             color_valid, 2)
                sample_rgb = samples[idx]
                sample_bgr = (
                    int(sample_rgb[2] * 255 + 0.5),
                    int(sample_rgb[1] * 255 + 0.5),
                    int(sample_rgb[0] * 255 + 0.5),
                )
                cy = (iy1 + iy2) // 2
                cx = (ix1 + ix2) // 2
                half = swatch_size // 2
                sy1 = max(cy - half, 0)
                sy2 = min(cy + half, canvas_h)
                sx1 = max(cx - half, 0)
                sx2 = min(cx + half, canvas_w)
                cv.rectangle(vis_bgr, (sx1, sy1), (sx2, sy2),
                             sample_bgr, cv.FILLED)
                cv.rectangle(vis_bgr, (sx1, sy1), (sx2, sy2),
                             (0, 0, 0), 1)
            else:
                cv.rectangle(vis_bgr, (ix1, iy1), (ix2 - 1, iy2 - 1),
                             color_excluded, 2)
                cv.line(vis_bgr, (x1, y1), (x2 - 1, y2 - 1),
                        color_excluded, 1)
                cv.line(vis_bgr, (x2 - 1, y1), (x1, y2 - 1),
                        color_excluded, 1)

    return vis_bgr

def draw_unstable_regions(
    image_srgb: np.ndarray,
    regions: list[dict],
) -> np.ndarray:
    """Draw unstable-region rectangles on the banknote image with a
    fractional-coordinate ruler on top and left edges.

    Each region is rendered as a semi-transparent red overlay with a
    label showing ``reason`` and its ``(x1,y1)-(x2,y2)`` coordinates.

    The rulers mark 0.0–1.0 in 0.05 steps (minor ticks) and 0.1 steps
    (major ticks with labels), so the user can read off positions to
    edit ``unstable_regions.json``.

    Parameters
    ----------
    image_srgb : (H, W, 3) float32 sRGB [0, 1].
    regions : list of ``{x1, x2, y1, y2, reason}`` dicts with
        fractional coordinates.

    Returns
    -------
    BGR uint8 image ready for ``cv.imwrite``.
    """
    h, w = image_srgb.shape[:2]

    # ── Margins for rulers ────────────────────────────────────────────
    margin_top = 28
    margin_left = 34
    canvas_h = h + margin_top
    canvas_w = w + margin_left

    base = (np.clip(image_srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    base = cv.cvtColor(base, cv.COLOR_RGB2BGR)

    vis = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
    vis[margin_top:, margin_left:] = base.copy()

    overlay = vis.copy()

    color_permanent_fill = (0, 0, 200)       # red BGR
    color_permanent_border = (0, 0, 255)
    color_conditional_fill = (0, 140, 230)    # orange BGR
    color_conditional_border = (0, 165, 255)
    color_text = (255, 255, 255)
    color_ruler = (60, 60, 60)
    color_tick_minor = (160, 160, 160)
    font = cv.FONT_HERSHEY_SIMPLEX

    font_scale = max(0.35, h / 1400.0)
    thickness = 1 if h < 700 else 2
    ruler_font = 0.30
    tick_major = 10
    tick_minor = 5

    # ── Top ruler (X axis: 0.0 → 1.0) ────────────────────────────────
    for i in range(21):  # 0.00, 0.05, 0.10, ..., 1.00
        frac = i * 0.05
        px = margin_left + int(frac * w)
        is_major = (i % 2 == 0)  # 0.0, 0.1, 0.2, ...
        tlen = tick_major if is_major else tick_minor
        tcol = color_ruler if is_major else color_tick_minor
        cv.line(vis, (px, margin_top - tlen), (px, margin_top), tcol, 1)
        if is_major:
            label = f".{i // 2}" if i > 0 else "0"
            (tw, _), _ = cv.getTextSize(label, font, ruler_font, 1)
            cv.putText(vis, label, (px - tw // 2, margin_top - tlen - 3),
                       font, ruler_font, color_ruler, 1, cv.LINE_AA)

    # ── Left ruler (Y axis: 0.0 → 1.0) ───────────────────────────────
    for i in range(21):
        frac = i * 0.05
        py = margin_top + int(frac * h)
        is_major = (i % 2 == 0)
        tlen = tick_major if is_major else tick_minor
        tcol = color_ruler if is_major else color_tick_minor
        cv.line(vis, (margin_left - tlen, py), (margin_left, py), tcol, 1)
        if is_major:
            label = f".{i // 2}" if i > 0 else "0"
            (tw, th_text), _ = cv.getTextSize(label, font, ruler_font, 1)
            cv.putText(vis, label, (margin_left - tlen - tw - 2,
                                    py + th_text // 2),
                       font, ruler_font, color_ruler, 1, cv.LINE_AA)

    # ── Region overlays ───────────────────────────────────────────────
    for reg in regions:
        rx1 = margin_left + int(reg["x1"] * w)
        rx2 = margin_left + int(reg["x2"] * w)
        ry1 = margin_top + int(reg["y1"] * h)
        ry2 = margin_top + int(reg["y2"] * h)
        reason = reg.get("reason", "")
        threshold = reg.get("min_cell_size")

        is_conditional = threshold is not None
        cfill = color_conditional_fill if is_conditional else color_permanent_fill
        cborder = color_conditional_border if is_conditional else color_permanent_border

        cv.rectangle(overlay, (rx1, ry1), (rx2, ry2), cfill, cv.FILLED)
        cv.rectangle(vis, (rx1, ry1), (rx2, ry2), cborder, 2)

        # Label: "reason  (x1,y1)-(x2,y2)  [≥64: ok]"
        coords = f"({reg['x1']:.2f},{reg['y1']:.2f})-({reg['x2']:.2f},{reg['y2']:.2f})"
        parts = [reason, coords] if reason else [coords]
        if is_conditional:
            parts.append(f"[>={threshold}px: ok]")
        label = "  ".join(parts)
        (tw, th), _ = cv.getTextSize(label, font, font_scale, thickness)
        tx = rx1 + 4
        ty = ry1 + th + 6
        if ty > ry2:
            ty = ry2 - 4
        cv.rectangle(vis, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4),
                     (0, 0, 0), cv.FILLED)
        cv.putText(vis, label, (tx, ty), font, font_scale,
                   color_text, thickness, cv.LINE_AA)

    cv.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

    return vis


# ---------------------------------------------------------------------------
# Public API — high-level: sample + save
# ---------------------------------------------------------------------------


def sample_and_save(
    image_srgb: np.ndarray,
    output_dir: Path | str,
    cell_sizes: list[int] | None = None,
    unstable_regions: list[dict] | None = None,
    inner_ratio: float = _DEFAULT_INNER_RATIO,
    blur_ksize: int = _DEFAULT_BLUR_KSIZE,
    aggregation: str = "median",
) -> dict[int, dict]:
    """Sample a banknote image at multiple cell sizes and persist results.

    Creates (or updates) ``samples.json`` and ``masks.json`` in
    *output_dir*, and writes one grid visualization PNG per cell size.

    Parameters
    ----------
    image_srgb : (H, W, 3) float32 sRGB [0, 1].
    output_dir : directory for output files.  Created if absent.
    cell_sizes : list of cell sizes to sample (default [32, 64, 128]).
    unstable_regions : fractional region dicts ``{x1, x2, y1, y2, reason}``.
        Cells overlapping these regions are excluded.  ``None`` → no
        exclusions (all cells valid).
    inner_ratio : inner-window fraction.
    blur_ksize : Gaussian blur kernel size.
    aggregation : ``"median"`` or ``"mean"``.

    Returns
    -------
    Dict keyed by cell_size with per-size result dicts containing
    ``samples``, ``mask``, ``n_rows``, ``n_cols``, ``n_valid``.

    Output files
    ------------
    ``<output_dir>/samples.json``
        ``{"32": [[r,g,b], ...], "64": [...], "128": [...]}``
    ``<output_dir>/masks.json``
        ``{"32": [true, false, ...], "64": [...], ...}``
    ``<output_dir>/grid_32.png``, ``grid_64.png``, ``grid_128.png``
        Grid visualizations with exclusion overlays.
    """
    if cell_sizes is None:
        cell_sizes = list(_DEFAULT_CELL_SIZES)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load existing JSON files to merge into (upsert by cell-size key)
    samples_path = out / "samples.json"
    masks_path = out / "masks.json"

    all_samples: dict[str, list] = {}
    all_masks: dict[str, list] = {}
    if samples_path.exists():
        with open(samples_path) as f:
            all_samples = json.load(f)
    if masks_path.exists():
        with open(masks_path) as f:
            all_masks = json.load(f)

    results: dict[int, dict] = {}

    for cs in cell_sizes:
        key = str(cs)

        samples, n_rows, n_cols = sample_banknote_grid(
            image_srgb, cell_size=cs, inner_ratio=inner_ratio,
            blur_ksize=blur_ksize, aggregation=aggregation,
        )
        n_total = n_rows * n_cols

        if n_total == 0:
            logger.warning("cell_size=%d: image too small, skipping", cs)
            continue

        # Build exclusion mask from fractional regions
        if unstable_regions:
            excluded = frac_regions_to_cells(unstable_regions, n_rows, n_cols, cell_size=cs)
        else:
            excluded = None
        mask = make_exclusion_mask(n_total, n_cols, excluded)
        n_valid = int(np.count_nonzero(mask))

        # Visualization (always overwrite)
        viz_bgr = draw_sampling_grid(
            image_srgb, samples, n_rows, n_cols,
            valid_mask=mask, cell_size=cs, inner_ratio=inner_ratio,
        )
        cv.imwrite(str(out / f"grid_{cs}.png"), viz_bgr)

        # Upsert into JSON dicts
        all_samples[key] = samples.tolist()
        all_masks[key] = mask.tolist()

        results[cs] = {
            "samples": samples,
            "mask": mask,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_valid": n_valid,
            "n_excluded": n_total - n_valid,
        }

        logger.debug(
            "cell_size=%d: %dx%d=%d cells, %d valid",
            cs, n_rows, n_cols, n_total, n_valid,
        )

    # Write JSON (full replace with merged content)
    with open(samples_path, "w") as f:
        json.dump(all_samples, f)
    with open(masks_path, "w") as f:
        json.dump(all_masks, f)

    return results
