"""
colorchecker.py — SCK300 (Spyder Checkr Photo) detection and swatch measurement.

Pipeline:
    1. Call colour-checker-detection segmentation with custom SCK300 parameters
       (8 cols × 6 rows, aspect ratio 1.415, 48 swatches).
    2. Linearise the returned sRGB values via apply_cctf_decoding=True.
    3. Convert linear sRGB (D65) → XYZ (D65) → Bradford CAT → XYZ (D50).
    4. Convert both measured XYZ(D50) and reference XYZ(D50) to Lab(D50).
    5. Compute per-patch ΔE00; return aggregated stats.

Usage::

    from pathlib import Path
    from src.colorchecker import detect_and_measure

    image_srgb = load_image_rgb(Path("data/train/IMG_colorchecker.jpeg"))
    result = detect_and_measure(image_srgb, save_steps=True, log=True,
                                output_dir=Path("output/debug"))
    if result.detected:
        correction = apply_correction(
            image_srgb,
            result.swatch_colors_xyz,
            result.reference_colors_xyz,
            method="affine",
        )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import colour
import cv2 as cv
import numpy as np
from colour.models import eotf_inverse_sRGB, eotf_sRGB
from colour.utilities import metric_mse
from colour_checker_detection.detection.common import (
    DataDetectionColourChecker,
    as_int32_array,
    reformat_image,
    sample_colour_checker,
)
from colour_checker_detection.detection.segmentation import segmenter_default

from src.reference_data import (
    SCK300_LEFT_PANEL_SRGB_LINEAR,
    SCK300_PATCH_NAMES,
    SCK300_REFERENCE_XYZ,
    SCK300_RIGHT_PANEL_SRGB_LINEAR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Grid layout: 8 columns × 6 rows = 48 swatches
_SCK300_SWATCHES_HORIZONTAL: int = 8
_SCK300_SWATCHES_VERTICAL: int = 6
_SCK300_SWATCHES: int = 48

# ---------------------------------------------------------------------------
# Panel detection constants — SCK300 open-book two-panel (24+24) strategy
# ---------------------------------------------------------------------------

# Option A — Portrait panel detection
# The colour-checker-detection library normalises aspect ratio using max()/min()
# regardless of physical orientation. A portrait panel (~55mm wide × 106mm tall)
# is normalised to: max(106, 55) / min(106, 55) = 106/55 ≈ 1.93.
_PANEL_ASPECT_RATIO_A: float = 106.0 / 55.0  # ≈ 1.927

_PANEL_SEGMENTER_KWARGS_A: dict = {
    "swatches_horizontal": 4,  # 4 columns (A–D or E–H per panel)
    "swatches_vertical": 6,  # 6 rows (full card height)
    "swatches": 24,  # 24 swatches per panel
    "aspect_ratio": _PANEL_ASPECT_RATIO_A,
    "aspect_ratio_minimum": _PANEL_ASPECT_RATIO_A * 0.80,  # ≈ 1.54
    "aspect_ratio_maximum": _PANEL_ASPECT_RATIO_A * 1.20,  # ≈ 2.31
    "swatches_count_minimum": 12,  # 24 * 0.5
    "swatches_count_maximum": 36,  # 24 * 1.5
}

# Option B — Landscape panel detection
# If each panel is photographed in landscape orientation (card rotated, or panels
# stacked so their long axis is horizontal), the library normalises to 6/4 = 1.5.
# This coincides with the Classic 24 defaults, but with explicit count bounds.
_PANEL_ASPECT_RATIO_B: float = 6.0 / 4.0  # = 1.5

_PANEL_SEGMENTER_KWARGS_B: dict = {
    "swatches_horizontal": 6,  # 6 rows become "columns" in landscape warp
    "swatches_vertical": 4,  # 4 columns become "rows" in landscape warp
    "swatches": 24,
    "aspect_ratio": _PANEL_ASPECT_RATIO_B,
    "aspect_ratio_minimum": _PANEL_ASPECT_RATIO_B * 0.85,  # = 1.275
    "aspect_ratio_maximum": _PANEL_ASPECT_RATIO_B * 1.15,  # = 1.725
    "swatches_count_minimum": 12,
    "swatches_count_maximum": 36,
}

# Portrait panel: correct working_height and target rectangle for sample_colour_checker.
# The Classic default (960) is wrong for portrait panels — causes swatch_masks to
# sample outside the warped content area.
_PANEL_WORKING_HEIGHT_A: int = int(1440 / _PANEL_ASPECT_RATIO_A)  # 747
_PANEL_RECTANGLE_A: np.ndarray = as_int32_array(
    [[1440, 0], [1440, _PANEL_WORKING_HEIGHT_A], [0, _PANEL_WORKING_HEIGHT_A], [0, 0]]
)


# Checker type identifier string — used in ColorCheckerResult.checker_type
_CHECKER_TYPE: str = "SCK300"

# Internal working width that colour-checker-detection uses when resizing images.
# Quadrilateral coordinates returned by the library are in this pixel space.
_WORKING_WIDTH: int = 1440

# Visualization: swatch tile size in pixels for grids and comparison images
_VIZ_SWATCH_PX: int = 80  # 80px cells give room for ΔE00 text labels

# Output filenames when save_steps=True
_FILENAME_VIZ_CONTOURS: str = "viz_01_contours.jpg"
_FILENAME_VIZ_LEFT_PANEL: str = "viz_02_left_panel.jpg"
_FILENAME_VIZ_RIGHT_PANEL: str = "viz_03_right_panel.jpg"
_FILENAME_VIZ_SWATCHES: str = "viz_04_swatch_grid.jpg"
_FILENAME_VIZ_COMPARISON: str = "viz_05_comparison.jpg"

# Pre-computed illuminant references for XYZ conversions — always D50 for ΔE00
_D50_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
_D65_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
_D50_XYZ = colour.xy_to_XYZ(_D50_XY)
_D65_XYZ = colour.xy_to_XYZ(_D65_XY)

# OQ#1 reference: patch A1 Lab = [61.35, 34.81, 18.38] — narančasto-crvena nijansa.
# Used to verify that detected swatch order matches reference_data.py column-major order.
_A1_LAB_REFERENCE = np.array([61.35, 34.81, 18.38], dtype=np.float64)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ColorCheckerResult:
    """Structured output of the SCK300 detection and measurement pipeline.

    On success (``detected=True``) all measurement fields are populated.
    On failure only ``detected`` and ``failure_reason`` are set; all other
    fields are ``None``.

    .. warning::
       ``swatch_colors_xyz`` and ``reference_colors_xyz`` use **different
       illuminants** (D65 vs D50).  Before fitting a correction matrix,
       adapt the reference to D65::

           ref_d65 = adapt_d50_to_d65(result.reference_colors_xyz)
           params = calibrate(method, measured=result.swatch_colors_xyz,
                              reference=ref_d65)
    """

    detected: bool
    failure_reason: str | None  # format: "step: reason"
    checker_type: str | None  # "SCK300" on success, None on failure
    n_swatches: int | None  # 48 on success, None on failure
    swatch_colors_xyz: np.ndarray | None  # (48, 3) float64, XYZ under D65
    reference_colors_xyz: np.ndarray | None  # (48, 3) float64, XYZ under D50 (darktable)
    delta_e_stats: dict | None  # {"mean": float, "median": float, "p95": float}
    detection_image: np.ndarray | None  # shape (H, W, 3), uint8, BGR


# ---------------------------------------------------------------------------
# Private helpers — copied from detection.py (not imported; they are private)
# ---------------------------------------------------------------------------


def _float_to_bgr_u8(image_srgb: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0, 1] to uint8 BGR for OpenCV drawing functions."""
    image_rgb_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv.cvtColor(image_rgb_u8, cv.COLOR_RGB2BGR)


def _save_bgr(path: Path, bgr: np.ndarray) -> None:
    """Save a BGR uint8 image to *path* (creates parent directories)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), bgr)


# ---------------------------------------------------------------------------
# Private helpers — swatch order and orientation
# ---------------------------------------------------------------------------


def _reorder_swatches(swatches: np.ndarray) -> np.ndarray:
    """Reverse the swatch order to correct a 180° card rotation.

    When the detected swatch sequence is inverted relative to the reference
    (i.e., the right saturated panel appears first), reversing the array
    restores column-major A1-first order expected by reference_data.py.

    Args:
        swatches: (48, 3) float64 swatch array in reversed order.

    Returns:
        (48, 3) float64 swatch array in corrected A1-first order.
    """
    return swatches[::-1].copy()


def _row_major_to_column_major(
    swatches: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Convert a swatch array from row-major to column-major traversal order.

    The colour-checker-detection library returns swatches in row-major order:
    the outer loop iterates rows, the inner loop iterates columns, so index 0
    is (row 0, col 0), index 1 is (row 0, col 1), …, index n_cols is (row 1,
    col 0). The SCK300 reference in reference_data.py uses column-major order:
    index 0 is (row 0, col 0), index 1 is (row 1, col 0), …, i.e. A1, A2, A3,
    A4, A5, A6, B1, B2, … for a 6-row × 4-col panel.

    Parameters
    ----------
    swatches : np.ndarray
        Shape (n_rows * n_cols, 3), float64. Swatch colours in row-major order
        as returned by the colour-checker-detection library.
    n_rows : int
        Number of swatch rows (swatches_vertical). Must be 6 for SCK300 panels.
    n_cols : int
        Number of swatch columns (swatches_horizontal). Must be 4 for SCK300
        portrait panels (Pass 2) or 6 for landscape panels (Pass 3 — see note).

    Returns
    -------
    np.ndarray
        Shape (n_rows * n_cols, 3), float64. Same swatch colours in column-major
        order, matching the indexing convention of reference_data.py.
    """
    return (
        swatches.reshape(n_rows, n_cols, 3)
        .transpose(1, 0, 2)
        .reshape(n_rows * n_cols, 3)
        .copy()
    )


def _check_panel_saturation_order(swatches_linear_srgb: np.ndarray) -> bool:
    """Return True if panel order matches reference_data.py (left = neutral).

    The SCK300 left panel (columns A–D, indices 0–23) contains neutral and
    skin-tone patches with low chroma. The right panel (columns E–H, indices
    24–47) contains highly saturated colours. If mean(a²+b²) of the first 24
    swatches exceeds that of the last 24, the order is inverted.

    Args:
        swatches_linear_srgb: (48, 3) float64 linear sRGB swatch array.

    Returns:
        True if order is correct (left panel = lower saturation). False if
        inverted and _reorder_swatches() should be applied.
    """
    # Convert to approximate CIE ab for chroma: use gamma-space as proxy
    # (linear sRGB → Lab via XYZ takes more effort; ab² proxy is sufficient
    # for a relative comparison between the two panels)
    swatch_xyz_d65 = colour.RGB_to_XYZ(
        swatches_linear_srgb,
        colour.RGB_COLOURSPACES["sRGB"],
        chromatic_adaptation_transform=None,
    )
    swatch_xyz_d50 = colour.adaptation.chromatic_adaptation_VonKries(
        np.asarray(swatch_xyz_d65, dtype=np.float64),
        _D65_XYZ,
        _D50_XYZ,
        transform="Bradford",
    )
    swatch_lab_d50 = colour.XYZ_to_Lab(np.asarray(swatch_xyz_d50, dtype=np.float64), illuminant=_D50_XY)
    swatch_lab_d50 = np.asarray(swatch_lab_d50, dtype=np.float64)

    chroma_sq = swatch_lab_d50[:, 1] ** 2 + swatch_lab_d50[:, 2] ** 2  # a*² + b*²
    left_chroma = float(np.mean(chroma_sq[:24]))
    right_chroma = float(np.mean(chroma_sq[24:]))

    # Correct order: left (neutral/skin) has lower chroma than right (saturated)
    return left_chroma <= right_chroma


def _check_orientation(quadrilateral: np.ndarray | None, log: bool) -> None:
    """Log a warning if the detected bounding quad is taller than it is wide.

    Portrait orientation suggests the card may not be lying flat (landscape).
    Detection is not blocked — this is advisory only.

    Args:
        quadrilateral: (4, 2) float array of corner points, or None.
        log: If True, emit warning via logger.
    """
    if quadrilateral is None:
        return

    pts = np.asarray(quadrilateral, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return

    xs, ys = pts[:, 0], pts[:, 1]
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())

    if height > width:
        msg = (
            "colorchecker_orientation: bounding box is taller than wide "
            f"(w={width:.0f} px, h={height:.0f} px) — card may not be open "
            "flat (landscape). Detection continues but swatch order may be wrong."
        )
        if log:
            logger.warning(msg)
        else:
            logger.warning(msg)

# ---------------------------------------------------------------------------
# Private helpers — colourimetry
# ---------------------------------------------------------------------------




def _run_segmentation_v2(
    image_srgb: np.ndarray,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray | None],
    list[DataDetectionColourChecker | None],
    int,
    dict,
]:
    """Detect two SCK300 panels using per-panel reference values and correct working_height.

    Calls ``segmenter_default`` for rectangle detection, then
    ``sample_colour_checker`` for each panel with the correct
    ``reference_values`` (enabling the 4-rotation MSE search) and the
    correct ``working_height`` (matching the panel aspect ratio so that
    ``swatch_masks`` samples within the warped content area).

    Returns a 5-tuple: (swatch_arrays, quads, items, pass_id, seg_extra).
    """
    image_linear_srgb = eotf_sRGB(image_srgb)
    image_work_linear = reformat_image(
        np.asarray(image_linear_srgb, dtype=np.float32),
        _WORKING_WIDTH,
        cv.INTER_CUBIC,
    )

    rectangles = None
    panel_cfg = "A"

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        seg_data = segmenter_default(
            image_work_linear, additional_data=True, **_PANEL_SEGMENTER_KWARGS_A
        )
    if hasattr(seg_data, "rectangles") and len(seg_data.rectangles) >= 2:
        rectangles = seg_data.rectangles[:2]
        panel_cfg = "A"
        logger.debug("_run_segmentation_v2: portrait pass found %d rectangles", len(seg_data.rectangles))
    else:
        logger.debug("_run_segmentation_v2: portrait pass found <2 rectangles, trying landscape")
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            seg_data = segmenter_default(
                image_work_linear, additional_data=True, **_PANEL_SEGMENTER_KWARGS_B
            )
        if hasattr(seg_data, "rectangles") and len(seg_data.rectangles) >= 2:
            rectangles = seg_data.rectangles[:2]
            panel_cfg = "B"
            logger.debug("_run_segmentation_v2: landscape pass found %d rectangles", len(seg_data.rectangles))

    if rectangles is None:
        logger.debug("_run_segmentation_v2: no panel configuration found 2 rectangles")
        return [], [], [], 0, {}

    refs = [SCK300_LEFT_PANEL_SRGB_LINEAR, SCK300_RIGHT_PANEL_SRGB_LINEAR]
    sample_kwargs = dict(
        swatches_horizontal=4,
        swatches_vertical=6,
        working_width=_WORKING_WIDTH,
        working_height=_PANEL_WORKING_HEIGHT_A,
    )

    sampled: dict[tuple[int, int], tuple[DataDetectionColourChecker, float]] = {}
    for qi in range(2):
        for ri in range(2):
            data = sample_colour_checker(
                image_work_linear, rectangles[qi], _PANEL_RECTANGLE_A, 32,
                reference_values=refs[ri], **sample_kwargs,
            )
            mse = float(metric_mse(refs[ri], data.swatch_colours))
            sampled[(qi, ri)] = (data, mse)

    mse_a = sampled[(0, 0)][1] + sampled[(1, 1)][1]
    mse_b = sampled[(0, 1)][1] + sampled[(1, 0)][1]

    if mse_a <= mse_b:
        left_data = sampled[(0, 0)][0]
        right_data = sampled[(1, 1)][0]
    else:
        left_data = sampled[(1, 0)][0]
        right_data = sampled[(0, 1)][0]

    logger.debug(
        "_run_segmentation_v2: MSE assign A=%.4f B=%.4f → %s",
        mse_a, mse_b, "A" if mse_a <= mse_b else "B",
    )

    left_colmaj = _row_major_to_column_major(
        np.asarray(left_data.swatch_colours, dtype=np.float64), n_rows=6, n_cols=4
    )
    right_colmaj = _row_major_to_column_major(
        np.asarray(right_data.swatch_colours, dtype=np.float64), n_rows=6, n_cols=4
    )
    combined = np.concatenate([left_colmaj, right_colmaj], axis=0)

    return (
        [combined],
        [
            np.asarray(left_data.quadrilateral, dtype=np.float64),
            np.asarray(right_data.quadrilateral, dtype=np.float64),
        ],
        [left_data, right_data],
        2 if panel_cfg == "A" else 3,
        {"seg_data": seg_data, "image_work": image_work_linear},
    )


def _swatch_srgb_to_xyz(swatches_linear_srgb: np.ndarray) -> np.ndarray:
    """Convert (48, 3) linear sRGB float64 [0, 1] to CIE XYZ float64.

    The sRGB colour space has a D65 white point (IEC 61966-2-1).
    No chromatic adaptation is applied here — the caller decides whether to
    adapt to D50 before ΔE00 computation.

    Args:
        swatches_linear_srgb: (48, 3) float64 linear sRGB, range [0, 1].

    Returns:
        np.ndarray of shape (48, 3), float64, XYZ under D65.
    """
    swatches_xyz_d65 = colour.RGB_to_XYZ(
        swatches_linear_srgb,
        colour.RGB_COLOURSPACES["sRGB"],
        chromatic_adaptation_transform=None,  # keep D65 white point as-is
    )
    return np.asarray(swatches_xyz_d65, dtype=np.float64)


def _compute_delta_e_stats(
    measured_xyz_d65: np.ndarray,
    reference_xyz_d50: np.ndarray,
) -> dict:
    """Compute per-patch ΔE00 and return aggregated statistics.

    Both arrays must be XYZ. ``measured_xyz_d65`` is assumed to be under D65
    (from sRGB/camera pipeline); ``reference_xyz_d50`` is under D50 (darktable
    reference). A Bradford CAT is applied to bring the measured values to D50
    before converting both to Lab(D50) for a valid ΔE00 comparison.

    Args:
        measured_xyz_d65: (48, 3) float64, XYZ under D65.
        reference_xyz_d50: (48, 3) float64, XYZ under D50.

    Returns:
        dict with keys "mean", "median", "p95" — all finite floats ≥ 0.
    """
    # Bradford CAT: measured XYZ (D65) → XYZ (D50)
    measured_xyz_d50 = colour.adaptation.chromatic_adaptation_VonKries(
        measured_xyz_d65,
        _D65_XYZ,
        _D50_XYZ,
        transform="Bradford",
    )
    measured_xyz_d50 = np.asarray(measured_xyz_d50, dtype=np.float64)

    # Convert both to Lab(D50) — always explicit illuminant (version-safe)
    measured_lab_d50 = colour.XYZ_to_Lab(measured_xyz_d50, illuminant=_D50_XY)
    reference_lab_d50 = colour.XYZ_to_Lab(reference_xyz_d50, illuminant=_D50_XY)

    measured_lab_d50 = np.asarray(measured_lab_d50, dtype=np.float64)
    reference_lab_d50 = np.asarray(reference_lab_d50, dtype=np.float64)

    # Per-patch ΔE00 — shape (48,)
    delta_e_per_patch = colour.delta_E(measured_lab_d50, reference_lab_d50, method="CIE 2000")
    delta_e_per_patch = np.asarray(delta_e_per_patch, dtype=np.float64).ravel()

    return {
        "mean": float(np.mean(delta_e_per_patch)),
        "median": float(np.median(delta_e_per_patch)),
        "p95": float(np.percentile(delta_e_per_patch, 95)),
    }


def _compute_delta_e_per_patch(
    measured_xyz_d65: np.ndarray,
    reference_xyz_d50: np.ndarray,
) -> np.ndarray:
    """Return per-patch ΔE00 array (shape (48,)) for panel-level logging.

    Applies the same Bradford CAT + Lab(D50) conversion as
    _compute_delta_e_stats. Used internally to split log output by panel.
    """
    measured_xyz_d50 = colour.adaptation.chromatic_adaptation_VonKries(
        measured_xyz_d65,
        _D65_XYZ,
        _D50_XYZ,
        transform="Bradford",
    )
    measured_lab_d50 = colour.XYZ_to_Lab(
        np.asarray(measured_xyz_d50, dtype=np.float64), illuminant=_D50_XY
    )
    reference_lab_d50 = colour.XYZ_to_Lab(reference_xyz_d50, illuminant=_D50_XY)
    de = colour.delta_E(
        np.asarray(measured_lab_d50, dtype=np.float64),
        np.asarray(reference_lab_d50, dtype=np.float64),
        method="CIE 2000",
    )
    return np.asarray(de, dtype=np.float64).ravel()


def _draw_contours_overlay(
    image_srgb: np.ndarray,
    quads: list[np.ndarray | None],
    seg_extra: dict | None = None,
) -> np.ndarray:
    """Draw detection overlay in the original image orientation.  Returns BGR uint8.

    Draws segmentation swatches (magenta) and clusters (cyan) from
    ``seg_extra`` on the library working image, then rotates back to
    the source orientation so the saved file looks natural.  Text
    labels are drawn *after* rotation so they read correctly.
    """
    is_portrait = image_srgb.shape[1] < image_srgb.shape[0]

    if seg_extra and "image_work" in seg_extra and "seg_data" in seg_extra:
        image_work_linear = np.copy(seg_extra["image_work"])
        seg_data = seg_extra["seg_data"]

        if hasattr(seg_data, "swatches") and seg_data.swatches is not None:
            cv.drawContours(
                image_work_linear,
                [as_int32_array(s) for s in seg_data.swatches],
                -1,
                (1, 0, 1),
                3,
            )
        if hasattr(seg_data, "clusters") and seg_data.clusters is not None:
            cv.drawContours(
                image_work_linear,
                [as_int32_array(c) for c in seg_data.clusters],
                -1,
                (0, 1, 1),
                3,
            )

        gamma = np.asarray(
            eotf_inverse_sRGB(np.clip(image_work_linear, 0.0, 1.0)), dtype=np.float32
        )
        bgr = cv.cvtColor(
            (gamma * 255.0 + 0.5).astype(np.uint8), cv.COLOR_RGB2BGR
        )
    else:
        bgr = _float_to_bgr_u8(image_srgb)
        h, w = bgr.shape[:2]
        if w < h:
            bgr = cv.rotate(bgr, cv.ROTATE_90_CLOCKWISE)
            h, w = bgr.shape[:2]
        new_h = int(round(h * _WORKING_WIDTH / w))
        bgr = cv.resize(bgr, (_WORKING_WIDTH, new_h), interpolation=cv.INTER_AREA)

    palette = [(0, 230, 0), (0, 165, 255), (230, 0, 230)]
    n_found = sum(1 for q in quads if q is not None)
    w_work = bgr.shape[1]

    # Phase 1 — geometry on the (possibly rotated) working image
    for idx, quad in enumerate(quads):
        if quad is None:
            continue
        color = palette[idx % len(palette)]
        pts = (
            np.round(np.asarray(quad, dtype=np.float64))
            .astype(np.int32)
            .reshape(-1, 1, 2)
        )
        cv.polylines(
            bgr, [pts], isClosed=True, color=color, thickness=4, lineType=cv.LINE_AA
        )
        for pt in pts.reshape(-1, 2):
            cv.circle(bgr, tuple(pt.tolist()), 10, color, -1, cv.LINE_AA)

    # Phase 2 — rotate back to original orientation
    if is_portrait:
        bgr = cv.rotate(bgr, cv.ROTATE_90_COUNTERCLOCKWISE)

    # Phase 3 — text labels in the final (natural) orientation
    def _to_final(pts_work: np.ndarray) -> np.ndarray:
        """Map working-image coords to the CCW-rotated output."""
        if not is_portrait:
            return pts_work
        out = np.empty_like(pts_work)
        out[:, 0] = pts_work[:, 1]
        out[:, 1] = w_work - 1 - pts_work[:, 0]
        return out

    for idx, quad in enumerate(quads):
        if quad is None:
            continue
        color = palette[idx % len(palette)]
        pts = np.round(np.asarray(quad, dtype=np.float64)).astype(np.int32).reshape(-1, 2)
        pts_f = _to_final(pts)
        for corner, pt in enumerate(pts_f):
            cv.putText(
                bgr, str(corner + 1), (pt[0] + 12, pt[1] + 6),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA,
            )
        centroid = pts_f.mean(axis=0).astype(int)
        label = f"Panel {idx + 1}/{n_found}"
        cv.putText(
            bgr, label, (int(centroid[0]) - 60, int(centroid[1]) - 15),
            cv.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv.LINE_AA,
        )

    h_final = bgr.shape[0]
    cv.putText(
        bgr,
        f"SCK300 - {n_found} panel(s) detected",
        (10, h_final - 14), cv.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv.LINE_AA,
    )
    return bgr


def _draw_cropped_card(item: DataDetectionColourChecker | None) -> np.ndarray | None:
    """Render the perspective-corrected card with swatch regions zeroed out.

    The library returns the card crop in linear sRGB at working resolution.
    We gamma-encode it for display and black out the swatch sampling regions
    (as in the colour-checker-detection example notebook).

    Args:
        item: DataDetectionColourChecker object or None.

    Returns:
        uint8 BGR array, or None if item lacks colour_checker data.
    """
    if item is None:
        return None
    card = getattr(item, "colour_checker", None)
    if card is None:
        return None

    masked = np.array(card, dtype=np.float32).copy()

    masks = getattr(item, "swatch_masks", None)
    if masks is not None:
        for m in masks:
            masked[m[0] : m[1], m[2] : m[3], ...] = 0.0

    gamma = np.asarray(
        eotf_inverse_sRGB(np.clip(masked, 0.0, 1.0)), dtype=np.float32
    )
    rgb_u8 = (gamma * 255.0 + 0.5).astype(np.uint8)
    return cv.cvtColor(rgb_u8, cv.COLOR_RGB2BGR)


def _draw_swatch_grid(swatches_linear_srgb: np.ndarray) -> np.ndarray:
    """Build an 8×6 grid of detected swatch colors with patch name labels.

    Column-major layout matching reference_data.py: A1–A6, B1–B6, …, H1–H6.
    Labels are white on dark patches, black on light patches.

    Args:
        swatches_linear_srgb: (48, 3) float64 linear sRGB swatch colours.

    Returns:
        uint8 BGR array, shape (6 * _VIZ_SWATCH_PX, 8 * _VIZ_SWATCH_PX, 3).
    """
    cols = _SCK300_SWATCHES_HORIZONTAL
    rows = _SCK300_SWATCHES_VERTICAL
    px = _VIZ_SWATCH_PX

    grid = np.zeros((rows * px, cols * px, 3), dtype=np.uint8)
    swatch_display = np.asarray(
        eotf_inverse_sRGB(np.clip(swatches_linear_srgb, 0.0, 1.0)), dtype=np.float64
    )

    patch_names = (
        list(SCK300_PATCH_NAMES) if hasattr(SCK300_PATCH_NAMES, "__iter__") else []
    )

    for i in range(_SCK300_SWATCHES):
        col_idx = i // rows
        row_idx = i % rows
        x0 = col_idx * px
        y0 = row_idx * px

        rgb = swatch_display[i]
        b = int(rgb[2] * 255 + 0.5)
        g = int(rgb[1] * 255 + 0.5)
        r = int(rgb[0] * 255 + 0.5)
        grid[y0 : y0 + px, x0 : x0 + px] = (b, g, r)

        label = patch_names[i] if i < len(patch_names) else f"P{i + 1}"
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = (0, 0, 0) if lum > 0.45 else (255, 255, 255)
        cv.putText(
            grid,
            label,
            (x0 + 4, y0 + px - 8),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv.LINE_AA,
        )

    return grid


def _draw_swatch_comparison(
    measured_xyz_d65: np.ndarray,
    reference_xyz_d50: np.ndarray,
    de_per_patch: np.ndarray | None = None,
) -> np.ndarray:
    """Build an 8x6 comparison grid with diagonal-split cells.

    Each cell is split by a diagonal from bottom-left to top-right.
    Left/bottom triangle: reference colour with patch name.
    Right/top triangle: measured colour with ΔE00 value.
    """
    cols = _SCK300_SWATCHES_HORIZONTAL
    rows = _SCK300_SWATCHES_VERTICAL
    px = _VIZ_SWATCH_PX

    grid = np.zeros((rows * px, cols * px, 3), dtype=np.uint8)

    meas_srgb = colour.XYZ_to_RGB(
        measured_xyz_d65, colour.RGB_COLOURSPACES["sRGB"],
        chromatic_adaptation_transform=None,
    )
    meas_srgb = np.clip(np.asarray(meas_srgb, dtype=np.float64), 0.0, 1.0)

    ref_xyz_d65 = colour.adaptation.chromatic_adaptation_VonKries(
        reference_xyz_d50, _D50_XYZ, _D65_XYZ, transform="Bradford",
    )
    ref_srgb = colour.XYZ_to_RGB(
        np.asarray(ref_xyz_d65, dtype=np.float64),
        colour.RGB_COLOURSPACES["sRGB"],
        chromatic_adaptation_transform=None,
    )
    ref_srgb = np.clip(np.asarray(ref_srgb, dtype=np.float64), 0.0, 1.0)

    meas_display = np.asarray(eotf_inverse_sRGB(meas_srgb), dtype=np.float64)
    ref_display = np.asarray(eotf_inverse_sRGB(ref_srgb), dtype=np.float64)

    patch_names = list(SCK300_PATCH_NAMES)

    for i in range(_SCK300_SWATCHES):
        col_idx = i // rows
        row_idx = i % rows
        x0 = col_idx * px
        y0 = row_idx * px

        m = meas_display[i]
        r = ref_display[i]
        bgr_m = (int(m[2] * 255 + 0.5), int(m[1] * 255 + 0.5), int(m[0] * 255 + 0.5))
        bgr_r = (int(r[2] * 255 + 0.5), int(r[1] * 255 + 0.5), int(r[0] * 255 + 0.5))

        grid[y0 : y0 + px, x0 : x0 + px] = bgr_m

        tri_ref = np.array([[0, 0], [0, px], [px, px]], dtype=np.int32) + [x0, y0]
        cv.fillPoly(grid, [tri_ref], bgr_r)

        label = patch_names[i] if i < len(patch_names) else ""
        lum_r = 0.299 * r[0] + 0.587 * r[1] + 0.114 * r[2]
        lc = (0, 0, 0) if lum_r > 0.45 else (220, 220, 220)
        cv.putText(
            grid, label, (x0 + 3, y0 + px - 5),
            cv.FONT_HERSHEY_SIMPLEX, 0.40, lc, 1, cv.LINE_AA,
        )

        if de_per_patch is not None:
            de_val = float(de_per_patch[i])
            if de_val < 5.0:
                dc = (30, 200, 30)
            elif de_val < 10.0:
                dc = (0, 140, 255)
            else:
                dc = (30, 30, 210)
            lum_m = 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]
            if lum_m > 0.55:
                dc = tuple(max(0, c - 40) for c in dc)
            cv.putText(
                grid, f"{de_val:.1f}", (x0 + px - 34, y0 + 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.35, dc, 1, cv.LINE_AA,
            )

    return grid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_and_measure(
    image_srgb: np.ndarray,
    show_steps: bool = False,
    save_steps: bool = False,
    log: bool = False,
    output_dir: Path | None = None,
) -> ColorCheckerResult:
    """Detect a DataColor Spyder Checkr Photo (SCK300) in *image_srgb* and measure
    the 48 swatch colours against the stored reference.

    Args:
        image_srgb: sRGB gamma float32 RGB [0, 1], shape (H, W, 3).  Same
            convention as ``detection.py`` and ``utils.load_image_rgb``.
        show_steps: If True, open two non-blocking OpenCV windows — (a)
            detection overlay, (b) 48-patch comparison grid.
        save_steps: If True, save debug visualisation images to ``output_dir``.
            When ``output_dir`` is None, logs a warning and skips saving
            (does NOT raise).
        log: If True, print structured progress to stdout via logger.info.
        output_dir: Directory for saved debug images; created automatically
            (``parents=True, exist_ok=True``) when it does not exist.
            Ignored when ``save_steps=False``.

    Returns:
        :class:`ColorCheckerResult`.  Never returns ``None``.
    """

    # ── Internal failure factory ─────────────────────────────────────────────
    def _fail(reason: str) -> ColorCheckerResult:
        if log:
            logger.info("FAILURE: %s", reason)
        return ColorCheckerResult(
            detected=False,
            failure_reason=reason,
            checker_type=None,
            n_swatches=None,
            swatch_colors_xyz=None,
            reference_colors_xyz=None,
            delta_e_stats=None,
            detection_image=None,
        )

    # ── Step 1: Detection ────────────────────────────────────────────────────
    if log:
        logger.info("=== DETEKCIJA ===")

    try:
        detections, quadrilaterals, raw_items, _, seg_extra = _run_segmentation_v2(image_srgb)
    except Exception as exc:
        return _fail(f"colorchecker_detection: library error -- {exc}")

    if len(detections) == 0:
        return _fail("colorchecker_detection: no quadrilateral found")

    # Keep ALL pre-merge quads and items for the contours visualization.
    # After merging/selection below, quadrilaterals[0] may drop one panel's quad.
    viz_quads: list[np.ndarray | None] = list(quadrilaterals)
    viz_items: list[DataDetectionColourChecker | None] = list(raw_items)

    # ── Step 2: Orientation check ────────────────────────────────────────────
    _check_orientation(quadrilaterals[0], log=log)

    swatches_raw_linear_srgb = detections[0]

    # ── Step 3: Validate swatch array shape ──────────────────────────────────
    if swatches_raw_linear_srgb.ndim != 2 or swatches_raw_linear_srgb.shape[1] != 3:
        return _fail(
            f"colorchecker_detection: unexpected swatch array shape {swatches_raw_linear_srgb.shape}"
        )

    n_detected = swatches_raw_linear_srgb.shape[0]
    if n_detected != _SCK300_SWATCHES:
        return _fail(
            f"colorchecker_detection: expected {_SCK300_SWATCHES} swatches, got {n_detected}"
        )

    swatches_linear_srgb = swatches_raw_linear_srgb.astype(np.float64)

    # ── Step 4: Panel saturation verification ────────────────────────────────
    # Check if left 24 (neutral/skin A–D) has lower chroma than right 24 (saturated E–H).
    # If inverted, apply _reorder_swatches() to fix the column-major order.
    order_correct = _check_panel_saturation_order(swatches_linear_srgb)
    if not order_correct:
        if log:
            logger.info(
                "Panel saturation check FAILED — left panel appears more saturated "
                "than right panel. Applying _reorder_swatches() to correct order."
            )
        swatches_linear_srgb = _reorder_swatches(swatches_linear_srgb)
    else:
        if log:
            logger.info(
                "Panel saturation check OK — left panel is less saturated (neutral/skin)."
            )

    # ── Step 5: OQ#1 Verification ────────────────────────────────────────────
    # Convert first swatch from linear sRGB → XYZ(D65) → XYZ(D50) → Lab(D50)
    # and compare with A1 reference [61.35, 34.81, 18.38].
    first_xyz_d65 = _swatch_srgb_to_xyz(swatches_linear_srgb[:1])
    first_xyz_d50 = colour.adaptation.chromatic_adaptation_VonKries(
        first_xyz_d65, _D65_XYZ, _D50_XYZ, transform="Bradford"
    )
    first_lab = colour.XYZ_to_Lab(
        np.asarray(first_xyz_d50, dtype=np.float64), illuminant=_D50_XY
    )
    first_lab = np.asarray(first_lab, dtype=np.float64).ravel()

    if log:
        logger.info(
            "=== MJERENJE === (detected %d swatches)",
            n_detected,
        )
        logger.info(
            "OQ#1 — First swatch Lab:      L=%.2f  a=%.2f  b=%.2f",
            first_lab[0],
            first_lab[1],
            first_lab[2],
        )
        logger.info(
            "OQ#1 — A1 reference Lab:      L=%.2f  a=%.2f  b=%.2f",
            _A1_LAB_REFERENCE[0],
            _A1_LAB_REFERENCE[1],
            _A1_LAB_REFERENCE[2],
        )
        de_a1 = float(
            colour.delta_E(
                first_lab.reshape(1, 3),
                _A1_LAB_REFERENCE.reshape(1, 3),
                method="CIE 2000",
            ).ravel()[0]
        )
        logger.info("OQ#1 — ΔE00(first swatch, A1 ref) = %.2f", de_a1)
        if de_a1 > 10.0:
            logger.warning(
                "OQ#1 WARNING: ΔE00 = %.2f > 10 — first swatch may not correspond "
                "to patch A1. Consider adding manual _reorder_swatches() if "
                "visual inspection confirms wrong order. If detected via two-panel "
                "(Option A portrait pass), also try swapping swatches_horizontal/"
                "swatches_vertical in _PANEL_SEGMENTER_KWARGS_A (4×6 → 6×4).",
                de_a1,
            )

    # ── Step 6: sRGB → XYZ ──────────────────────────────────────────────────
    measured_xyz_d65 = _swatch_srgb_to_xyz(swatches_linear_srgb)

    if not np.all(np.isfinite(measured_xyz_d65)):
        return _fail(
            "colorchecker_measurement: XYZ conversion produced non-finite values"
        )

    # ── Step 7: Load reference ───────────────────────────────────────────────
    reference_xyz_d50 = SCK300_REFERENCE_XYZ  # shape (48, 3), float64, XYZ D50

    # ── Step 8: ΔE00 with split panel logging ────────────────────────────────
    if log:
        logger.info("=== DELTA-E ===")

    delta_e_stats = _compute_delta_e_stats(measured_xyz_d65, reference_xyz_d50)

    if log:
        # Compute per-patch ΔE00 for panel-level breakdown
        de_per_patch = _compute_delta_e_per_patch(measured_xyz_d65, reference_xyz_d50)
        de_left = de_per_patch[:24]  # columns A–D (neutral/skin)
        de_right = de_per_patch[24:]  # columns E–H (saturated)

        logger.info(
            "=== LIJEVI PANEL (A–D, skin/neutral) ===  "
            "mean=%.2f  median=%.2f  p95=%.2f",
            float(np.mean(de_left)),
            float(np.median(de_left)),
            float(np.percentile(de_left, 95)),
        )
        logger.info(
            "=== DESNI PANEL (E–H, saturirane) ===  "
            "mean=%.2f  median=%.2f  p95=%.2f",
            float(np.mean(de_right)),
            float(np.median(de_right)),
            float(np.percentile(de_right, 95)),
        )
        logger.info(
            "=== UKUPNO (48 patcheva) ===  " "mean=%.2f  median=%.2f  p95=%.2f",
            delta_e_stats["mean"],
            delta_e_stats["median"],
            delta_e_stats["p95"],
        )

    # ── Step 9: Build visualizations ────────────────────────────────────────
    # Compute per-patch ΔE00 for use in the comparison grid labels.
    de_per_patch = _compute_delta_e_per_patch(measured_xyz_d65, reference_xyz_d50)

    detection_image_bgr = _draw_contours_overlay(image_srgb, viz_quads, seg_extra)
    comparison_bgr = _draw_swatch_comparison(measured_xyz_d65, reference_xyz_d50, de_per_patch)

    # ── Step 10: show/save debug visualisations ───────────────────────────────
    if show_steps:
        cv.imshow("SCK300 — contours on scene", detection_image_bgr)
        cv.imshow(
            "SCK300 — swatch comparison (top=measured, bottom=ref)", comparison_bgr
        )
        cv.waitKey(1)

    if save_steps:
        if output_dir is None:
            logger.warning("save_steps=True but output_dir is None — skipping file output")
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            _save_bgr(out_dir / _FILENAME_VIZ_CONTOURS, detection_image_bgr)

            saved = [_FILENAME_VIZ_CONTOURS]

            left_img = _draw_cropped_card(viz_items[0] if len(viz_items) > 0 else None)
            if left_img is not None:
                _save_bgr(out_dir / _FILENAME_VIZ_LEFT_PANEL, left_img)
                saved.append(_FILENAME_VIZ_LEFT_PANEL)

            right_img = _draw_cropped_card(viz_items[1] if len(viz_items) > 1 else None)
            if right_img is not None:
                _save_bgr(out_dir / _FILENAME_VIZ_RIGHT_PANEL, right_img)
                saved.append(_FILENAME_VIZ_RIGHT_PANEL)

            _save_bgr(
                out_dir / _FILENAME_VIZ_SWATCHES, _draw_swatch_grid(swatches_linear_srgb)
            )
            saved.append(_FILENAME_VIZ_SWATCHES)

            _save_bgr(out_dir / _FILENAME_VIZ_COMPARISON, comparison_bgr)
            saved.append(_FILENAME_VIZ_COMPARISON)

            if log:
                logger.info(
                    "Saved %d visualisations to %s: %s",
                    len(saved),
                    out_dir,
                    ", ".join(saved),
                )

    # ── Step 11: Return success result ───────────────────────────────────────
    return ColorCheckerResult(
        detected=True,
        failure_reason=None,
        checker_type=_CHECKER_TYPE,
        n_swatches=_SCK300_SWATCHES,
        swatch_colors_xyz=measured_xyz_d65,
        reference_colors_xyz=reference_xyz_d50,
        delta_e_stats=delta_e_stats,
        detection_image=detection_image_bgr,
    )
