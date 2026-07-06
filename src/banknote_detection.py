"""
Banknote detection and colour measurement.

Wraps :func:`~src.detection.detect_banknote` (SIFT + RANSAC homography)
with banknote-specific logic: reference image selection, grid sampling
of the warped result, and pairing with pre-computed reference samples.

The result is a :class:`BanknoteResult` analogous to
:class:`~src.colorchecker.ColorCheckerResult` — the downstream
calibration pipeline treats both identically.

Pipeline
--------
1. Load the reference PNG for the given denomination and side.
2. Run SIFT + RANSAC detection → homography → perspective warp.
3. Grid-sample the warped banknote at a chosen cell size.
4. Load pre-computed reference samples and exclusion mask.
5. Apply the same mask to both measured and reference samples.
6. Return paired (N, 3) sRGB arrays ready for calibration.

Colour-space convention
-----------------------
All sample arrays are **sRGB float32 [0, 1]**.  Conversion to XYZ-D65
for calibration is the caller's responsibility (via
:func:`~src.color_calibration.srgb_to_xyz` on the flattened arrays).

Visualisation steps (saved with ``save_steps=True``):
    viz_01_matches.jpg            ← inlier/outlier feature matches
    viz_02_detection.jpg          ← banknote quad on original scene
    viz_03_warped.jpg             ← scene warped to reference frame
    viz_04_comparison.jpg         ← reference vs warped side-by-side
    viz_05_grid_{cell_size}.png   ← sampling grid with exclusions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from src.detection import (
    DetectionConfig,
    DetectionResult,
    detect_banknote,
    project_corners_to_scene,
    warp_scene_to_reference,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tuned DetectionConfig for banknote detection.
# The reference banknote PNGs are 512px wide while scenes are 4000–6000px.
# The large scale ratio inflates the SVD condition number (σ_max/σ_min) of
# the 3×3 homography to 14M–22M for valid detections.  The standard
# threshold (5000) and the previous banknote override (1e7) both reject
# these.  Empirically verified on 10 denomination×side combinations:
#   min=13.8M, max=21.6M, all with RMSE < 2.5 px and 50–122 inliers.
# 5e7 gives ~2× headroom.  Inlier count, inlier ratio, quad geometry,
# and area fraction checks still guard against degenerate detections.
_BANKNOTE_DETECTION_CONFIG = DetectionConfig(max_condition_number=5e7)

# Euro banknote physical dimensions (ES2 Europa series), millimetres.
# Source: ECB "Design elements".
_EURO_DIMENSIONS_MM: dict[int, tuple[float, float]] = {
    5: (120.0, 62.0),
    10: (127.0, 67.0),
    20: (133.0, 72.0),
    50: (140.0, 77.0),
    100: (147.0, 77.0),
}

# Crop margin (pixels) around the initial SIFT detection quad when running
# the second-pass SIFT for more inliers at the banknote's scale.
_CROP_MARGIN: int = 150

# Output filenames
_FILENAME_VIZ_MATCHES: str = "viz_01_matches.jpg"
_FILENAME_VIZ_DETECTION: str = "viz_02_detection.jpg"
_FILENAME_VIZ_WARPED: str = "viz_03_warped.jpg"
_FILENAME_VIZ_COMPARISON: str = "viz_04_comparison.jpg"
_FILENAME_VIZ_GRID: str = "viz_05_grid_{cell_size}.png"

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BanknoteResult:
    """Structured output of the banknote detection and sampling pipeline.

    On success (``detected=True``) all measurement fields are populated.
    On failure only ``detected`` and ``failure_reason`` are set; others
    are ``None``.
    """

    detected: bool
    failure_reason: str | None

    denomination: int | None          # 5, 10, 20, 50, 100
    side: str | None                  # "single" or "double"

    # Paired colour samples — sRGB float32 [0, 1], shape (N, 3)
    measured_srgb: np.ndarray | None  # from the scene (warped)
    reference_srgb: np.ndarray | None  # from pre-computed reference

    n_samples: int | None             # N — number of valid paired cells

    # Detection statistics (from SIFT + RANSAC)
    n_inliers: int | None
    confidence: float | None
    reproj_rmse: float | None

    # Warped scene image — sRGB float32 [0, 1], same dims as reference
    warped_srgb: np.ndarray | None

    # Full detection result for visualization / debugging
    detection_result: DetectionResult | None


# ---------------------------------------------------------------------------
# Private helpers — drawing / IO
# ---------------------------------------------------------------------------


def _float_to_bgr_u8(image_srgb: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0, 1] to uint8 BGR for OpenCV drawing functions."""
    rgb_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv.cvtColor(rgb_u8, cv.COLOR_RGB2BGR)


def _save_bgr(path: Path, bgr: np.ndarray) -> None:
    """Save a BGR uint8 image to *path* (creates parent directories)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), bgr)


def _show(title: str, bgr: np.ndarray) -> None:
    """Display a BGR image in a non-blocking OpenCV window."""
    cv.imshow(title, bgr)
    cv.waitKey(1)


# ---------------------------------------------------------------------------
# Visualisation — one function per step
# ---------------------------------------------------------------------------


def _draw_matches(det: DetectionResult) -> np.ndarray:
    """Draw RANSAC inlier matches (green) and outliers (red).

    Returns a BGR uint8 image, or an empty placeholder when no matches
    are available.
    """
    if not det.good_matches or det.inlier_mask is None:
        # Minimal placeholder — side-by-side grayscale
        h = max(det.scene_gray.shape[0], det.reference_gray.shape[0])
        w = det.scene_gray.shape[1] + det.reference_gray.shape[1]
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        cv.putText(placeholder, "No matches", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
        return placeholder

    inlier_list = det.inlier_mask.tolist()
    img = cv.drawMatches(
        det.scene_gray, det.keypoints_scene,
        det.reference_gray, det.keypoints_ref,
        det.good_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(80, 80, 80),
        matchesMask=inlier_list,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    outlier_list = (1 - det.inlier_mask).tolist()
    img = cv.drawMatches(
        det.scene_gray, det.keypoints_scene,
        det.reference_gray, det.keypoints_ref,
        det.good_matches, img,
        matchColor=(0, 0, 255),
        singlePointColor=None,
        matchesMask=outlier_list,
        flags=(cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
               | cv.DrawMatchesFlags_DRAW_OVER_OUTIMG),
    )
    return img


def _draw_detection(det: DetectionResult) -> np.ndarray:
    """Draw the detected banknote quadrilateral on the original scene.

    Returns BGR uint8 image with quad and stats overlay.
    """
    scene_bgr = _float_to_bgr_u8(det.scene_img)

    if det.scene_corners is not None:
        pts = det.scene_corners.astype(np.int32).reshape(-1, 1, 2)
        color = (0, 255, 0) if det.success else (0, 200, 255)
        cv.polylines(scene_bgr, [pts], isClosed=True, color=color, thickness=3)

        for idx, (x, y) in enumerate(det.scene_corners.astype(int)):
            cv.circle(scene_bgr, (x, y), 6, color, -1)
            cv.putText(scene_bgr, str(idx + 1), (x + 8, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)

    status = (
        f"Inliers: {det.n_inliers}/{det.n_matches_lowe}  "
        f"RMSE: {det.reproj_rmse:.1f}px  "
        f"Conf: {det.confidence:.2f}"
    )
    cv.putText(scene_bgr, status, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

    return scene_bgr


def _draw_comparison(
    reference_srgb: np.ndarray,
    warped_srgb: np.ndarray,
) -> np.ndarray:
    """Build a side-by-side comparison: reference | warped scene.

    Returns BGR uint8 image with a white separator and labels.
    """
    ref_bgr = _float_to_bgr_u8(reference_srgb)
    warped_bgr = _float_to_bgr_u8(warped_srgb)

    h = max(ref_bgr.shape[0], warped_bgr.shape[0])
    sep = 4  # white separator width

    # Pad to same height
    def _pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] >= target_h:
            return img
        pad = np.full((target_h - img.shape[0], img.shape[1], 3), 255,
                      dtype=np.uint8)
        return np.vstack([img, pad])

    ref_padded = _pad_h(ref_bgr, h)
    warped_padded = _pad_h(warped_bgr, h)
    separator = np.full((h, sep, 3), 255, dtype=np.uint8)

    canvas = np.hstack([ref_padded, separator, warped_padded])

    # Labels
    label_h = 30
    header = np.full((label_h, canvas.shape[1], 3), 40, dtype=np.uint8)
    ref_w = ref_padded.shape[1]
    cv.putText(header, "Reference", (10, 22),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(header, "Detected (warped)", (ref_w + sep + 10, 22),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return np.vstack([header, canvas])


# ---------------------------------------------------------------------------
# Homography refinement — crop re-detection + AR correction
# ---------------------------------------------------------------------------


def _crop_and_redetect(
    image_srgb: np.ndarray,
    reference_srgb: np.ndarray,
    initial_corners: np.ndarray,
) -> DetectionResult | None:
    """Re-run SIFT on a scene crop around the initial detection.

    Cropping lets SIFT extract features at the banknote's own scale,
    producing more inliers (especially in texture-poor regions).
    Returns a new :class:`DetectionResult` whose ``src_pts`` are in
    **full-scene** coordinates, or ``None`` if the second pass fails.
    """
    sh, sw = image_srgb.shape[:2]
    m = _CROP_MARGIN
    x1 = max(0, int(initial_corners[:, 0].min()) - m)
    y1 = max(0, int(initial_corners[:, 1].min()) - m)
    x2 = min(sw, int(initial_corners[:, 0].max()) + m)
    y2 = min(sh, int(initial_corners[:, 1].max()) + m)

    crop = image_srgb[y1:y2, x1:x2]
    det2 = detect_banknote(crop, reference_srgb, _BANKNOTE_DETECTION_CONFIG)
    if not det2.success:
        return None

    # Translate the crop-local homography back to full-scene coordinates.
    # H_crop maps crop→ref; to get scene→ref we pre-compose with the
    # translation T that maps scene→crop: p_crop = p_scene - (x1, y1).
    T = np.eye(3, dtype=np.float64)
    T[0, 2] = -x1
    T[1, 2] = -y1
    H_scene = det2.homography @ T
    H_scene /= H_scene[2, 2]

    # Patch the result so callers see full-scene geometry.
    det2.homography = H_scene
    ref_shape = reference_srgb.shape[:2]
    det2.scene_corners = project_corners_to_scene(H_scene, ref_shape)
    det2.warped_scene = warp_scene_to_reference(image_srgb, H_scene, ref_shape)
    # Keep original scene image for visualisations
    det2.scene_img = image_srgb
    return det2


def _correct_homography_ar(
    det: DetectionResult,
    ref_shape: tuple[int, int],
    known_ar: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct a SIFT homography so the scene quad has the known aspect ratio.

    SIFT inliers are often concentrated in one half of the banknote,
    leaving the opposite pair of corners poorly constrained.  This
    function keeps the well-constrained "bottom" edge fixed (the one
    closer to the inlier centroid) and scales the "left" and "right"
    edge vectors so the resulting quadrilateral has the correct AR.

    Parameters
    ----------
    det : DetectionResult with ``scene_corners`` set.
    ref_shape : (ref_h, ref_w).
    known_ar : physical width / height of the banknote.

    Returns
    -------
    (H_corrected, corners_corrected)
        H_corrected : (3, 3) float64 — scene → reference homography.
        corners_corrected : (4, 2) float32 — corrected scene corners
        in the same order as ``det.scene_corners``.
    """
    ref_h, ref_w = ref_shape
    # Corner order from detection.py: [TL, TR, BR, BL] in the reference frame.
    cTL, cTR, cBR, cBL = det.scene_corners

    # Identify the well-constrained edge.  The bottom edge in the
    # reference (BL→BR) has the most inliers, so we keep it fixed.
    bottom_len = float(np.linalg.norm(cBR - cBL))
    top_len = float(np.linalg.norm(cTR - cTL))
    avg_width = (bottom_len + top_len) / 2.0

    left_len = float(np.linalg.norm(cTL - cBL))
    right_len = float(np.linalg.norm(cTR - cBR))
    avg_height = (left_len + right_len) / 2.0

    target_height = avg_width / known_ar
    scale = target_height / avg_height if avg_height > 0 else 1.0

    # Scale both "vertical" edges (BL→TL, BR→TR) by the correction factor.
    cTL_fixed = cBL + (cTL - cBL) * scale
    cTR_fixed = cBR + (cTR - cBR) * scale

    corners = np.array([cTL_fixed, cTR_fixed, cBR, cBL], dtype=np.float32)
    ref_corners = np.array(
        [[0, 0], [ref_w - 1, 0], [ref_w - 1, ref_h - 1], [0, ref_h - 1]],
        dtype=np.float32,
    )
    H = cv.getPerspectiveTransform(corners, ref_corners)
    return H.astype(np.float64), corners


# ---------------------------------------------------------------------------
# Reference data loading
# ---------------------------------------------------------------------------


def load_reference_image_srgb(
    ref_dir: Path,
    denomination: int,
    side: str,
) -> np.ndarray | None:
    """Load a reference banknote PNG as sRGB float32 [0, 1].

    Tries ``<ref_dir>/<denom>_<side>.png``.
    Returns ``None`` if the file does not exist or cannot be decoded.
    """
    ref_path = ref_dir / f"{denomination}_{side}.png"
    if not ref_path.exists():
        logger.warning("Reference image not found: %s", ref_path)
        return None

    bgr = cv.imread(str(ref_path))
    if bgr is None:
        logger.warning("Failed to decode: %s", ref_path)
        return None

    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_reference_samples(
    ref_dir: Path,
    denomination: int,
    side: str,
    cell_size: int = 32,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load pre-computed reference samples and mask for a banknote.

    Reads ``<ref_dir>/<denom>_<side>/samples.json`` and
    ``<ref_dir>/<denom>_<side>/masks.json`` for the given cell size.

    Returns
    -------
    (samples_srgb, valid_mask) or ``None`` if files are missing.
        samples_srgb : (N, 3) float32 — all grid cells, row-major.
        valid_mask : (N,) bool — True = valid cell.
    """
    stem = f"{denomination}_{side}"
    samples_path = ref_dir / stem / "samples.json"
    masks_path = ref_dir / stem / "masks.json"

    if not samples_path.exists() or not masks_path.exists():
        logger.warning(
            "Reference samples/masks not found for %s (cell_size=%d)",
            stem, cell_size,
        )
        return None

    key = str(cell_size)

    with open(samples_path) as f:
        all_samples = json.load(f)
    with open(masks_path) as f:
        all_masks = json.load(f)

    if key not in all_samples or key not in all_masks:
        logger.warning(
            "Cell size %d not found in reference data for %s", cell_size, stem,
        )
        return None

    samples_srgb = np.array(all_samples[key], dtype=np.float32)
    valid_mask = np.array(all_masks[key], dtype=bool)

    return samples_srgb, valid_mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_and_measure_banknote(
    image_srgb: np.ndarray,
    denomination: int,
    side: str,
    ref_dir: Path,
    cell_size: int = 32,
    *,
    show_steps: bool = False,
    save_steps: bool = False,
    log: bool = False,
    output_dir: Path | None = None,
) -> BanknoteResult:
    """Detect a banknote in the scene and measure its colours.

    Parameters
    ----------
    image_srgb : (H, W, 3) float32 sRGB [0, 1].
        Full scene photograph containing the banknote.
    denomination : 5, 10, 20, 50, or 100.
    side : ``"single"`` or ``"double"``.
    ref_dir : path to the reference directory (contains PNGs and
        per-banknote sample folders).
    cell_size : grid cell size for sampling (default 32).
    show_steps : If True, open non-blocking OpenCV windows for each step.
    save_steps : If True, save debug visualisations to *output_dir*.
        When ``output_dir`` is None, logs a warning and skips saving
        (does NOT raise).
    log : If True, emit structured progress via ``logger.info``.
    output_dir : directory for saved debug images; created automatically.
        Ignored when ``save_steps=False``.

    Returns
    -------
    :class:`BanknoteResult`.  Check ``result.detected`` before using
    measurement fields.
    """
    from src.banknote_sampling import draw_sampling_grid, sample_banknote_grid

    ref_name = f"{denomination}_{side}"

    def _fail(reason: str) -> BanknoteResult:
        if log:
            logger.info("Banknote FAIL: %s", reason)
        return BanknoteResult(
            detected=False,
            failure_reason=reason,
            denomination=denomination,
            side=side,
            measured_srgb=None,
            reference_srgb=None,
            n_samples=None,
            n_inliers=None,
            confidence=None,
            reproj_rmse=None,
            warped_srgb=None,
            detection_result=None,
        )

    if save_steps and output_dir is None:
        logger.warning("save_steps=True but output_dir is None — skipping saves")

    out = Path(output_dir) if output_dir else None

    # ── 1. Load reference image ───────────────────────────────────────

    if log:
        logger.info("=== BANKNOTE DETECTION: %s ===", ref_name)

    ref_image_srgb = load_reference_image_srgb(ref_dir, denomination, side)
    if ref_image_srgb is None:
        return _fail(f"reference image not found: {ref_name}.png")

    if log:
        logger.info(
            "  Reference loaded: %s (%dx%d)",
            ref_name, ref_image_srgb.shape[1], ref_image_srgb.shape[0],
        )

    # ── 2. SIFT + RANSAC detection (pass 1: full scene) ──────────────

    det = detect_banknote(image_srgb, ref_image_srgb, _BANKNOTE_DETECTION_CONFIG)

    if log:
        logger.info(
            "  SIFT pass 1: kp_scene=%d, kp_ref=%d, "
            "matches_raw=%d, matches_lowe=%d",
            det.n_keypoints_scene, det.n_keypoints_ref,
            det.n_matches_raw, det.n_matches_lowe,
        )
        if det.n_inliers > 0:
            logger.info(
                "  RANSAC pass 1: inliers=%d/%d (%.1f%%), RMSE=%.2f px",
                det.n_inliers, det.n_matches_lowe,
                100.0 * det.inlier_ratio, det.reproj_rmse,
            )

    # ── Viz 01: Feature matches ───────────────────────────────────────

    if (save_steps or show_steps) and det.good_matches:
        matches_bgr = _draw_matches(det)
        if save_steps and out:
            _save_bgr(out / _FILENAME_VIZ_MATCHES, matches_bgr)
        if show_steps:
            _show(f"Banknote matches — {ref_name}", matches_bgr)

    if not det.success:
        if save_steps or show_steps:
            detection_bgr = _draw_detection(det)
            if save_steps and out:
                _save_bgr(out / _FILENAME_VIZ_DETECTION, detection_bgr)
            if show_steps:
                _show(f"Banknote detection — {ref_name}", detection_bgr)
        return _fail(f"detection failed: {det.message}")

    # ── 2b. Re-detect on crop for more inliers (pass 2) ──────────────

    det2 = _crop_and_redetect(image_srgb, ref_image_srgb, det.scene_corners)
    if det2 is not None:
        if log:
            logger.info(
                "  SIFT pass 2 (crop): inliers=%d/%d (%.1f%%), RMSE=%.2f px",
                det2.n_inliers, det2.n_matches_lowe,
                100.0 * det2.inlier_ratio, det2.reproj_rmse,
            )
        det = det2
    elif log:
        logger.info("  Pass 2 (crop) failed — keeping pass 1 result")

    # ── 2c. AR correction using known banknote dimensions ─────────────

    ref_shape = ref_image_srgb.shape[:2]
    dims = _EURO_DIMENSIONS_MM.get(denomination)
    if dims is not None:
        known_ar = dims[0] / dims[1]
        H_corr, corners_corr = _correct_homography_ar(det, ref_shape, known_ar)
        warped_srgb = warp_scene_to_reference(
            image_srgb, H_corr, ref_shape,
        )
        det.homography = H_corr
        det.scene_corners = corners_corr
        det.warped_scene = warped_srgb
        # Recompute RMSE and confidence for the corrected homography
        if det.src_pts is not None and det.dst_pts is not None:
            from src.detection import compute_reprojection_rmse, compute_confidence
            det.reproj_rmse = compute_reprojection_rmse(
                H_corr, det.src_pts, det.dst_pts, None,  # all matched points
            )
            det.confidence = compute_confidence(
                det.n_inliers, det.n_matches_lowe,
                det.reproj_rmse, 5.0,  # default RANSAC threshold
            )
        if log:
            logger.info("  AR correction applied (known AR=%.3f)", known_ar)
    else:
        if log:
            logger.info("  No known dimensions for %d€ — skipping AR correction",
                         denomination)

    if log:
        logger.info(
            "  Detection OK — confidence=%.2f", det.confidence,
        )

    warped_srgb = det.warped_scene  # float32 RGB [0,1], same dims as ref

    # ── Viz 02: Detection overlay ─────────────────────────────────────

    if save_steps or show_steps:
        detection_bgr = _draw_detection(det)
        if save_steps and out:
            _save_bgr(out / _FILENAME_VIZ_DETECTION, detection_bgr)
        if show_steps:
            _show(f"Banknote detection — {ref_name}", detection_bgr)

    # ── Viz 03: Warped scene ──────────────────────────────────────────

    if save_steps or show_steps:
        warped_bgr = _float_to_bgr_u8(warped_srgb)
        if save_steps and out:
            _save_bgr(out / _FILENAME_VIZ_WARPED, warped_bgr)
        if show_steps:
            _show(f"Banknote warped — {ref_name}", warped_bgr)

    # ── Viz 04: Side-by-side comparison ───────────────────────────────

    if save_steps or show_steps:
        comparison_bgr = _draw_comparison(ref_image_srgb, warped_srgb)
        if save_steps and out:
            _save_bgr(out / _FILENAME_VIZ_COMPARISON, comparison_bgr)
        if show_steps:
            _show(f"Banknote comparison — {ref_name}", comparison_bgr)

    # ── 3. Load pre-computed reference samples ────────────────────────

    ref_data = load_reference_samples(ref_dir, denomination, side, cell_size)
    if ref_data is None:
        return _fail(
            f"reference samples missing for {ref_name} cell_size={cell_size}"
        )

    ref_samples_srgb, ref_valid_mask = ref_data

    # ── 4. Sample the warped scene ────────────────────────────────────

    measured_samples_srgb, n_rows, n_cols = sample_banknote_grid(
        warped_srgb, cell_size=cell_size,
    )

    if measured_samples_srgb.shape[0] != ref_samples_srgb.shape[0]:
        return _fail(
            f"grid mismatch: measured {measured_samples_srgb.shape[0]} cells "
            f"vs reference {ref_samples_srgb.shape[0]} cells "
            f"(warped {warped_srgb.shape[1]}x{warped_srgb.shape[0]}, "
            f"ref {ref_image_srgb.shape[1]}x{ref_image_srgb.shape[0]})"
        )

    # ── 5. Apply mask — same cells excluded on both sides ─────────────

    measured_valid_srgb = measured_samples_srgb[ref_valid_mask]
    reference_valid_srgb = ref_samples_srgb[ref_valid_mask]
    n_valid = int(np.count_nonzero(ref_valid_mask))

    if n_valid == 0:
        return _fail("all grid cells excluded by mask — 0 valid samples")

    if log:
        n_total = n_rows * n_cols
        logger.info(
            "  Sampling: %dx%d grid (%d cells), %d valid, %d excluded",
            n_rows, n_cols, n_total, n_valid, n_total - n_valid,
        )
        logger.info(
            "  Warped dims: %dx%d", warped_srgb.shape[1], warped_srgb.shape[0],
        )

    # ── Viz 05: Sampling grid ─────────────────────────────────────────

    if save_steps or show_steps:
        viz_grid_bgr = draw_sampling_grid(
            warped_srgb, measured_samples_srgb, n_rows, n_cols,
            valid_mask=ref_valid_mask, cell_size=cell_size,
        )
        if save_steps and out:
            grid_name = _FILENAME_VIZ_GRID.format(cell_size=cell_size)
            _save_bgr(out / grid_name, viz_grid_bgr)
        if show_steps:
            _show(f"Banknote grid — {ref_name}", viz_grid_bgr)

    # ── 6. Return result ──────────────────────────────────────────────

    if log:
        logger.info("  Banknote OK — %d paired samples", n_valid)

    return BanknoteResult(
        detected=True,
        failure_reason=None,
        denomination=denomination,
        side=side,
        measured_srgb=measured_valid_srgb,
        reference_srgb=reference_valid_srgb,
        n_samples=n_valid,
        n_inliers=det.n_inliers,
        confidence=det.confidence,
        reproj_rmse=det.reproj_rmse,
        warped_srgb=warped_srgb,
        detection_result=det,
    )
