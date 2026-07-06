"""
detection.py — Production-quality banknote detection using SIFT + RANSAC homography.

The pipeline detects a euro banknote in a scene photograph by matching it against
a clean reference image. It returns a DetectionResult that carries the homography,
warped banknote, matching statistics, and all intermediate data needed for
visualisation and downstream color calibration.

Pipeline:
    1. Convert float RGB inputs to uint8 grayscale.
    2. Enhance contrast with CLAHE for robust feature detection.
    3. Detect SIFT keypoints and compute 128-dim L2 descriptors.
    4. Match descriptors via FLANN KD-tree with Lowe's ratio test (threshold 0.7).
    5. Estimate a perspective homography using RANSAC.
    6. Validate the homography (condition number, corner geometry, area fraction).
    7. Warp the scene image to the reference frame.
    8. Compute a detection confidence score.

Usage::

    from pathlib import Path
    from detection import detect_banknote, save_detection_steps
    from utils import load_image_rgb

    scene_img = load_image_rgb(Path("data/train/IMG_2611.jpeg"))
    reference_img = load_image_rgb(Path("data/ref/images/egp_100.jpg"))

    result = detect_banknote(scene_img, reference_img)
    if result.success:
        save_detection_steps(result, Path("output/detection"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default algorithm parameters
# ---------------------------------------------------------------------------

# SIFT: lower contrastThreshold than the OpenCV default (0.04) to detect more
# keypoints when the banknote occupies a small fraction of the scene.
_SIFT_N_FEATURES: int = 8_000
_SIFT_CONTRAST_THRESHOLD: float = 0.02
_SIFT_EDGE_THRESHOLD: int = 10
_SIFT_SIGMA: float = 1.6

# FLANN: KD-tree index for float (SIFT) descriptors.
_FLANN_TREES: int = 5
_FLANN_CHECKS: int = 100

# Lowe's ratio test threshold — 0.7 rejects ~10% of correct matches but
# eliminates the vast majority of false positives.
_LOWE_RATIO: float = 0.70

# RANSAC: reprojection error tolerance (pixels) and solver parameters.
_RANSAC_REPROJ_THRESHOLD: float = 5.0
_RANSAC_CONFIDENCE: float = 0.995
_RANSAC_MAX_ITERS: int = 2_000

# Acceptance thresholds for a valid detection.
_MIN_INLIERS: int = 10
_MIN_INLIER_RATIO: float = 0.15

# Homography geometry validation.
_MAX_CONDITION_NUMBER: float = 5_000.0   # SVD σ_max / σ_min
_MIN_AREA_FRACTION: float = 0.005        # detected quad ≥ 0.5 % of scene area
_MAX_AREA_FRACTION: float = 0.98         # detected quad ≤ 98 % of scene area
_MIN_CORNER_ANGLE_DEG: float = 10.0      # all interior angles ≥ 10 °


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DetectionConfig:
    """Tunable parameters for the banknote detection pipeline.

    All fields have production defaults. Override only what you need to change.
    """

    # SIFT detector
    sift_n_features: int = _SIFT_N_FEATURES
    sift_contrast_threshold: float = _SIFT_CONTRAST_THRESHOLD
    sift_edge_threshold: int = _SIFT_EDGE_THRESHOLD
    sift_sigma: float = _SIFT_SIGMA

    # FLANN matcher
    flann_trees: int = _FLANN_TREES
    flann_checks: int = _FLANN_CHECKS

    # Matching
    lowe_ratio: float = _LOWE_RATIO

    # RANSAC homography
    ransac_reproj_threshold: float = _RANSAC_REPROJ_THRESHOLD
    ransac_confidence: float = _RANSAC_CONFIDENCE
    ransac_max_iters: int = _RANSAC_MAX_ITERS

    # Acceptance criteria
    min_inliers: int = _MIN_INLIERS
    min_inlier_ratio: float = _MIN_INLIER_RATIO

    # Homography geometry checks
    max_condition_number: float = _MAX_CONDITION_NUMBER
    min_area_fraction: float = _MIN_AREA_FRACTION
    max_area_fraction: float = _MAX_AREA_FRACTION
    min_corner_angle_deg: float = _MIN_CORNER_ANGLE_DEG

    # Preprocessing
    use_clahe: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """Structured output of the banknote detection pipeline.

    On success (``success=True``) all optional fields are populated.
    On failure only ``success``, ``message``, and the keypoint counts
    (which may be zero) are guaranteed to be set.
    """

    # ── Status ──────────────────────────────────────────────────────────────
    success: bool
    message: str
    confidence: float  # 0.0 – 1.0

    # ── Input images ────────────────────────────────────────────────────────
    scene_img: np.ndarray       # float32 RGB [0, 1], H×W×3
    reference_img: np.ndarray   # float32 RGB [0, 1], H×W×3

    # ── Preprocessed grayscale (saved in step-image functions) ──────────────
    scene_gray: np.ndarray      # uint8 grayscale after CLAHE
    reference_gray: np.ndarray  # uint8 grayscale after CLAHE

    # ── Detection geometry ───────────────────────────────────────────────────
    homography: Optional[np.ndarray]     # 3×3 float64
    scene_corners: Optional[np.ndarray]  # (4, 2) float32 — banknote quad in scene
    warped_scene: Optional[np.ndarray]   # float32 RGB [0, 1] — scene in ref frame

    # ── Feature statistics ───────────────────────────────────────────────────
    keypoints_scene: list   # list[cv.KeyPoint]
    keypoints_ref: list     # list[cv.KeyPoint]
    n_keypoints_scene: int
    n_keypoints_ref: int

    n_matches_raw: int    # before Lowe ratio test
    n_matches_lowe: int   # after Lowe ratio test
    n_inliers: int        # RANSAC inliers
    inlier_ratio: float   # n_inliers / n_matches_lowe
    reproj_rmse: float    # pixels

    # ── Correspondence data (for visualisation) ──────────────────────────────
    src_pts: Optional[np.ndarray]    # (N, 1, 2) float32 — scene matched points
    dst_pts: Optional[np.ndarray]    # (N, 1, 2) float32 — reference matched points
    good_matches: list               # list[cv.DMatch] after Lowe filter
    inlier_mask: Optional[np.ndarray]  # (N,) uint8 — 1 = RANSAC inlier


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------


def _float_rgb_to_gray_uint8(img_float: np.ndarray) -> np.ndarray:
    """Convert a float32 RGB [0, 1] image to uint8 grayscale.

    Args:
        img_float: float32 array of shape (H, W, 3) in RGB order, values [0, 1].

    Returns:
        uint8 grayscale array of shape (H, W).
    """
    arr = np.clip(img_float, 0.0, 1.0)
    rgb_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    return cv.cvtColor(rgb_u8, cv.COLOR_RGB2GRAY)


def preprocess_for_sift(
    gray_uint8: np.ndarray,
    config: DetectionConfig,
) -> np.ndarray:
    """Enhance contrast with CLAHE to improve SIFT feature detection.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) boosts local
    contrast without amplifying noise. It is especially helpful when the
    banknote is photographed under uneven or low-key lighting.

    Args:
        gray_uint8: uint8 grayscale image.
        config: Detection configuration; CLAHE is skipped when
            ``config.use_clahe`` is False.

    Returns:
        uint8 grayscale image (CLAHE-enhanced or unchanged).
    """
    if not config.use_clahe:
        return gray_uint8

    clahe = cv.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    return clahe.apply(gray_uint8)


# ---------------------------------------------------------------------------
# SIFT feature detection
# ---------------------------------------------------------------------------


def _create_sift(config: DetectionConfig) -> cv.SIFT:
    """Instantiate a SIFT detector with parameters from *config*."""
    return cv.SIFT_create(
        nfeatures=config.sift_n_features,
        contrastThreshold=config.sift_contrast_threshold,
        edgeThreshold=config.sift_edge_threshold,
        sigma=config.sift_sigma,
    )


def detect_keypoints_and_descriptors(
    gray: np.ndarray,
    sift: cv.SIFT,
) -> tuple[list, Optional[np.ndarray]]:
    """Detect SIFT keypoints and compute descriptors on a grayscale image.

    Args:
        gray: uint8 grayscale image.
        sift: Pre-instantiated SIFT detector.

    Returns:
        Tuple ``(keypoints, descriptors)`` where *descriptors* is a float32
        array of shape (N, 128) or None if no features were found.
    """
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


# ---------------------------------------------------------------------------
# Feature matching
# ---------------------------------------------------------------------------


def _create_flann(config: DetectionConfig) -> cv.FlannBasedMatcher:
    """Create a FLANN matcher for L2 (SIFT) descriptors."""
    FLANN_INDEX_KDTREE = getattr(cv, "FLANN_INDEX_KDTREE", 1)
    index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": config.flann_trees}
    search_params = {"checks": config.flann_checks}
    return cv.FlannBasedMatcher(index_params, search_params)


def match_features_flann(
    des1: np.ndarray,
    des2: np.ndarray,
    config: DetectionConfig,
) -> list:
    """Match SIFT descriptors using FLANN KNN (k=2).

    Args:
        des1: Descriptors from the scene image, shape (N, 128) float32.
        des2: Descriptors from the reference image, shape (M, 128) float32.
        config: Detection configuration (FLANN parameters).

    Returns:
        List of pairs ``[m, n]`` (cv.DMatch) for each query descriptor.
        An empty list is returned when fewer than two reference descriptors
        exist.
    """
    if des1 is None or des2 is None or len(des2) < 2:
        return []

    flann = _create_flann(config)
    return flann.knnMatch(des1, des2, k=2)


def apply_lowe_ratio_test(
    knn_matches: list,
    ratio_threshold: float = _LOWE_RATIO,
) -> list:
    """Filter matches with Lowe's ratio test.

    A match is accepted only when the best candidate is significantly closer
    than the second-best, indicating a distinctive and reliable correspondence.

    Args:
        knn_matches: List of [m, n] pairs from knnMatch(k=2).
        ratio_threshold: Reject matches where ``m.distance / n.distance``
            exceeds this value. Default 0.7 (Lowe 2004).

    Returns:
        List of accepted cv.DMatch objects.
    """
    good: list = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good.append(m)
    return good


# ---------------------------------------------------------------------------
# Homography estimation
# ---------------------------------------------------------------------------


def estimate_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    config: DetectionConfig,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate a perspective homography using RANSAC.

    Args:
        src_pts: Matched scene keypoint coordinates, shape (N, 1, 2) float32.
        dst_pts: Matched reference keypoint coordinates, shape (N, 1, 2) float32.
        config: Detection configuration (RANSAC parameters).

    Returns:
        Tuple ``(H, inlier_mask)``.  Both are None when RANSAC fails or fewer
        than four correspondences are provided.
        *H* is a (3, 3) float64 matrix.
        *inlier_mask* is a (N, 1) uint8 array with 1 for inliers.
    """
    if src_pts is None or len(src_pts) < 4:
        return None, None

    H, mask = cv.findHomography(
        src_pts,
        dst_pts,
        method=cv.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold,
        confidence=config.ransac_confidence,
        maxIters=config.ransac_max_iters,
    )
    return H, mask


# ---------------------------------------------------------------------------
# Homography validation
# ---------------------------------------------------------------------------


def _condition_number(H: np.ndarray) -> float:
    """Compute the condition number of *H* via SVD (σ_max / σ_min).

    A large condition number indicates a near-singular (degenerate) matrix.
    """
    _, s, _ = np.linalg.svd(H)
    if s[-1] < 1e-12:
        return float("inf")
    return float(s[0] / s[-1])


def _quad_area(corners: np.ndarray) -> float:
    """Shoelace formula: signed area of a polygon given (N, 2) vertex array."""
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _min_interior_angle_deg(corners: np.ndarray) -> float:
    """Return the smallest interior angle of a quadrilateral in degrees.

    Args:
        corners: (4, 2) array of vertices in order (TL, TR, BR, BL or similar).
    """
    angles = []
    n = len(corners)
    for i in range(n):
        v_prev = corners[(i - 1) % n] - corners[i]
        v_next = corners[(i + 1) % n] - corners[i]
        cos_a = np.dot(v_prev, v_next) / (
            np.linalg.norm(v_prev) * np.linalg.norm(v_next) + 1e-12
        )
        angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    return float(min(angles))


def validate_homography(
    H: np.ndarray,
    n_inliers: int,
    n_matches_lowe: int,
    scene_shape: tuple[int, int],
    ref_shape: tuple[int, int],
    config: DetectionConfig,
) -> tuple[bool, str]:
    """Apply a battery of geometric checks to accept or reject *H*.

    Checks performed (in order):
    1. Condition number — rejects near-singular matrices.
    2. Inlier count and inlier ratio — rejects weak correspondences.
    3. Corner geometry — verifies that the projected quadrilateral is convex
       and has reasonable interior angles.
    4. Area fraction — rejects results that are too small or too large relative
       to the scene.

    Args:
        H: 3×3 homography.
        n_inliers: Number of RANSAC inliers.
        n_matches_lowe: Total Lowe-filtered matches (denominator for ratio).
        scene_shape: (H, W) of the scene image.
        ref_shape: (H, W) of the reference image.
        config: Detection configuration with acceptance thresholds.

    Returns:
        ``(is_valid, reason)`` — *reason* describes the rejection cause or
        "OK" on success.
    """
    # 1. Condition number
    cond = _condition_number(H)
    if not np.isfinite(cond) or cond > config.max_condition_number:
        return False, f"degenerate homography (condition number {cond:.0f})"

    # 2. Inlier count and ratio
    if n_inliers < config.min_inliers:
        return False, f"too few inliers ({n_inliers} < {config.min_inliers})"

    inlier_ratio = n_inliers / max(1, n_matches_lowe)
    if inlier_ratio < config.min_inlier_ratio:
        return (
            False,
            f"low inlier ratio ({inlier_ratio:.2f} < {config.min_inlier_ratio})",
        )

    # 3. Corner geometry — project reference corners into the scene
    corners = project_corners_to_scene(H, ref_shape)
    if corners is None:
        return False, "corner projection failed"

    min_angle = _min_interior_angle_deg(corners)
    if min_angle < config.min_corner_angle_deg:
        return (
            False,
            f"degenerate quadrilateral (min angle {min_angle:.1f}° < "
            f"{config.min_corner_angle_deg}°)",
        )

    # 4. Area fraction
    scene_area = scene_shape[0] * scene_shape[1]
    detected_area = _quad_area(corners)
    area_frac = detected_area / max(1, scene_area)

    if area_frac < config.min_area_fraction:
        return False, f"detected region too small (area fraction {area_frac:.4f})"

    if area_frac > config.max_area_fraction:
        return False, f"detected region too large (area fraction {area_frac:.4f})"

    return True, "OK"


# ---------------------------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------------------------


def compute_reprojection_rmse(
    H: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    inlier_mask: Optional[np.ndarray],
) -> float:
    """Compute the RMSE of reprojection error for RANSAC inliers.

    Args:
        H: 3×3 homography.
        src_pts: Source points, shape (N, 1, 2) float32.
        dst_pts: Destination points, shape (N, 1, 2) float32.
        inlier_mask: (N, 1) uint8 array; only inlier rows are used.

    Returns:
        RMSE in pixels, or ``inf`` when no inliers are available.
    """
    if H is None or src_pts is None or dst_pts is None:
        return float("inf")

    src = src_pts.reshape(-1, 2).astype(np.float64)
    dst = dst_pts.reshape(-1, 2).astype(np.float64)

    if inlier_mask is not None:
        mask = inlier_mask.ravel().astype(bool)
        src, dst = src[mask], dst[mask]

    if len(src) == 0:
        return float("inf")

    # Project src through H
    src_h = np.column_stack([src, np.ones(len(src))])
    proj_h = (H @ src_h.T).T
    proj = proj_h[:, :2] / (proj_h[:, 2:3] + 1e-12)

    errors = np.linalg.norm(proj - dst, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def compute_reference_corners(ref_shape: tuple[int, int]) -> np.ndarray:
    """Return the four corners of the reference image as a (4, 2) float32 array.

    Corners are ordered: top-left, top-right, bottom-right, bottom-left.

    Args:
        ref_shape: (H, W) of the reference image.
    """
    h, w = ref_shape
    return np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=np.float32,
    )


def project_corners_to_scene(
    H: np.ndarray,
    ref_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Map the reference image corners through *H⁻¹* to get the banknote
    quadrilateral in scene coordinates.

    The inverse of H (scene → reference) gives reference → scene, which is
    what we need to draw the detected banknote boundary.

    Args:
        H: 3×3 homography that maps scene to reference.
        ref_shape: (H, W) of the reference image.

    Returns:
        (4, 2) float32 array of banknote corners in the scene image, or None
        if the inverse cannot be computed.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    ref_corners = compute_reference_corners(ref_shape).reshape(-1, 1, 2)
    scene_corners = cv.perspectiveTransform(ref_corners, H_inv)
    if scene_corners is None:
        return None
    return scene_corners.reshape(4, 2).astype(np.float32)


def warp_scene_to_reference(
    scene_img: np.ndarray,
    H: np.ndarray,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    """Warp *scene_img* into the reference image frame using homography *H*.

    Args:
        scene_img: float32 RGB [0, 1] scene image.
        H: 3×3 homography (scene → reference).
        ref_shape: (H, W) output dimensions.

    Returns:
        Warped float32 RGB image of shape (ref_H, ref_W, 3).
    """
    h, w = ref_shape
    warped = cv.warpPerspective(scene_img, H, (w, h), flags=cv.INTER_LINEAR)
    return np.clip(warped, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------


def compute_confidence(
    n_inliers: int,
    n_matches_lowe: int,
    reproj_rmse: float,
    ransac_thresh: float,
) -> float:
    """Compute a scalar detection confidence in [0, 1].

    Combines inlier ratio and reprojection accuracy into a single score.
    The formula is heuristic and designed so that a typical good detection
    (>50 % inliers, sub-pixel reprojection error) scores above 0.8.

    Args:
        n_inliers: Number of RANSAC inliers.
        n_matches_lowe: Total matches after Lowe ratio test.
        reproj_rmse: Reprojection RMSE in pixels.
        ransac_thresh: RANSAC reprojection threshold (used for normalisation).

    Returns:
        Float in [0, 1].
    """
    if n_matches_lowe == 0:
        return 0.0

    inlier_score = n_inliers / max(1, n_matches_lowe)  # [0, 1]

    # Normalise RMSE: 0 px → 1.0, ransac_thresh px → 0.0, beyond → clipped 0.
    if not np.isfinite(reproj_rmse):
        reproj_score = 0.0
    else:
        reproj_score = max(0.0, 1.0 - reproj_rmse / ransac_thresh)

    # Weighted combination
    confidence = 0.6 * inlier_score + 0.4 * reproj_score
    return float(np.clip(confidence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------


def detect_banknote(
    scene_img: np.ndarray,
    reference_img: np.ndarray,
    config: Optional[DetectionConfig] = None,
) -> DetectionResult:
    """Detect a banknote in *scene_img* by matching it against *reference_img*.

    Both images must be float32 RGB in [0, 1], as produced by
    ``utils.load_image_rgb``.

    Args:
        scene_img: Photo containing the banknote to be detected.
        reference_img: Clean reference photograph of the banknote.
        config: Algorithm parameters; uses production defaults when None.

    Returns:
        :class:`DetectionResult` with all fields populated. Check
        ``result.success`` before using ``result.homography`` or
        ``result.warped_scene``.
    """
    if config is None:
        config = DetectionConfig()

    # ── 1. Prepare grayscale inputs ──────────────────────────────────────────
    scene_gray_raw = _float_rgb_to_gray_uint8(scene_img)
    ref_gray_raw = _float_rgb_to_gray_uint8(reference_img)

    scene_gray = preprocess_for_sift(scene_gray_raw, config)
    ref_gray = preprocess_for_sift(ref_gray_raw, config)

    # ── 2. SIFT keypoints and descriptors ────────────────────────────────────
    sift = _create_sift(config)
    kp_scene, des_scene = detect_keypoints_and_descriptors(scene_gray, sift)
    kp_ref, des_ref = detect_keypoints_and_descriptors(ref_gray, sift)

    logger.debug(
        "Keypoints: scene=%d, reference=%d", len(kp_scene), len(kp_ref)
    )

    # Common failure result factory — keeps field initialisation in one place.
    def _fail(message: str) -> DetectionResult:
        return DetectionResult(
            success=False,
            message=message,
            confidence=0.0,
            scene_img=scene_img,
            reference_img=reference_img,
            scene_gray=scene_gray,
            reference_gray=ref_gray,
            homography=None,
            scene_corners=None,
            warped_scene=None,
            keypoints_scene=kp_scene,
            keypoints_ref=kp_ref,
            n_keypoints_scene=len(kp_scene),
            n_keypoints_ref=len(kp_ref),
            n_matches_raw=0,
            n_matches_lowe=0,
            n_inliers=0,
            inlier_ratio=0.0,
            reproj_rmse=float("inf"),
            src_pts=None,
            dst_pts=None,
            good_matches=[],
            inlier_mask=None,
        )

    if des_scene is None or des_ref is None:
        return _fail("no descriptors found in scene or reference image")

    # ── 3. FLANN matching + Lowe ratio test ──────────────────────────────────
    knn_matches = match_features_flann(des_scene, des_ref, config)
    n_matches_raw = len(knn_matches)

    good_matches = apply_lowe_ratio_test(knn_matches, config.lowe_ratio)
    n_matches_lowe = len(good_matches)

    logger.debug(
        "Matches: raw=%d, after Lowe=%.0f",
        n_matches_raw,
        n_matches_lowe,
    )

    if n_matches_lowe < 4:
        return _fail(
            f"too few matches after Lowe ratio test ({n_matches_lowe}); "
            "minimum 4 required for homography"
        )

    src_pts = np.float32(
        [kp_scene[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_ref[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # ── 4. RANSAC homography ─────────────────────────────────────────────────
    H, inlier_mask = estimate_homography(src_pts, dst_pts, config)

    if H is None:
        return _fail("RANSAC homography estimation failed")

    inlier_mask_flat = inlier_mask.ravel().astype(np.uint8)
    n_inliers = int(inlier_mask_flat.sum())
    inlier_ratio = n_inliers / max(1, n_matches_lowe)

    reproj_rmse = compute_reprojection_rmse(H, src_pts, dst_pts, inlier_mask)

    logger.debug(
        "RANSAC: inliers=%d / %d (%.1f %%), reproj RMSE=%.2f px",
        n_inliers,
        n_matches_lowe,
        100.0 * inlier_ratio,
        reproj_rmse,
    )

    # ── 5. Homography validation ─────────────────────────────────────────────
    ref_shape = reference_img.shape[:2]
    scene_shape = scene_img.shape[:2]

    is_valid, reason = validate_homography(
        H, n_inliers, n_matches_lowe, scene_shape, ref_shape, config
    )

    if not is_valid:
        return DetectionResult(
            success=False,
            message=f"homography rejected: {reason}",
            confidence=0.0,
            scene_img=scene_img,
            reference_img=reference_img,
            scene_gray=scene_gray,
            reference_gray=ref_gray,
            homography=H,
            scene_corners=None,
            warped_scene=None,
            keypoints_scene=kp_scene,
            keypoints_ref=kp_ref,
            n_keypoints_scene=len(kp_scene),
            n_keypoints_ref=len(kp_ref),
            n_matches_raw=n_matches_raw,
            n_matches_lowe=n_matches_lowe,
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio,
            reproj_rmse=reproj_rmse,
            src_pts=src_pts,
            dst_pts=dst_pts,
            good_matches=good_matches,
            inlier_mask=inlier_mask_flat,
        )

    # ── 6. Warp and extract corners ──────────────────────────────────────────
    warped_scene = warp_scene_to_reference(scene_img, H, ref_shape)
    scene_corners = project_corners_to_scene(H, ref_shape)
    confidence = compute_confidence(
        n_inliers, n_matches_lowe, reproj_rmse, config.ransac_reproj_threshold
    )

    logger.info(
        "Detection successful — inliers=%d, reproj RMSE=%.2f px, confidence=%.2f",
        n_inliers,
        reproj_rmse,
        confidence,
    )

    return DetectionResult(
        success=True,
        message="OK",
        confidence=confidence,
        scene_img=scene_img,
        reference_img=reference_img,
        scene_gray=scene_gray,
        reference_gray=ref_gray,
        homography=H,
        scene_corners=scene_corners,
        warped_scene=warped_scene,
        keypoints_scene=kp_scene,
        keypoints_ref=kp_ref,
        n_keypoints_scene=len(kp_scene),
        n_keypoints_ref=len(kp_ref),
        n_matches_raw=n_matches_raw,
        n_matches_lowe=n_matches_lowe,
        n_inliers=n_inliers,
        inlier_ratio=inlier_ratio,
        reproj_rmse=reproj_rmse,
        src_pts=src_pts,
        dst_pts=dst_pts,
        good_matches=good_matches,
        inlier_mask=inlier_mask_flat,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers — internal
# ---------------------------------------------------------------------------


def _float_to_bgr_u8(img_float: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0, 1] to uint8 BGR for OpenCV drawing functions."""
    rgb_u8 = (np.clip(img_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv.cvtColor(rgb_u8, cv.COLOR_RGB2BGR)


def _save_bgr(path: Path, bgr: np.ndarray) -> None:
    """Save a BGR uint8 image to *path* (creates parent directories)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), bgr)


# ---------------------------------------------------------------------------
# Step-image save functions
# ---------------------------------------------------------------------------


def save_preprocessed_images(result: DetectionResult, output_dir: Path) -> None:
    """Save CLAHE-enhanced grayscale images for both scene and reference.

    Output files:
        ``01_scene_preprocessed.png``
        ``02_reference_preprocessed.png``
    """
    _save_bgr(output_dir / "01_scene_preprocessed.png", result.scene_gray)
    _save_bgr(output_dir / "02_reference_preprocessed.png", result.reference_gray)
    logger.debug("Saved preprocessed grayscale images to %s", output_dir)


def save_keypoints_images(result: DetectionResult, output_dir: Path) -> None:
    """Draw and save SIFT keypoints (with scale and orientation) on both images.

    Output files:
        ``03_scene_keypoints.png``
        ``04_reference_keypoints.png``
    """
    flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    kp_scene_img = cv.drawKeypoints(
        result.scene_gray,
        result.keypoints_scene,
        None,
        color=(0, 120, 255),
        flags=flags,
    )
    kp_ref_img = cv.drawKeypoints(
        result.reference_gray,
        result.keypoints_ref,
        None,
        color=(0, 120, 255),
        flags=flags,
    )

    _save_bgr(output_dir / "03_scene_keypoints.png", kp_scene_img)
    _save_bgr(output_dir / "04_reference_keypoints.png", kp_ref_img)
    logger.debug("Saved keypoint images to %s", output_dir)


def save_all_matches_image(result: DetectionResult, output_dir: Path) -> None:
    """Save all Lowe-filtered matches as a side-by-side comparison image.

    Every accepted match is drawn in gray. Useful for assessing descriptor
    quality before RANSAC.

    Output file: ``05_matches_all.png``
    """
    if not result.good_matches:
        logger.debug("No matches to draw in save_all_matches_image")
        return

    img = cv.drawMatches(
        result.scene_gray,
        result.keypoints_scene,
        result.reference_gray,
        result.keypoints_ref,
        result.good_matches,
        None,
        matchColor=(200, 200, 200),
        singlePointColor=(80, 80, 80),
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    _save_bgr(output_dir / "05_matches_all.png", img)
    logger.debug("Saved all-matches image to %s", output_dir)


def save_inlier_matches_image(result: DetectionResult, output_dir: Path) -> None:
    """Save RANSAC inlier matches (green) and outlier matches (red).

    Requires ``result.inlier_mask`` to be set. Skipped silently when no
    matches are available.

    Output file: ``06_matches_inliers_outliers.png``
    """
    if not result.good_matches or result.inlier_mask is None:
        logger.debug("Skipping inlier match image: no matches or mask")
        return

    inlier_mask_list = result.inlier_mask.tolist()

    img = cv.drawMatches(
        result.scene_gray,
        result.keypoints_scene,
        result.reference_gray,
        result.keypoints_ref,
        result.good_matches,
        None,
        matchColor=(0, 255, 0),       # inliers: green
        singlePointColor=(80, 80, 80),
        matchesMask=inlier_mask_list,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Overlay outliers in red
    outlier_mask = (1 - result.inlier_mask).tolist()
    img = cv.drawMatches(
        result.scene_gray,
        result.keypoints_scene,
        result.reference_gray,
        result.keypoints_ref,
        result.good_matches,
        img,
        matchColor=(0, 0, 255),       # outliers: red
        singlePointColor=None,
        matchesMask=outlier_mask,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv.DrawMatchesFlags_DRAW_OVER_OUTIMG,
    )

    _save_bgr(output_dir / "06_matches_inliers_outliers.png", img)
    logger.debug("Saved inlier/outlier match image to %s", output_dir)


def save_scene_detection_image(result: DetectionResult, output_dir: Path) -> None:
    """Draw the detected banknote quadrilateral on the original scene image.

    The polygon is drawn in bright green when detection succeeded. When the
    result carries corners but detection failed (rare), a yellow polygon is
    drawn to indicate a rejected candidate.

    Output file: ``07_scene_detection.png``
    """
    scene_bgr = _float_to_bgr_u8(result.scene_img)

    if result.scene_corners is not None:
        pts = result.scene_corners.astype(np.int32).reshape(-1, 1, 2)
        color = (0, 255, 0) if result.success else (0, 200, 255)
        cv.polylines(scene_bgr, [pts], isClosed=True, color=color, thickness=3)

        # Annotate corners with index numbers
        for idx, (x, y) in enumerate(result.scene_corners.astype(int)):
            cv.circle(scene_bgr, (x, y), 6, color, -1)
            cv.putText(
                scene_bgr,
                str(idx + 1),
                (x + 8, y - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv.LINE_AA,
            )

    # Overlay statistics
    status_text = (
        f"Inliers: {result.n_inliers} / {result.n_matches_lowe}  "
        f"RMSE: {result.reproj_rmse:.1f} px  "
        f"Conf: {result.confidence:.2f}"
    )
    cv.putText(
        scene_bgr,
        status_text,
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    _save_bgr(output_dir / "07_scene_detection.png", scene_bgr)
    logger.debug("Saved scene detection image to %s", output_dir)


def save_warped_banknote_image(result: DetectionResult, output_dir: Path) -> None:
    """Save the scene image warped to the reference frame.

    Output file: ``08_warped_banknote.png``
    """
    if result.warped_scene is None:
        logger.debug("Skipping warped banknote image: detection was unsuccessful")
        return

    _save_bgr(output_dir / "08_warped_banknote.png", _float_to_bgr_u8(result.warped_scene))
    logger.debug("Saved warped banknote image to %s", output_dir)


def save_overlay_image(result: DetectionResult, output_dir: Path) -> None:
    """Save a semi-transparent overlay of the warped scene on the reference.

    Useful for visually checking alignment quality. The warped scene is blended
    at 50 % opacity onto the reference image.

    Output file: ``09_overlay_on_reference.png``
    """
    if result.warped_scene is None:
        logger.debug("Skipping overlay image: detection was unsuccessful")
        return

    ref_bgr = _float_to_bgr_u8(result.reference_img)
    warped_bgr = _float_to_bgr_u8(result.warped_scene)

    overlay = cv.addWeighted(ref_bgr, 0.5, warped_bgr, 0.5, 0)
    _save_bgr(output_dir / "09_overlay_on_reference.png", overlay)
    logger.debug("Saved overlay image to %s", output_dir)


# ---------------------------------------------------------------------------
# Master save function
# ---------------------------------------------------------------------------


def save_detection_steps(result: DetectionResult, output_dir: Path) -> None:
    """Save one image per detection pipeline step to *output_dir*.

    Saved files (numbered for easy ordering):
        01_scene_preprocessed.png
        02_reference_preprocessed.png
        03_scene_keypoints.png
        04_reference_keypoints.png
        05_matches_all.png
        06_matches_inliers_outliers.png
        07_scene_detection.png
        08_warped_banknote.png       (only when detection succeeded)
        09_overlay_on_reference.png  (only when detection succeeded)

    Args:
        result: Output from :func:`detect_banknote`.
        output_dir: Directory where step images are written.
            Created automatically if it does not exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_preprocessed_images(result, output_dir)
    save_keypoints_images(result, output_dir)
    save_all_matches_image(result, output_dir)
    save_inlier_matches_image(result, output_dir)
    save_scene_detection_image(result, output_dir)
    save_warped_banknote_image(result, output_dir)
    save_overlay_image(result, output_dir)

    logger.info(
        "Detection step images saved to %s  (success=%s, confidence=%.2f)",
        output_dir,
        result.success,
        result.confidence,
    )
