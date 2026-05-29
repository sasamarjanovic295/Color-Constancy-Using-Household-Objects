"""
hand_segmentation.py — Hand detection and segmentation using MediaPipe Hands.

Pipeline:
    1. Run MediaPipe Hand Landmarker on the input RGB image.
    2. If no hand detected → return failure result.
    3. Extract 21 (x, y) landmark coordinates.
    4. Compute convex hull from all 21 landmarks → polygon mask.
    5. Erode mask to exclude fingertips, nail regions and edge artifacts.
    6. Return binary mask (uint8, 0/255) for skin colour measurement.

Usage::

    from pathlib import Path
    from src.hand_segmentation import detect_and_segment

    image = load_image_rgb(Path("data/train/IMG_hand.jpeg"))
    result = detect_and_segment(image, save=True, log=True,
                                output_dir=Path("output/debug"))
    if result.detected:
        skin_mask = result.mask  # uint8, shape (H, W), values 0 or 255
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_LANDMARKS: int = 21
_EROSION_KERNEL_SIZE: int = 15
_MIN_HAND_AREA_RATIO: float = 0.005

# Landmark indices for the five fingertips
_FINGERTIP_INDICES: list[int] = [4, 8, 12, 16, 20]

# Output filenames when save=True
_FILENAME_VIZ_LANDMARKS: str = "hand_viz_01_landmarks.jpg"
_FILENAME_VIZ_HULL: str = "hand_viz_02_convex_hull.jpg"
_FILENAME_VIZ_MASK_RAW: str = "hand_viz_03_mask_raw.jpg"
_FILENAME_VIZ_MASK_ERODED: str = "hand_viz_04_mask_eroded.jpg"
_FILENAME_VIZ_OVERLAY: str = "hand_viz_05_overlay.jpg"

# MediaPipe model — resolve path relative to project root
_MODEL_PATH: Path = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

_MP_MAX_NUM_HANDS: int = 2
_MP_MIN_DETECTION_CONFIDENCE: float = 0.5
_MP_MIN_HAND_PRESENCE_CONFIDENCE: float = 0.5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HandSegmentationResult:
    """Structured output of the hand detection and segmentation pipeline.

    On success (``detected=True``) the mask and landmark fields are populated.
    On failure only ``detected`` and ``failure_reason`` are set.
    """

    detected: bool
    failure_reason: str | None
    n_hands: int | None
    landmarks: np.ndarray | None  # shape (21, 2), float32, pixel coords
    mask: np.ndarray | None  # shape (H, W), uint8, values 0 or 255
    mask_eroded: np.ndarray | None  # shape (H, W), uint8, eroded version
    hand_area_px: int | None  # number of non-zero pixels in mask
    handedness: str | None  # "Left" or "Right" (from MediaPipe)
    confidence: float | None  # hand detection confidence


# ---------------------------------------------------------------------------
# Private helpers — drawing / saving
# ---------------------------------------------------------------------------


def _float_to_bgr_u8(img_float: np.ndarray) -> np.ndarray:
    rgb_u8 = (np.clip(img_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv.cvtColor(rgb_u8, cv.COLOR_RGB2BGR)


def _save_bgr(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), bgr)


def _draw_landmarks_overlay(
    image_bgr: np.ndarray,
    landmarks_px: np.ndarray,
) -> np.ndarray:
    vis = image_bgr.copy()
    pts = landmarks_px.astype(np.int32)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    for i, j in connections:
        cv.line(vis, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 2)

    for idx, pt in enumerate(pts):
        color = (0, 0, 255) if idx in _FINGERTIP_INDICES else (255, 0, 0)
        radius = 6 if idx in _FINGERTIP_INDICES else 4
        cv.circle(vis, tuple(pt), radius, color, -1)
        cv.putText(vis, str(idx), (pt[0] + 5, pt[1] - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return vis


def _draw_hull_overlay(
    image_bgr: np.ndarray,
    landmarks_px: np.ndarray,
    hull: np.ndarray,
) -> np.ndarray:
    vis = image_bgr.copy()
    overlay = vis.copy()
    cv.fillConvexPoly(overlay, hull, (0, 200, 0))
    cv.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
    cv.drawContours(vis, [hull], -1, (0, 255, 0), 2)

    pts = landmarks_px.astype(np.int32)
    for pt in pts:
        cv.circle(vis, tuple(pt), 3, (0, 0, 255), -1)
    return vis


def _draw_mask_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 200, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    vis = image_bgr.copy()
    overlay = vis.copy()
    overlay[mask > 0] = color
    cv.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(vis, contours, -1, color, 2)
    return vis


# ---------------------------------------------------------------------------
# Private helpers — mask construction
# ---------------------------------------------------------------------------


def _landmarks_to_pixel_coords(
    hand_landmarks: list,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    coords = np.array(
        [(lm.x * image_w, lm.y * image_h) for lm in hand_landmarks],
        dtype=np.float32,
    )
    return coords


def _build_convex_hull_mask(
    landmarks_px: np.ndarray,
    image_h: int,
    image_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    pts_int = landmarks_px.astype(np.int32)
    hull = cv.convexHull(pts_int)
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    cv.fillConvexPoly(mask, hull, 255)
    return mask, hull


def _erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv.erode(mask, kernel, iterations=1)


def _select_best_hand(
    result,
    image_h: int,
    image_w: int,
    log: bool = False,
) -> tuple[int, np.ndarray, str, float]:
    """Select the hand with the largest convex hull area from detected hands."""
    best_idx = 0
    best_area = 0
    best_landmarks = None
    best_handedness = "Unknown"
    best_confidence = 0.0

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        lm_px = _landmarks_to_pixel_coords(hand_landmarks, image_h, image_w)
        hull = cv.convexHull(lm_px.astype(np.int32))
        area = cv.contourArea(hull)

        handedness_label = "Unknown"
        hand_confidence = 0.0
        if result.handedness and i < len(result.handedness):
            cat = result.handedness[i][0]
            handedness_label = cat.category_name
            hand_confidence = cat.score

        if log:
            logger.info(
                "  Hand %d: %s (conf=%.3f), hull area=%d px",
                i, handedness_label, hand_confidence, int(area),
            )

        if area > best_area:
            best_idx = i
            best_area = area
            best_landmarks = lm_px
            best_handedness = handedness_label
            best_confidence = hand_confidence

    return best_idx, best_landmarks, best_handedness, best_confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_and_segment(
    image: np.ndarray,
    view: bool = False,
    save: bool = False,
    log: bool = False,
    output_dir: Path | None = None,
    erosion_kernel: int = _EROSION_KERNEL_SIZE,
) -> HandSegmentationResult:
    """Detect a hand in *image* and return a binary segmentation mask.

    Args:
        image: float32 RGB [0, 1], shape (H, W, 3).  Same convention as
            ``colorchecker.detect_and_measure``.
        view: If True, open non-blocking OpenCV windows with debug overlays.
        save: If True, save debug visualisation images to ``output_dir``.
        log: If True, emit structured progress messages via logger.info.
        output_dir: Directory for saved debug images.
        erosion_kernel: Size of the elliptical erosion kernel applied to
            the convex hull mask to exclude edge artifacts (default 15px).

    Returns:
        :class:`HandSegmentationResult`.  Never returns ``None``.
    """

    def _fail(reason: str) -> HandSegmentationResult:
        if log:
            logger.info("FAILURE: %s", reason)
        return HandSegmentationResult(
            detected=False,
            failure_reason=f"hand_segmentation: {reason}",
            n_hands=None,
            landmarks=None,
            mask=None,
            mask_eroded=None,
            hand_area_px=None,
            handedness=None,
            confidence=None,
        )

    image_h, image_w = image.shape[:2]

    # ── Step 1: Convert to uint8 for MediaPipe ──────────────────────────────
    if log:
        logger.info("=== HAND SEGMENTATION ===")
        logger.info("Image size: %d x %d px", image_w, image_h)

    image_u8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # ── Step 2: Run MediaPipe Hand Landmarker (Tasks API) ───────────────────
    if log:
        logger.info("Step 1: Running MediaPipe HandLandmarker")

    if not _MODEL_PATH.exists():
        return _fail(f"model file not found: {_MODEL_PATH}")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        num_hands=_MP_MAX_NUM_HANDS,
        min_hand_detection_confidence=_MP_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=_MP_MIN_HAND_PRESENCE_CONFIDENCE,
    )
    detector = HandLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_u8)
    result = detector.detect(mp_image)
    detector.close()

    # ── Step 3: Check detection ─────────────────────────────────────────────
    if not result.hand_landmarks:
        return _fail("mediapipe returned 0 landmarks")

    n_hands = len(result.hand_landmarks)
    if log:
        logger.info("Step 2: Detected %d hand(s)", n_hands)

    # ── Step 4: Select best hand (largest hull area) ────────────────────────
    best_idx, landmarks_px, handedness, confidence = _select_best_hand(
        result, image_h, image_w, log=log,
    )

    if landmarks_px is None:
        return _fail("landmark extraction failed")

    if log:
        logger.info(
            "Step 3: Selected hand %d — %s (confidence=%.3f)",
            best_idx, handedness, confidence,
        )

    # ── Step 5: Build convex hull mask ──────────────────────────────────────
    mask_raw, hull = _build_convex_hull_mask(landmarks_px, image_h, image_w)
    raw_area = int(np.count_nonzero(mask_raw))
    total_pixels = image_h * image_w
    area_ratio = raw_area / total_pixels

    if log:
        logger.info(
            "Step 4: Convex hull mask — %d px (%.2f%% of image)",
            raw_area, area_ratio * 100,
        )

    if area_ratio < _MIN_HAND_AREA_RATIO:
        return _fail(
            f"hand area too small ({area_ratio:.4f} < {_MIN_HAND_AREA_RATIO})"
        )

    # ── Step 6: Erode mask ──────────────────────────────────────────────────
    mask_eroded = _erode_mask(mask_raw, erosion_kernel)
    eroded_area = int(np.count_nonzero(mask_eroded))

    if eroded_area == 0:
        if log:
            logger.warning(
                "Erosion removed all pixels (kernel=%d) — using raw mask",
                erosion_kernel,
            )
        mask_eroded = mask_raw.copy()
        eroded_area = raw_area

    if log:
        logger.info(
            "Step 5: Eroded mask (kernel=%d) — %d px (%.1f%% of raw hull)",
            erosion_kernel, eroded_area, eroded_area / max(raw_area, 1) * 100,
        )

    # ── Step 7: Visualizations ──────────────────────────────────────────────
    image_bgr = _float_to_bgr_u8(image)

    viz_landmarks = _draw_landmarks_overlay(image_bgr, landmarks_px)
    viz_hull = _draw_hull_overlay(image_bgr, landmarks_px, hull)
    viz_mask_raw = cv.cvtColor(mask_raw, cv.COLOR_GRAY2BGR)
    viz_mask_eroded = cv.cvtColor(mask_eroded, cv.COLOR_GRAY2BGR)
    viz_overlay = _draw_mask_overlay(image_bgr, mask_eroded)

    # ── Step 8: view / save ─────────────────────────────────────────────────
    if view:
        cv.imshow("Hand — landmarks", viz_landmarks)
        cv.imshow("Hand — convex hull", viz_hull)
        cv.imshow("Hand — mask (raw)", viz_mask_raw)
        cv.imshow("Hand — mask (eroded)", viz_mask_eroded)
        cv.imshow("Hand — overlay", viz_overlay)
        cv.waitKey(1)

    if save:
        if output_dir is None:
            logger.warning("save=True but output_dir is None — skipping")
        else:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            _save_bgr(out / _FILENAME_VIZ_LANDMARKS, viz_landmarks)
            _save_bgr(out / _FILENAME_VIZ_HULL, viz_hull)
            _save_bgr(out / _FILENAME_VIZ_MASK_RAW, viz_mask_raw)
            _save_bgr(out / _FILENAME_VIZ_MASK_ERODED, viz_mask_eroded)
            _save_bgr(out / _FILENAME_VIZ_OVERLAY, viz_overlay)

            if log:
                logger.info(
                    "Saved 5 visualisations to %s", out,
                )

    # ── Step 9: Return success ──────────────────────────────────────────────
    if log:
        logger.info("=== HAND SEGMENTATION OK ===")

    return HandSegmentationResult(
        detected=True,
        failure_reason=None,
        n_hands=n_hands,
        landmarks=landmarks_px,
        mask=mask_raw,
        mask_eroded=mask_eroded,
        hand_area_px=eroded_area,
        handedness=handedness,
        confidence=confidence,
    )
