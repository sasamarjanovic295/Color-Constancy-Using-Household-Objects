"""
hand_segmentation.py — Hand detection and segmentation.

Implements the Sánchez-Brizuela et al. (2023) skeleton-based pipeline:
    1. MediaPipe Hands → 21 landmarks
    2. Skeleton lines → M1
    3. Dilate M1 → M2 (search boundary)
    4. Sample skin colour from skeleton in CIELab
    5. Colour-range segmentation within M2 → Mab
    6. Morphological cleanup → M3
    7. Output = M3 ∩ M2

Reference:
    Sánchez-Brizuela, G., Cisnal, A., et al. (2023).
    "Lightweight real-time hand segmentation leveraging MediaPipe
    landmark detection." Virtual Reality, 27, 3125–3132.
    DOI: 10.1007/s10055-023-00858-0

Usage::

    from src.hand_segmentation import detect_and_segment

    result = detect_and_segment(image_rgb_f32, save=True, log=True,
                                output_dir=Path("debug_output"))
    if result.detected:
        skin_mask = result.mask   # uint8 (H, W), values 0 or 255
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
_FINGERTIP_INDICES: list[int] = [4, 8, 12, 16, 20]

_MODEL_PATH: Path = (
    Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
)
_MP_MAX_NUM_HANDS: int = 2
_MP_MIN_DETECTION_CONFIDENCE: float = 0.5
_MP_MIN_HAND_PRESENCE_CONFIDENCE: float = 0.5

_TARGET_WIDTH: int = 800

# Sánchez-Brizuela pipeline parameters
_SKELETON_THICKNESS: int = 5
_M2_DILATE_SIZE: int = 21
_N_SAMPLES: int = 150
_CLOSE_SIZE: int = 13
_OPEN_SIZE: int = 13
_FINAL_DILATE_SIZE: int = 7
_RANGE_MARGIN: float = 5.0  # CIELab units added to Q1/Q3

_MIN_HAND_AREA_RATIO: float = 0.005

# Forearm: extend skeleton from wrist in the direction opposite to middle finger
_FOREARM_EXTENSION: float = 1.5  # multiplier of wrist→middle_mcp distance
_FOREARM_THICKNESS: int = 9

_SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_VIZ_FILENAMES: list[str] = [
    "hand_viz_01_landmarks.jpg",
    "hand_viz_02_M1_skeleton.jpg",
    "hand_viz_03_M2_dilated.jpg",
    "hand_viz_04_skin_sample.jpg",
    "hand_viz_05_Mab.jpg",
    "hand_viz_06_M3.jpg",
    "hand_viz_07_output_mask.jpg",
    "hand_viz_08_overlay.jpg",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HandSegmentationResult:
    """Structured output of hand detection and segmentation.

    On success (``detected=True``) the mask and landmark fields are populated.
    On failure only ``detected`` and ``failure_reason`` carry values.
    """

    detected: bool
    failure_reason: str | None
    n_hands: int | None
    landmarks: np.ndarray | None  # (21, 2) float32, pixel coords at original res
    mask: np.ndarray | None  # (H, W) uint8, 0/255, original resolution
    hand_area_px: int | None
    handedness: str | None
    confidence: float | None


# ---------------------------------------------------------------------------
# Private helpers — drawing / IO
# ---------------------------------------------------------------------------


def _float_to_bgr_u8(img_float: np.ndarray) -> np.ndarray:
    rgb_u8 = (np.clip(img_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv.cvtColor(rgb_u8, cv.COLOR_RGB2BGR)


def _save_bgr(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), bgr)


def _draw_landmarks_viz(
    image_bgr: np.ndarray, landmarks_px: np.ndarray
) -> np.ndarray:
    vis = image_bgr.copy()
    pts = landmarks_px.astype(np.int32)
    for i, j in _SKELETON_CONNECTIONS:
        cv.line(vis, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 2)
    for idx, pt in enumerate(pts):
        color = (0, 0, 255) if idx in _FINGERTIP_INDICES else (255, 0, 0)
        radius = 6 if idx in _FINGERTIP_INDICES else 4
        cv.circle(vis, tuple(pt), radius, color, -1)
        cv.putText(
            vis, str(idx), (pt[0] + 5, pt[1] - 5),
            cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
        )
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
# Private helpers — pipeline stages
# ---------------------------------------------------------------------------


def _landmarks_to_pixel_coords(
    hand_landmarks: list, image_h: int, image_w: int
) -> np.ndarray:
    return np.array(
        [(lm.x * image_w, lm.y * image_h) for lm in hand_landmarks],
        dtype=np.float32,
    )


def _select_best_hand(
    result, image_h: int, image_w: int, log: bool = False
) -> tuple[int, np.ndarray | None, str, float]:
    best_idx, best_area = 0, 0
    best_landmarks = None
    best_handedness, best_confidence = "Unknown", 0.0

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        lm_px = _landmarks_to_pixel_coords(hand_landmarks, image_h, image_w)
        hull = cv.convexHull(lm_px.astype(np.int32))
        area = cv.contourArea(hull)

        label, conf = "Unknown", 0.0
        if result.handedness and i < len(result.handedness):
            cat = result.handedness[i][0]
            label, conf = cat.category_name, cat.score

        if log:
            logger.info(
                "  Hand %d: %s (conf=%.3f), area=%d px",
                i, label, conf, int(area),
            )
        if area > best_area:
            best_idx = i
            best_area = area
            best_landmarks = lm_px
            best_handedness = label
            best_confidence = conf

    return best_idx, best_landmarks, best_handedness, best_confidence


def _build_skeleton_mask(
    landmarks_px: np.ndarray, h: int, w: int
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = landmarks_px.astype(np.int32)
    for i, j in _SKELETON_CONNECTIONS:
        cv.line(mask, tuple(pts[i]), tuple(pts[j]), 255, _SKELETON_THICKNESS)

    # Forearm extension: wrist → direction opposite to middle_mcp
    wrist = landmarks_px[0]
    middle_mcp = landmarks_px[9]
    direction = wrist - middle_mcp
    length = np.linalg.norm(direction)
    if length > 0:
        forearm_end = wrist + (direction / length) * length * _FOREARM_EXTENSION
        forearm_end = np.clip(forearm_end, [0, 0], [w - 1, h - 1])
        cv.line(
            mask, tuple(pts[0]), tuple(forearm_end.astype(np.int32)),
            255, _FOREARM_THICKNESS,
        )

    return mask


def _sample_skin_lab(
    image_lab: np.ndarray, skeleton_mask: np.ndarray, n: int
) -> np.ndarray:
    ys, xs = np.where(skeleton_mask > 0)
    rng = np.random.default_rng(42)
    if len(ys) <= n:
        idx = np.arange(len(ys))
    else:
        idx = rng.choice(len(ys), size=n, replace=False)
    return image_lab[ys[idx], xs[idx]]


def _build_mab(
    image_lab: np.ndarray,
    search_mask: np.ndarray,
    a_range: tuple[float, float],
    b_range: tuple[float, float],
) -> np.ndarray:
    a = image_lab[:, :, 1]
    b = image_lab[:, :, 2]
    in_range = (
        (a >= a_range[0]) & (a <= a_range[1])
        & (b >= b_range[0]) & (b <= b_range[1])
    )
    mab = np.zeros(image_lab.shape[:2], dtype=np.uint8)
    mab[in_range & (search_mask > 0)] = 255
    return mab


def _morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    def _k(size: int) -> np.ndarray:
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))

    out = cv.morphologyEx(mask, cv.MORPH_CLOSE, _k(_CLOSE_SIZE))
    out = cv.morphologyEx(out, cv.MORPH_OPEN, _k(_OPEN_SIZE))
    out = cv.dilate(out, _k(_FINAL_DILATE_SIZE), iterations=1)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_and_segment(
    image: np.ndarray,
    view: bool = False,
    save: bool = False,
    log: bool = False,
    output_dir: Path | None = None,
) -> HandSegmentationResult:
    """Detect a hand and return a binary segmentation mask.

    Uses the Sánchez-Brizuela et al. (2023) pipeline: MediaPipe landmarks
    → skeleton → CIELab skin-colour model → morphological refinement.

    Args:
        image: float32 RGB [0, 1], shape (H, W, 3).
        view: Open non-blocking OpenCV windows with debug overlays.
        save: Save debug visualisation images to *output_dir*.
        log: Emit structured progress messages via ``logger.info``.
        output_dir: Directory for saved debug images.

    Returns:
        :class:`HandSegmentationResult`.  Never ``None``.
    """

    def _fail(reason: str) -> HandSegmentationResult:
        if log:
            logger.info("FAILURE: %s", reason)
        return HandSegmentationResult(
            detected=False,
            failure_reason=f"hand_segmentation: {reason}",
            n_hands=None, landmarks=None, mask=None,
            hand_area_px=None, handedness=None, confidence=None,
        )

    image_h, image_w = image.shape[:2]

    if log:
        logger.info("=== HAND SEGMENTATION (Sánchez-Brizuela pipeline) ===")
        logger.info("Image: %d × %d px", image_w, image_h)

    # ── 1. MediaPipe detection ─────────────────────────────────────────────
    if log:
        logger.info("Step 1: MediaPipe HandLandmarker")

    if not _MODEL_PATH.exists():
        return _fail(f"model file not found: {_MODEL_PATH}")

    image_u8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        num_hands=_MP_MAX_NUM_HANDS,
        min_hand_detection_confidence=_MP_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=_MP_MIN_HAND_PRESENCE_CONFIDENCE,
    )
    detector = HandLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_u8)
    mp_result = detector.detect(mp_image)
    detector.close()

    if not mp_result.hand_landmarks:
        return _fail("mediapipe returned 0 landmarks")

    n_hands = len(mp_result.hand_landmarks)
    if log:
        logger.info("  Detected %d hand(s)", n_hands)

    # ── 2. Select best hand ────────────────────────────────────────────────
    best_idx, landmarks_px, handedness, confidence = _select_best_hand(
        mp_result, image_h, image_w, log=log,
    )
    if landmarks_px is None:
        return _fail("landmark extraction failed")

    if log:
        logger.info(
            "Step 2: Selected hand %d — %s (conf=%.3f)",
            best_idx, handedness, confidence,
        )

    # ── 3. Downscale for processing ────────────────────────────────────────
    if image_w > _TARGET_WIDTH:
        scale = _TARGET_WIDTH / image_w
        proc_w = _TARGET_WIDTH
        proc_h = int(image_h * scale)
    else:
        scale = 1.0
        proc_w, proc_h = image_w, image_h

    image_proc = cv.resize(image, (proc_w, proc_h), interpolation=cv.INTER_AREA)
    lm_proc = landmarks_px * scale

    if log:
        logger.info(
            "Step 3: Processing at %d × %d (scale=%.3f)", proc_w, proc_h, scale,
        )

    # Convert processing image to CIELab (float32 → L [0,100], a/b [-127,127])
    bgr_f32 = cv.cvtColor(image_proc, cv.COLOR_RGB2BGR)
    image_lab = cv.cvtColor(bgr_f32, cv.COLOR_BGR2Lab)

    # Also need a uint8 BGR for viz
    bgr_proc_u8 = _float_to_bgr_u8(image_proc)

    # ── 4. M1 — skeleton mask ──────────────────────────────────────────────
    m1 = _build_skeleton_mask(lm_proc, proc_h, proc_w)
    if log:
        logger.info("Step 4: M1 skeleton — %d px", np.count_nonzero(m1))

    # ── 5. M2 — dilated skeleton (search boundary) ─────────────────────────
    kernel_m2 = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (_M2_DILATE_SIZE, _M2_DILATE_SIZE),
    )
    m2 = cv.dilate(m1, kernel_m2, iterations=1)
    if log:
        logger.info("Step 5: M2 dilated — %d px", np.count_nonzero(m2))

    # ── 6. Sample skin colour from skeleton ────────────────────────────────
    sampled = _sample_skin_lab(image_lab, m1, _N_SAMPLES)
    a_vals = sampled[:, 1]
    b_vals = sampled[:, 2]

    a_q1, a_q3 = float(np.percentile(a_vals, 25)), float(np.percentile(a_vals, 75))
    b_q1, b_q3 = float(np.percentile(b_vals, 25)), float(np.percentile(b_vals, 75))

    a_range = (a_q1 - _RANGE_MARGIN, a_q3 + _RANGE_MARGIN)
    b_range = (b_q1 - _RANGE_MARGIN, b_q3 + _RANGE_MARGIN)

    if log:
        logger.info("Step 6: Skin colour (%d samples)", len(sampled))
        logger.info(
            "  a* [%.1f … %.1f]  (Q1=%.1f Q3=%.1f)",
            a_range[0], a_range[1], a_q1, a_q3,
        )
        logger.info(
            "  b* [%.1f … %.1f]  (Q1=%.1f Q3=%.1f)",
            b_range[0], b_range[1], b_q1, b_q3,
        )

    # ── 7. Mab — skin colour mask ──────────────────────────────────────────
    mab = _build_mab(image_lab, m2, a_range, b_range)
    if log:
        logger.info("Step 7: Mab — %d px", np.count_nonzero(mab))

    # ── 8. M3 — morphological cleanup ──────────────────────────────────────
    m3 = _morphological_cleanup(mab)
    if log:
        logger.info("Step 8: M3 (close→open→dilate) — %d px", np.count_nonzero(m3))

    # ── 9. Output = M3 ∩ M2 ───────────────────────────────────────────────
    output_proc = cv.bitwise_and(m3, m2)
    if log:
        logger.info("Step 9: M3 ∩ M2 — %d px", np.count_nonzero(output_proc))

    # ── 10. Upscale to original resolution ─────────────────────────────────
    if scale < 1.0:
        output_mask = cv.resize(
            output_proc, (image_w, image_h), interpolation=cv.INTER_NEAREST,
        )
    else:
        output_mask = output_proc

    hand_area = int(np.count_nonzero(output_mask))
    area_ratio = hand_area / (image_h * image_w)

    if area_ratio < _MIN_HAND_AREA_RATIO:
        return _fail(
            f"hand area too small ({area_ratio:.4f} < {_MIN_HAND_AREA_RATIO})"
        )

    if log:
        logger.info(
            "Final mask: %d px (%.2f%% of image)", hand_area, area_ratio * 100,
        )

    # ── Visualisations ─────────────────────────────────────────────────────
    image_bgr_full = _float_to_bgr_u8(image)
    viz_landmarks = _draw_landmarks_viz(image_bgr_full, landmarks_px)

    viz_m1 = cv.cvtColor(m1, cv.COLOR_GRAY2BGR)
    viz_m2 = cv.cvtColor(m2, cv.COLOR_GRAY2BGR)

    viz_sample = bgr_proc_u8.copy()
    ys, xs = np.where(m1 > 0)
    rng = np.random.default_rng(42)
    if len(ys) > _N_SAMPLES:
        idx = rng.choice(len(ys), size=_N_SAMPLES, replace=False)
    else:
        idx = np.arange(len(ys))
    for y, x in zip(ys[idx], xs[idx]):
        cv.circle(viz_sample, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv.putText(
        viz_sample,
        f"a*[{a_range[0]:.0f},{a_range[1]:.0f}] b*[{b_range[0]:.0f},{b_range[1]:.0f}]",
        (10, proc_h - 10),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
    )

    viz_mab = cv.cvtColor(mab, cv.COLOR_GRAY2BGR)
    viz_m3 = cv.cvtColor(m3, cv.COLOR_GRAY2BGR)
    viz_output = cv.cvtColor(output_proc, cv.COLOR_GRAY2BGR)
    viz_overlay = _draw_mask_overlay(image_bgr_full, output_mask)

    titles = [
        "Landmarks", "M1 skeleton", "M2 dilated", "Skin samples",
        "Mab colour", "M3 morph", "Output mask", "Overlay",
    ]
    images = [
        viz_landmarks, viz_m1, viz_m2, viz_sample,
        viz_mab, viz_m3, viz_output, viz_overlay,
    ]

    if view:
        for title, img in zip(titles, images):
            cv.imshow(f"Hand — {title}", img)
        cv.waitKey(1)

    if save:
        if output_dir is None:
            logger.warning("save=True but output_dir is None — skipping")
        else:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for fname, img in zip(_VIZ_FILENAMES, images):
                _save_bgr(out / fname, img)
            if log:
                logger.info("Saved %d visualisations to %s", len(images), out)

    if log:
        logger.info("=== HAND SEGMENTATION OK ===")

    return HandSegmentationResult(
        detected=True,
        failure_reason=None,
        n_hands=n_hands,
        landmarks=landmarks_px,
        mask=output_mask,
        hand_area_px=hand_area,
        handedness=handedness,
        confidence=confidence,
    )
