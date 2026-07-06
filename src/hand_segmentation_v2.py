"""
hand_segmentation_v2.py — Skin segmentation via MediaPipe multiclass model.

Uses the ``selfie_multiclass_256x256`` model to classify every pixel as one
of six semantic classes (background, hair, body-skin, face-skin, clothes,
others).  The ``body-skin`` class provides a pixel-accurate skin mask that
covers the hand **and** forearm without leaking onto background objects.

Optionally combines with MediaPipe Hands landmarks so the caller still gets
the 21 hand keypoints (needed for ROI sub-region analysis).

Pipeline:
    1. MediaPipe Image Segmenter → 6-class confidence maps
    2. Threshold ``body-skin`` confidence → binary mask
    3. (Optional) MediaPipe Hands → 21 landmarks
    4. Return :class:`HandSegmentationResult`

References:
    MediaPipe Image Segmenter — Multiclass selfie segmentation.
    https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter

Usage::

    from src.hand_segmentation_v2 import detect_and_segment

    result = detect_and_segment(image_srgb, save_steps=True, log=True,
                                output_dir=Path("debug_output"))
    if result.detected:
        skin_mask = result.mask        # uint8 (H, W), values 0 or 255
        landmarks = result.landmarks   # (21, 2) float32 or None
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
    ImageSegmenter,
    ImageSegmenterOptions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models"
_SEGMENTER_MODEL: Path = _MODEL_DIR / "selfie_multiclass_256x256.tflite"
_HAND_LANDMARKER_MODEL: Path = _MODEL_DIR / "hand_landmarker.task"

# Multiclass segmenter output indices
_CLASS_BACKGROUND: int = 0
_CLASS_HAIR: int = 1
_CLASS_BODY_SKIN: int = 2
_CLASS_FACE_SKIN: int = 3
_CLASS_CLOTHES: int = 4
_CLASS_OTHERS: int = 5

_SKIN_CONFIDENCE_THRESHOLD: float = 0.5
_MIN_SKIN_AREA_RATIO: float = 0.005  # reject if skin < 0.5 % of image

# MediaPipe Hands settings
_MP_MAX_NUM_HANDS: int = 2
_MP_MIN_DETECTION_CONFIDENCE: float = 0.5
_MP_MIN_HAND_PRESENCE_CONFIDENCE: float = 0.5

# Morphological cleanup
_OPEN_SIZE: int = 5   # remove small noise specks
_CLOSE_SIZE: int = 7  # fill small holes within skin

# Visualisation file names (when save_steps=True)
_VIZ_FILENAMES: list[str] = [
    "viz_01_body_skin_mask.jpg",
    "viz_02_confidence_heatmap.jpg",
    "viz_03_overlay.jpg",
    "viz_04_landmarks.jpg",
]

_FINGERTIP_INDICES: list[int] = [4, 8, 12, 16, 20]
_SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HandSegmentationResult:
    """Structured output of skin detection and segmentation.

    On success (``detected=True``) the mask is always populated.
    Landmarks may be ``None`` if ``detect_landmarks=False`` or if
    MediaPipe Hands found no hand in the image.
    """

    detected: bool
    failure_reason: str | None
    mask: np.ndarray | None        # (H, W) uint8, 0/255, original resolution
    skin_area_px: int | None
    landmarks: np.ndarray | None   # (21, 2) float32, pixel coords, or None
    n_hands: int | None
    handedness: str | None         # "Left" | "Right" | None
    confidence: float | None       # hand detection confidence, or None
    skin_confidence: np.ndarray | None  # (H, W) float32, raw body-skin prob


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _float_to_bgr_u8(image_srgb: np.ndarray) -> np.ndarray:
    return (np.clip(image_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)[
        :, :, ::-1
    ]  # RGB→BGR via slice (no copy needed for imwrite)


def _save_bgr(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), image_bgr)


def _draw_mask_overlay(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    color: tuple[int, int, int] = (0, 200, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    vis = image_bgr.copy()
    overlay = vis.copy()
    overlay[mask_u8 > 0] = color
    cv.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)
    contours, _ = cv.findContours(mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(vis, contours, -1, color, 2)
    return vis


def _draw_landmarks_on(
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
    return vis


def _select_best_hand(
    result, image_h: int, image_w: int
) -> tuple[np.ndarray | None, str, float]:
    """Pick the hand with the largest bounding-box area."""
    best_idx, best_area = 0, 0
    for i, hand_lm in enumerate(result.hand_landmarks):
        xs = [lm.x for lm in hand_lm]
        ys = [lm.y for lm in hand_lm]
        area = (max(xs) - min(xs)) * image_w * (max(ys) - min(ys)) * image_h
        if area > best_area:
            best_area = area
            best_idx = i

    hand_lm = result.hand_landmarks[best_idx]
    landmarks_px = np.array(
        [[lm.x * image_w, lm.y * image_h] for lm in hand_lm],
        dtype=np.float32,
    )
    handedness = result.handedness[best_idx][0].category_name
    confidence = result.handedness[best_idx][0].score
    return landmarks_px, handedness, confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_and_segment(
    image_srgb: np.ndarray,
    detect_landmarks: bool = True,
    skin_threshold: float = _SKIN_CONFIDENCE_THRESHOLD,
    show_steps: bool = False,
    save_steps: bool = False,
    log: bool = False,
    output_dir: Path | None = None,
) -> HandSegmentationResult:
    """Detect skin and (optionally) hand landmarks.

    Args:
        image_srgb: float32 RGB [0, 1], shape (H, W, 3).
        detect_landmarks: Also run MediaPipe Hands for 21 keypoints.
            If ``False``, ``landmarks`` / ``handedness`` / ``confidence``
            will be ``None`` in the result.
        skin_threshold: Confidence threshold for the body-skin class.
            Pixels with ``P(body-skin) >= threshold`` are included.
        show_steps: Open non-blocking OpenCV windows with debug overlays.
        save_steps: Save debug visualisation images to *output_dir*.
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
            mask=None, skin_area_px=None, landmarks=None,
            n_hands=None, handedness=None, confidence=None,
            skin_confidence=None,
        )

    image_h, image_w = image_srgb.shape[:2]

    if log:
        logger.info("=== SKIN SEGMENTATION (v2 — multiclass) ===")
        logger.info("Image: %d × %d px", image_w, image_h)

    # ── 1. Multiclass segmentation ─────────────────────────────────────
    if not _SEGMENTER_MODEL.exists():
        return _fail(f"segmenter model not found: {_SEGMENTER_MODEL}")

    image_srgb_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_srgb_u8)

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(_SEGMENTER_MODEL)),
        output_confidence_masks=True,
        output_category_mask=False,
    )
    segmenter = ImageSegmenter.create_from_options(options)
    seg_result = segmenter.segment(mp_image)
    segmenter.close()

    body_skin_conf = seg_result.confidence_masks[_CLASS_BODY_SKIN].numpy_view().squeeze()

    if log:
        logger.info("Step 1: Multiclass segmenter — body-skin max=%.3f",
                     float(np.max(body_skin_conf)))

    # ── 2. Threshold → binary mask ─────────────────────────────────────
    raw_mask_u8 = (body_skin_conf >= skin_threshold).astype(np.uint8) * 255

    # Light morphological cleanup: open removes specks, close fills gaps
    k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (_OPEN_SIZE, _OPEN_SIZE))
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (_CLOSE_SIZE, _CLOSE_SIZE))
    mask_u8 = cv.morphologyEx(raw_mask_u8, cv.MORPH_OPEN, k_open)
    mask_u8 = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, k_close)

    skin_area = int(np.count_nonzero(mask_u8))
    area_ratio = skin_area / (image_h * image_w)

    if log:
        logger.info("Step 2: Threshold %.2f → %d px (%.2f%%)",
                     skin_threshold, skin_area, area_ratio * 100)

    if area_ratio < _MIN_SKIN_AREA_RATIO:
        return _fail(
            f"skin area too small ({area_ratio:.4f} < {_MIN_SKIN_AREA_RATIO})"
        )

    # ── 3. (Optional) MediaPipe Hands ──────────────────────────────────
    landmarks_px = None
    n_hands = None
    handedness = None
    confidence = None

    if detect_landmarks:
        if not _HAND_LANDMARKER_MODEL.exists():
            if log:
                logger.info("Step 3: Hand landmarker model not found — skipping")
        else:
            lm_options = HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(_HAND_LANDMARKER_MODEL)
                ),
                num_hands=_MP_MAX_NUM_HANDS,
                min_hand_detection_confidence=_MP_MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=_MP_MIN_HAND_PRESENCE_CONFIDENCE,
            )
            detector = HandLandmarker.create_from_options(lm_options)
            lm_result = detector.detect(mp_image)
            detector.close()

            if lm_result.hand_landmarks:
                n_hands = len(lm_result.hand_landmarks)
                landmarks_px, handedness, confidence = _select_best_hand(
                    lm_result, image_h, image_w,
                )
                if log:
                    logger.info(
                        "Step 3: %d hand(s) — %s (conf=%.3f)",
                        n_hands, handedness, confidence,
                    )
            elif log:
                logger.info("Step 3: MediaPipe Hands — 0 landmarks")

    if log:
        logger.info("=== SKIN SEGMENTATION OK — %d px (%.1f%%) ===",
                     skin_area, area_ratio * 100)

    # ── Visualisations ─────────────────────────────────────────────────
    image_bgr = _float_to_bgr_u8(image_srgb)

    viz_mask_bgr = cv.cvtColor(mask_u8, cv.COLOR_GRAY2BGR)

    heatmap_u8 = (body_skin_conf * 255).astype(np.uint8)
    viz_heatmap_bgr = cv.applyColorMap(heatmap_u8, cv.COLORMAP_JET)

    viz_overlay_bgr = _draw_mask_overlay(image_bgr, mask_u8)

    if landmarks_px is not None:
        viz_landmarks_bgr = _draw_landmarks_on(viz_overlay_bgr, landmarks_px)
    else:
        viz_landmarks_bgr = viz_overlay_bgr.copy()

    titles = ["Body-skin mask", "Confidence heatmap", "Overlay", "Landmarks"]
    images = [viz_mask_bgr, viz_heatmap_bgr, viz_overlay_bgr, viz_landmarks_bgr]

    if show_steps:
        for title, viz_bgr in zip(titles, images):
            cv.imshow(f"Skin — {title}", viz_bgr)
        cv.waitKey(1)

    if save_steps:
        if output_dir is None:
            logger.warning("save_steps=True but output_dir is None — skipping")
        else:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for fname, viz_bgr in zip(_VIZ_FILENAMES, images):
                _save_bgr(out / fname, viz_bgr)
            if log:
                logger.info("Saved %d visualisations to %s", len(images), out)

    return HandSegmentationResult(
        detected=True,
        failure_reason=None,
        mask=mask_u8,
        skin_area_px=skin_area,
        landmarks=landmarks_px,
        n_hands=n_hands,
        handedness=handedness,
        confidence=confidence,
        skin_confidence=body_skin_conf,
    )
