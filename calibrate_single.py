"""
Single-image calibration pipeline.

Detects BOTH the ColorChecker (SCK300) and a euro banknote in the scene,
segments the hand, measures skin tone, then applies every requested
correction method **twice** — once using ColorChecker reference colours,
once using banknote reference colours — and re-measures.

This produces rows for direct comparison of CC-based vs banknote-based
calibration under identical conditions.

Returns one :class:`~src.evaluation.ImageResult` per (method, reference_object)
combination, plus a single ``baseline`` row with uncalibrated measurements.

This module is a **pure compute function** — it never touches the
filesystem for I/O beyond optional debug visualisations.  Image loading
and CSV writing are the caller's responsibility.

Typical usage::

    from calibrate_single import calibrate_image

    image_srgb = load_my_image(path)
    results = calibrate_image(
        image_srgb,
        denomination=20,           # euro banknote denomination
        side="double",             # "single" or "double"
        ref_dir=Path("data/ref"),  # banknote reference directory
    )

    from src.evaluation import write_results_csv
    write_results_csv(results, "results.csv")

Pipeline order
--------------
1. Detect ColorChecker (SCK300) → swatch colours + reference
2. Detect banknote → grid-sampled colours + reference (if denomination given)
3. Segment hand → binary mask
4. Measure skin tone on the **original** image (baseline)
5. For each correction method:
   a. CC path: fit correction from CC swatch pairs → apply → measure
   b. BN path: fit correction from banknote sample pairs → apply → measure
   c. Both paths get CC ΔE00 (ground truth) AND BN ΔE00 (supplementary)
6. Return one ImageResult per (method, reference) + one baseline row
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.color_calibration import (
    METHODS,
    adapt_d50_to_d65,
    apply_calibration,
    calibrate,
    correct_swatches,
    srgb_to_xyz,
    srgb_to_xyz_samples,
    xyz_to_srgb,
)
from src.colorchecker import ColorCheckerResult, detect_and_measure
from src.evaluation import (
    DeltaEStats,
    ImageResult,
    compute_delta_e00,
    compute_delta_e00_mixed,
    delta_e_stats,
    fill_bn_delta_e,
    fill_bn_meta,
    fill_cc_delta_e,
    fill_skin_tone,
)
from src.hand_segmentation_v2 import detect_and_segment
from src.skin_measurement import SkinToneValues, measure_skin_tone

logger = logging.getLogger(__name__)

# Reference-based methods require a detected reference object (CC or banknote).
_REFERENCE_METHODS: frozenset[str] = frozenset(
    ("linear", "affine", "poly2", "poly3_cheung")
)

# Baseline methods work on the full image without any reference object.
_BASELINE_METHODS: frozenset[str] = frozenset(("gray_world", "shades_of_gray"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calibrate_image(
    image_srgb: np.ndarray,
    methods: str | list[str] = "all",
    *,
    image_path: str | None = None,
    lighting_id: str | None = None,
    person_id: int | None = None,
    hand: str | None = None,
    hand_side: str | None = None,
    split: str | None = None,
    denomination: int | None = None,
    banknote_side: str | None = None,
    side: str | None = None,
    ref_dir: Path | None = None,
    cell_size: int = 32,
    show_steps: bool = False,
    save_steps: bool = False,
    save_corrected: bool = True,
    log: bool = False,
    output_dir: Path | None = None,
) -> list[ImageResult]:
    """Run the full calibration pipeline on a single image.

    Parameters
    ----------
    image_srgb : (H, W, 3) float32 sRGB [0, 1].
        Already loaded image — this function never reads from disk.
    methods : str or list[str]
        ``"all"`` runs every method in :data:`~src.color_calibration.METHODS`.
    image_path : str, optional
        Original filename — stored in :attr:`ImageResult.image_path`.
    lighting_id, person_id, hand, hand_side, split : optional
        Metadata forwarded into every returned :class:`ImageResult`.
    denomination : int, optional
        Euro banknote denomination (5, 10, 20, 50, 100).  When provided
        together with *side* and *ref_dir*, banknote detection runs
        alongside ColorChecker detection.
    side : str, optional
        ``"single"`` or ``"double"`` — which side of the banknote.
    ref_dir : Path, optional
        Directory containing banknote reference data (PNGs + samples).
    cell_size : int
        Grid cell size for banknote sampling (default 32).
    show_steps, save_steps, log : bool
        Control debug visualisations and logging. ``save_steps`` only saves
        intermediate/debug artefacts; it does not control corrected outputs.
    save_corrected : bool
        Save final corrected images for each method/reference pair when
        ``output_dir`` is set. Defaults to True.
    output_dir : Path, optional
        Root directory for corrected images and optional debug images.

    Returns
    -------
    list[ImageResult]
        One ``baseline`` row, then one row per (method, reference_object).
        For each method, up to two rows appear: ``reference_object="colorchecker"``
        and ``reference_object="banknote"`` (when banknote is detected).
    """
    method_list = _resolve_methods(methods)
    stem = Path(image_path).stem if image_path else "unknown"

    # ══════════════════════════════════════════════════════════════════
    # 1. Detection phase — run ONCE, reuse for all methods
    # ══════════════════════════════════════════════════════════════════

    if log:
        logger.info("=" * 60)
        logger.info("calibrate_image: %s (%d methods)", stem, len(method_list))
        logger.info("=" * 60)

    # ── 1a. ColorChecker ──────────────────────────────────────────────
    cc_dir = Path(output_dir) / "colorchecker" if output_dir and save_steps else None
    cc_result = detect_and_measure(
        image_srgb,
        show_steps=show_steps,
        save_steps=save_steps,
        log=log,
        output_dir=cc_dir,
    )

    if log:
        if cc_result.detected:
            logger.info(
                "ColorChecker: DETECTED (%d swatches, ΔE00 mean=%.2f)",
                cc_result.n_swatches,
                cc_result.delta_e_stats["mean"],
            )
        else:
            logger.info("ColorChecker: NOT DETECTED (%s)", cc_result.failure_reason)

    # ── 1b. Banknote ─────────────────────────────────────────────────
    bn_result = None
    if denomination is not None and side is not None and ref_dir is not None:
        from src.banknote_detection import detect_and_measure_banknote

        bn_dir = Path(output_dir) / "banknote" if output_dir and save_steps else None
        bn_result = detect_and_measure_banknote(
            image_srgb,
            denomination=denomination,
            side=side,
            ref_dir=ref_dir,
            cell_size=cell_size,
            show_steps=show_steps,
            save_steps=save_steps,
            log=log,
            output_dir=bn_dir,
        )

        if log:
            if bn_result.detected:
                logger.info(
                    "Banknote: DETECTED (%d€ %s, %d samples, conf=%.2f, RMSE=%.2f)",
                    denomination,
                    side,
                    bn_result.n_samples,
                    bn_result.confidence,
                    bn_result.reproj_rmse,
                )
            else:
                logger.info("Banknote: NOT DETECTED (%s)", bn_result.failure_reason)

    # ── 1c. Hand segmentation ────────────────────────────────────────
    seg_dir = Path(output_dir) / "segmentation" if output_dir and save_steps else None
    seg_result = detect_and_segment(
        image_srgb,
        show_steps=show_steps,
        save_steps=save_steps,
        log=log,
        output_dir=seg_dir,
    )

    if log:
        if seg_result.detected:
            logger.info(
                "Hand: DETECTED (%d px, %s)",
                seg_result.skin_area_px,
                seg_result.handedness or "no landmarks",
            )
        else:
            logger.info("Hand: NOT DETECTED (%s)", seg_result.failure_reason)

    # ══════════════════════════════════════════════════════════════════
    # 2. Pre-compute shared data — run ONCE
    # ══════════════════════════════════════════════════════════════════

    # ── 2a. Skin tone on original image ──────────────────────────────
    skin_before: SkinToneValues | None = None
    if seg_result.detected:
        skin_dir = (
            Path(output_dir) / "skin_before" if output_dir and save_steps else None
        )
        skin_before = measure_skin_tone(
            image_srgb,
            seg_result.mask,
            show_steps=show_steps,
            save_steps=save_steps,
            log=log,
            output_dir=skin_dir,
        )

    # ── 2b. ColorChecker ΔE00 before calibration ────────────────────
    de_cc_before: DeltaEStats | None = None
    cc_measured_xyz_d65: np.ndarray | None = None
    cc_reference_xyz_d50: np.ndarray | None = None
    cc_reference_xyz_d65: np.ndarray | None = None

    if cc_result.detected:
        cc_measured_xyz_d65 = cc_result.swatch_colors_xyz
        cc_reference_xyz_d50 = cc_result.reference_colors_xyz
        cc_reference_xyz_d65 = adapt_d50_to_d65(cc_reference_xyz_d50)

        de_arr = compute_delta_e00_mixed(cc_measured_xyz_d65, cc_reference_xyz_d50)
        de_cc_before = delta_e_stats(de_arr)

    # ── 2c. Banknote ΔE00 before calibration ─────────────────────────
    de_bn_before: DeltaEStats | None = None
    bn_measured_xyz_d65: np.ndarray | None = None
    bn_reference_xyz_d65: np.ndarray | None = None

    if bn_result is not None and bn_result.detected:
        # Convert sRGB samples → XYZ-D65
        bn_measured_xyz_d65 = srgb_to_xyz_samples(bn_result.measured_srgb)
        bn_reference_xyz_d65 = srgb_to_xyz_samples(bn_result.reference_srgb)

        # Both are D65 — use compute_delta_e00 (not _mixed)
        de_arr = compute_delta_e00(bn_measured_xyz_d65, bn_reference_xyz_d65)
        de_bn_before = delta_e_stats(de_arr)

        if log:
            logger.info(
                "Banknote ΔE00 before: mean=%.2f, median=%.2f, p95=%.2f",
                de_bn_before.mean,
                de_bn_before.median,
                de_bn_before.p95,
            )

    # ── 2d. Pre-convert image to XYZ (shared by all methods) ─────────
    image_xyz_d65 = srgb_to_xyz(image_srgb)

    # ══════════════════════════════════════════════════════════════════
    # 3. Build common metadata for ImageResult rows
    # ══════════════════════════════════════════════════════════════════

    common = dict(
        image_path=image_path or "",
        image_stem=stem,
        lighting_id=lighting_id,
        person_id=person_id,
        hand=hand,
        hand_side=hand_side,
        split=split,
        denomination=denomination,
        banknote_side=banknote_side,
    )

    results: list[ImageResult] = []

    bn_detected = bn_result is not None and bn_result.detected

    # ── 3a. Baseline row (no calibration) ────────────────────────────
    baseline = ImageResult(**common, correction_method="baseline")
    fill_cc_delta_e(baseline, de_cc_before, None)
    fill_bn_delta_e(baseline, de_bn_before, None)
    if bn_detected:
        fill_bn_meta(baseline, bn_result)
    fill_skin_tone(baseline, skin_before, None)
    # If nothing was detected at all, mark baseline as failed
    if not cc_result.detected and not seg_result.detected and not bn_detected:
        baseline.success = False
        baseline.failure_reason = _combined_failure(cc_result, seg_result, bn_result)
    results.append(baseline)

    # ══════════════════════════════════════════════════════════════════
    # 4. Per-method calibration loop
    # ══════════════════════════════════════════════════════════════════

    for method in method_list:
        if log:
            logger.info("── Method: %s ──", method)

        # For reference-based methods, try BOTH reference objects.
        # For baseline methods (gray_world, shades_of_gray), one row only.
        if method in _REFERENCE_METHODS:
            ref_sources: list[tuple[str, np.ndarray | None, np.ndarray | None]] = []
            if cc_result.detected:
                ref_sources.append(
                    ("colorchecker", cc_measured_xyz_d65, cc_reference_xyz_d65)
                )
            if bn_detected:
                ref_sources.append(
                    ("banknote", bn_measured_xyz_d65, bn_reference_xyz_d65)
                )

            if not ref_sources:
                # Neither reference detected — record failure
                row = ImageResult(**common, correction_method=method)
                fill_cc_delta_e(row, de_cc_before, None)
                fill_bn_delta_e(row, de_bn_before, None)
                if bn_detected:
                    fill_bn_meta(row, bn_result)
                fill_skin_tone(row, skin_before, None)
                row.success = False
                row.failure_reason = "no reference object detected"
                results.append(row)
                if log:
                    logger.info("  SKIP %s — no reference object detected", method)
                continue

            for ref_obj, measured_xyz, reference_xyz in ref_sources:
                row = _run_reference_method(
                    method=method,
                    ref_obj=ref_obj,
                    measured_xyz_d65=measured_xyz,
                    reference_xyz_d65=reference_xyz,
                    image_xyz_d65=image_xyz_d65,
                    seg_result=seg_result,
                    skin_before=skin_before,
                    cc_measured_xyz_d65=cc_measured_xyz_d65,
                    cc_reference_xyz_d50=cc_reference_xyz_d50,
                    bn_measured_xyz_d65=bn_measured_xyz_d65,
                    bn_reference_xyz_d65=bn_reference_xyz_d65,
                    de_cc_before=de_cc_before,
                    de_bn_before=de_bn_before,
                    bn_result=bn_result,
                    common=common,
                    show_steps=show_steps,
                    save_steps=save_steps,
                    save_corrected=save_corrected,
                    log=log,
                    output_dir=output_dir,
                )
                results.append(row)

        else:
            # Baseline method (gray_world, shades_of_gray) — no reference object
            row = _run_baseline_method(
                method=method,
                image_xyz_d65=image_xyz_d65,
                seg_result=seg_result,
                skin_before=skin_before,
                cc_measured_xyz_d65=cc_measured_xyz_d65,
                cc_reference_xyz_d50=cc_reference_xyz_d50,
                bn_measured_xyz_d65=bn_measured_xyz_d65,
                bn_reference_xyz_d65=bn_reference_xyz_d65,
                de_cc_before=de_cc_before,
                de_bn_before=de_bn_before,
                bn_result=bn_result,
                common=common,
                show_steps=show_steps,
                save_steps=save_steps,
                save_corrected=save_corrected,
                log=log,
                output_dir=output_dir,
            )
            results.append(row)

    if log:
        logger.info("calibrate_image: %s done — %d results", stem, len(results))

    return results


# ---------------------------------------------------------------------------
# Private helpers — calibration runners
# ---------------------------------------------------------------------------


def _run_reference_method(
    *,
    method: str,
    ref_obj: str,
    measured_xyz_d65: np.ndarray,
    reference_xyz_d65: np.ndarray,
    image_xyz_d65: np.ndarray,
    seg_result,
    skin_before: SkinToneValues | None,
    cc_measured_xyz_d65: np.ndarray | None,
    cc_reference_xyz_d50: np.ndarray | None,
    bn_measured_xyz_d65: np.ndarray | None,
    bn_reference_xyz_d65: np.ndarray | None,
    de_cc_before: DeltaEStats | None,
    de_bn_before: DeltaEStats | None,
    bn_result,
    common: dict,
    show_steps: bool,
    save_steps: bool,
    save_corrected: bool,
    log: bool,
    output_dir: Path | None,
) -> ImageResult:
    """Run a single reference-based method and return an ImageResult row."""
    row = ImageResult(**common, correction_method=method, reference_object=ref_obj)
    fill_cc_delta_e(row, de_cc_before, None)
    fill_bn_delta_e(row, de_bn_before, None)
    if bn_result is not None and bn_result.detected:
        fill_bn_meta(row, bn_result)
    fill_skin_tone(row, skin_before, None)

    if log:
        logger.info("  %s ← %s", method, ref_obj)

    # ── Fit calibration ──────────────────────────────────────────────
    params = calibrate(
        method,
        measured_xyz_d65=measured_xyz_d65,
        reference_xyz_d65=reference_xyz_d65,
    )

    # ── Apply calibration to full image ──────────────────────────────
    corrected_xyz_d65 = apply_calibration(image_xyz_d65, params)
    corrected_srgb = xyz_to_srgb(corrected_xyz_d65)

    # ── CC ΔE00 after (ground truth evaluation) ──────────────────────
    _fill_cc_de_after(
        row,
        params,
        cc_measured_xyz_d65,
        cc_reference_xyz_d50,
        de_cc_before,
        log,
    )

    # ── BN ΔE00 after (supplementary) ────────────────────────────────
    _fill_bn_de_after(
        row,
        params,
        bn_measured_xyz_d65,
        bn_reference_xyz_d65,
        de_bn_before,
        log,
    )

    # ── Skin tone on corrected image ─────────────────────────────────
    _fill_skin_after(
        row,
        corrected_srgb,
        seg_result,
        skin_before,
        method,
        ref_obj,
        show_steps,
        save_steps,
        log,
        output_dir,
    )

    # ── Save corrected image ─────────────────────────────────────────
    _save_corrected(corrected_srgb, method, ref_obj, save_corrected, output_dir)

    row.success = True
    return row


def _run_baseline_method(
    *,
    method: str,
    image_xyz_d65: np.ndarray,
    seg_result,
    skin_before: SkinToneValues | None,
    cc_measured_xyz_d65: np.ndarray | None,
    cc_reference_xyz_d50: np.ndarray | None,
    bn_measured_xyz_d65: np.ndarray | None,
    bn_reference_xyz_d65: np.ndarray | None,
    de_cc_before: DeltaEStats | None,
    de_bn_before: DeltaEStats | None,
    bn_result,
    common: dict,
    show_steps: bool,
    save_steps: bool,
    save_corrected: bool,
    log: bool,
    output_dir: Path | None,
) -> ImageResult:
    """Run a baseline method (no reference object) and return an ImageResult row."""
    row = ImageResult(**common, correction_method=method, reference_object="none")
    fill_cc_delta_e(row, de_cc_before, None)
    fill_bn_delta_e(row, de_bn_before, None)
    if bn_result is not None and bn_result.detected:
        fill_bn_meta(row, bn_result)
    fill_skin_tone(row, skin_before, None)

    if log:
        logger.info("  %s (baseline — no reference)", method)

    # ── Fit calibration (from image statistics) ──────────────────────
    params = calibrate(method, image_xyz_d65=image_xyz_d65)

    # ── Apply calibration to full image ──────────────────────────────
    corrected_xyz_d65 = apply_calibration(image_xyz_d65, params)
    corrected_srgb = xyz_to_srgb(corrected_xyz_d65)

    # ── CC ΔE00 after ────────────────────────────────────────────────
    _fill_cc_de_after(
        row,
        params,
        cc_measured_xyz_d65,
        cc_reference_xyz_d50,
        de_cc_before,
        log,
    )

    # ── BN ΔE00 after ────────────────────────────────────────────────
    _fill_bn_de_after(
        row,
        params,
        bn_measured_xyz_d65,
        bn_reference_xyz_d65,
        de_bn_before,
        log,
    )

    # ── Skin tone on corrected image ─────────────────────────────────
    _fill_skin_after(
        row,
        corrected_srgb,
        seg_result,
        skin_before,
        method,
        "none",
        show_steps,
        save_steps,
        log,
        output_dir,
    )

    # ── Save corrected image ─────────────────────────────────────────
    _save_corrected(corrected_srgb, method, "none", save_corrected, output_dir)

    row.success = True
    return row


# ---------------------------------------------------------------------------
# Private helpers — shared logic
# ---------------------------------------------------------------------------


def _fill_cc_de_after(
    row: ImageResult,
    params,
    cc_measured_xyz_d65: np.ndarray | None,
    cc_reference_xyz_d50: np.ndarray | None,
    de_cc_before: DeltaEStats | None,
    log: bool,
) -> None:
    """Compute and fill CC ΔE00 after applying calibration to CC swatches."""
    if cc_measured_xyz_d65 is None or cc_reference_xyz_d50 is None:
        return
    corrected_swatches_xyz_d65 = correct_swatches(cc_measured_xyz_d65, params)
    de_arr_after = compute_delta_e00_mixed(
        corrected_swatches_xyz_d65, cc_reference_xyz_d50
    )
    de_after = delta_e_stats(de_arr_after)
    fill_cc_delta_e(row, de_cc_before, de_after)

    if log and de_cc_before is not None:
        logger.info(
            "    CC ΔE00: %.2f → %.2f (Δ%+.2f)",
            de_cc_before.mean,
            de_after.mean,
            de_cc_before.mean - de_after.mean,
        )


def _fill_bn_de_after(
    row: ImageResult,
    params,
    bn_measured_xyz_d65: np.ndarray | None,
    bn_reference_xyz_d65: np.ndarray | None,
    de_bn_before: DeltaEStats | None,
    log: bool,
) -> None:
    """Compute and fill BN ΔE00 after applying calibration to BN samples."""
    if bn_measured_xyz_d65 is None or bn_reference_xyz_d65 is None:
        return
    corrected_bn_xyz = correct_swatches(bn_measured_xyz_d65, params)
    # Both are D65 — use compute_delta_e00
    de_arr_after = compute_delta_e00(corrected_bn_xyz, bn_reference_xyz_d65)
    de_after = delta_e_stats(de_arr_after)
    fill_bn_delta_e(row, de_bn_before, de_after)

    if log and de_bn_before is not None:
        logger.info(
            "    BN ΔE00: %.2f → %.2f (Δ%+.2f)",
            de_bn_before.mean,
            de_after.mean,
            de_bn_before.mean - de_after.mean,
        )


def _fill_skin_after(
    row: ImageResult,
    corrected_srgb: np.ndarray,
    seg_result,
    skin_before: SkinToneValues | None,
    method: str,
    ref_obj: str,
    show_steps: bool,
    save_steps: bool,
    log: bool,
    output_dir: Path | None,
) -> None:
    """Measure skin tone on the corrected image and fill the row."""
    if not seg_result.detected:
        return
    method_skin_dir = (
        Path(output_dir) / f"{method}_{ref_obj}" / "skin_after"
        if output_dir and save_steps
        else None
    )
    skin_after = measure_skin_tone(
        corrected_srgb,
        seg_result.mask,
        show_steps=show_steps,
        save_steps=save_steps,
        log=log,
        output_dir=method_skin_dir,
    )
    fill_skin_tone(row, skin_before, skin_after)

    if log and skin_before and skin_after:
        logger.info(
            "    ITA: %.1f° → %.1f° (%s → %s)",
            skin_before.ITA,
            skin_after.ITA,
            skin_before.chardon_category,
            skin_after.chardon_category,
        )


def _save_corrected(
    corrected_srgb: np.ndarray,
    method: str,
    ref_obj: str,
    save_corrected: bool,
    output_dir: Path | None,
) -> None:
    """Save the corrected image to disk when enabled.

    ``save_corrected`` is independent from ``save_steps``.  Final corrected
    images are primary outputs; step images are debug artefacts.
    """
    if not save_corrected or output_dir is None:
        return
    import cv2 as cv

    method_dir = Path(output_dir) / f"{method}_{ref_obj}"
    method_dir.mkdir(parents=True, exist_ok=True)
    corrected_bgr = cv.cvtColor(
        (np.clip(corrected_srgb, 0, 1) * 255 + 0.5).astype(np.uint8),
        cv.COLOR_RGB2BGR,
    )
    cv.imwrite(str(method_dir / "corrected.jpg"), corrected_bgr)


def _resolve_methods(methods: str | list[str]) -> list[str]:
    """Normalise the *methods* argument to a concrete list of method names."""
    if isinstance(methods, str):
        if methods == "all":
            return list(METHODS)
        if methods not in METHODS:
            raise ValueError(
                f"Unknown method {methods!r}. Choose from {METHODS} or 'all'."
            )
        return [methods]

    for m in methods:
        if m not in METHODS:
            raise ValueError(f"Unknown method {m!r}. Choose from {METHODS}.")
    return list(methods)


def _combined_failure(
    cc: ColorCheckerResult,
    seg,
    bn=None,
) -> str:
    """Build a failure_reason string when all detection steps failed."""
    parts: list[str] = []
    if not cc.detected:
        parts.append(f"colorchecker: {cc.failure_reason}")
    if not seg.detected:
        parts.append(f"segmentation: {seg.failure_reason}")
    if bn is not None and not bn.detected:
        parts.append(f"banknote: {bn.failure_reason}")
    return "; ".join(parts) if parts else "unknown"
