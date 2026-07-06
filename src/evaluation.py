"""
Evaluation metrics for color calibration comparison.

Implements the complete evaluation strategy described in the research
documentation.

Metric categories
-----------------
1. **ΔE00 on ColorChecker swatches** — objective calibration quality
2. **ΔE00 component decomposition** — ΔL', ΔC', ΔH' diagnostics
3. **ITA / skin-tone consistency** — proxy metric (no ground truth)
4. **Cross-lighting stability** — std(ITA), std(L*), std(a*), std(b*)
5. **Chardon category consistency** — categorical stability
6. **Angular error** — for illuminant-estimation baselines only
7. **SSIM / PSNR** — global image similarity (secondary)
8. **Colorfulness** — perceptual saturation diagnostic
9. **Statistical tests** — paired Wilcoxon for method comparison
10. **Aggregation** — cross-method tables, improvement matrices

Color-space conventions
-----------------------
- ΔE00 on swatches: XYZ(D65) → Bradford CAT → XYZ(D50) → Lab(D50).
  SCK300 reference values are already D50.
- ITA on skin: sRGB [0,1] → OpenCV Lab (D65 implicit).
  These two Lab spaces MUST NOT be mixed.
- Angular error: linear RGB (never gamma-coded sRGB).
- SSIM / PSNR / colorfulness: sRGB [0,1].

Public API summary
------------------
    # per-patch ΔE00
    compute_delta_e00(a_xyz_d65, b_xyz_d65) → ndarray
    delta_e_stats(de) → DeltaEStats

    # ΔE00 component decomposition
    delta_e_components(a_xyz_d65, b_xyz_d65) → DeltaEComponents

    # angular error
    angular_error(estimated, ground_truth) → float
    angular_error_stats(errors) → dict

    # SSIM / PSNR
    compute_ssim(image_a_srgb, image_b_srgb) → float
    compute_psnr(image_a_srgb, image_b_srgb) → float

    # colorfulness
    compute_colorfulness(image_srgb) → float

    # cross-lighting stability
    cross_lighting_std(values) → CrossLightingStats
    ita_reduction(std_before, std_after) → float

    # Chardon consistency
    chardon_consistency(categories) → bool

    # paired statistical test
    paired_wilcoxon(a, b) → WilcoxonResult

    # aggregation helpers
    method_comparison_matrix(results, methods, metric_key) → DataFrame-like dict
    improvement_over_baseline(baseline_val, method_val) → float

    # result containers
    ImageResult — per-image record
    DeltaEStats — aggregated ΔE statistics
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import colour
import numpy as np
from scipy import stats as sp_stats
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


__all__ = [
    # Dataclasses
    "DeltaEStats",
    "DeltaEComponents",
    "CrossLightingStats",
    "WilcoxonResult",
    "ImageResult",
    # ΔE00
    "compute_delta_e00",
    "compute_delta_e00_mixed",
    "delta_e_stats",
    "delta_e_components",
    # Angular error
    "angular_error",
    "reproduction_angular_error",
    "angular_error_stats",
    # SSIM / PSNR
    "compute_ssim",
    "compute_psnr",
    # Colorfulness
    "compute_colorfulness",
    # ITA / Chardon
    "compute_ita",
    "classify_chardon",
    # Cross-lighting
    "cross_lighting_std",
    "ita_reduction",
    "coefficient_of_variation",
    # Chardon consistency
    "chardon_consistency",
    "chardon_consistency_rate",
    # Statistical tests
    "paired_wilcoxon",
    "pairwise_wilcoxon_matrix",
    # Aggregation
    "improvement_over_baseline",
    "method_comparison_matrix",
    "method_ranking",
    # I/O
    "write_results_csv",
    "read_results_csv",
    "write_results_json",
    # Convenience
    "fill_cc_delta_e",
    "fill_bn_delta_e",
    "fill_bn_meta",
    "fill_skin_tone",
]

# ---------------------------------------------------------------------------
# Illuminant constants — shared across all ΔE computations
# ---------------------------------------------------------------------------

_D50_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
_D65_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
_D50_XYZ = colour.xy_to_XYZ(_D50_XY)
_D65_XYZ = colour.xy_to_XYZ(_D65_XY)
_CHARDON_DARKEST: tuple[str, int] = ("Dark", 6)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DeltaEStats:
    """Full ΔE00 aggregation per Gijsenij 2011 recommendations.

    Fields
    ------
    mean, median, trimean : central tendency (trimean = (Q1+2*med+Q3)/4).
    p95, max              : tail behaviour.
    best25, worst25       : mean of lowest / highest quartile.
    per_patch             : raw per-sample array (e.g. 48 values for SCK300).
    """

    mean: float
    median: float
    trimean: float
    p95: float
    max: float
    best25: float
    worst25: float
    per_patch: list[float]


@dataclass
class DeltaEComponents:
    """Per-patch decomposition into CIELab lightness, chroma, and hue differences.

    NOTE: These are **raw CIELab differences**, not CIEDE2000-weighted terms.
    They diagnose *where* the colour error lies (lightness vs saturation vs hue)
    but do not sum to ΔE00 (which applies S_L, S_C, S_H weighting + R_T rotation).

    All arrays have shape (N,) matching the input patch count.
    """

    delta_L: list[float]  # L*_b − L*_a (positive = b is lighter)
    delta_C: list[float]  # C*_b − C*_a (positive = b is more chromatic)
    delta_h: list[float]  # absolute hue-angle difference in degrees [0, 180]
    # Aggregate means for quick reporting
    mean_abs_dL: float
    mean_abs_dC: float
    mean_dh: float


@dataclass
class CrossLightingStats:
    """Consistency of a scalar measurement across lighting conditions."""

    values: list[float]
    mean: float
    std: float
    range: float  # max - min
    n: int


@dataclass
class WilcoxonResult:
    """Result of a paired Wilcoxon signed-rank test."""

    statistic: float
    p_value: float
    n: int
    significant: bool  # p < 0.05
    median_diff: float


@dataclass
class ImageResult:
    """Per-image evaluation record — one row in results.csv.

    Designed for CSV serialisation: all fields are flat scalars or None.
    """

    # ── Identification ────────────────────────────────────────────────
    image_path: str
    image_stem: str
    lighting_id: str | None = None   # L1–L6
    person_id: int | None = None
    hand: str | None = None          # "left" | "right"
    hand_side: str | None = None     # "dorsal" | "palm"
    split: str | None = None         # "train" | "eval"

    # ── Method ────────────────────────────────────────────────────────
    reference_object: str = "none"   # "colorchecker" | "banknote" | "none"
    correction_method: str = "baseline"
    denomination: int | None = None  # 5, 10, 20, 50, 100 (banknote)
    banknote_side: str | None = None # "single_number" | "double_number"

    # ── ColorChecker ΔE00 (objective ground truth) ────────────────────
    cc_de_mean_before: float | None = None
    cc_de_median_before: float | None = None
    cc_de_trimean_before: float | None = None
    cc_de_p95_before: float | None = None
    cc_de_max_before: float | None = None
    cc_de_best25_before: float | None = None
    cc_de_worst25_before: float | None = None

    cc_de_mean_after: float | None = None
    cc_de_median_after: float | None = None
    cc_de_trimean_after: float | None = None
    cc_de_p95_after: float | None = None
    cc_de_max_after: float | None = None
    cc_de_best25_after: float | None = None
    cc_de_worst25_after: float | None = None

    # ── Banknote ΔE00 (supplementary — samples vs reference) ─────────
    bn_de_mean_before: float | None = None
    bn_de_median_before: float | None = None
    bn_de_trimean_before: float | None = None
    bn_de_p95_before: float | None = None
    bn_de_max_before: float | None = None
    bn_de_best25_before: float | None = None
    bn_de_worst25_before: float | None = None

    bn_de_mean_after: float | None = None
    bn_de_median_after: float | None = None
    bn_de_trimean_after: float | None = None
    bn_de_p95_after: float | None = None
    bn_de_max_after: float | None = None
    bn_de_best25_after: float | None = None
    bn_de_worst25_after: float | None = None

    # ── Banknote detection metadata ───────────────────────────────────
    bn_n_samples: int | None = None
    bn_confidence: float | None = None
    bn_reproj_rmse: float | None = None

    # ── Skin tone (proxy) ─────────────────────────────────────────────
    L_median_before: float | None = None
    a_median_before: float | None = None
    b_median_before: float | None = None
    ITA_before: float | None = None
    chardon_before: str | None = None

    L_median_after: float | None = None
    a_median_after: float | None = None
    b_median_after: float | None = None
    ITA_after: float | None = None
    chardon_after: str | None = None
    skin_measured: bool = False      # True when measure_skin_tone() succeeded

    # ── Status ────────────────────────────────────────────────────────
    success: bool = True
    failure_reason: str | None = None

    # ── helpers ───────────────────────────────────────────────────────

    def cc_improvement_mean(self) -> float | None:
        """ΔE00 mean improvement percentage (positive = better)."""
        if self.cc_de_mean_before is None or self.cc_de_mean_after is None:
            return None
        if self.cc_de_mean_before == 0:
            return 0.0
        return (
            (self.cc_de_mean_before - self.cc_de_mean_after)
            / self.cc_de_mean_before
            * 100.0
        )

    def bn_improvement_mean(self) -> float | None:
        """Banknote ΔE00 mean improvement percentage (positive = better)."""
        if self.bn_de_mean_before is None or self.bn_de_mean_after is None:
            return None
        if self.bn_de_mean_before == 0:
            return 0.0
        return (
            (self.bn_de_mean_before - self.bn_de_mean_after)
            / self.bn_de_mean_before
            * 100.0
        )


# CSV field order — defines the canonical column layout
_IMAGE_RESULT_FIELDS: list[str] = [f.name for f in fields(ImageResult)]


# ═══════════════════════════════════════════════════════════════════════════
# 1. ΔE00 (CIEDE2000) — primary objective metric
# ═══════════════════════════════════════════════════════════════════════════


def _adapt_d65_to_d50(xyz_d65: np.ndarray) -> np.ndarray:
    """Bradford chromatic adaptation XYZ(D65) → XYZ(D50)."""
    adapted = colour.adaptation.chromatic_adaptation_VonKries(
        xyz_d65, _D65_XYZ, _D50_XYZ, transform="Bradford",
    )
    return np.asarray(adapted, dtype=np.float64)


def _xyz_to_lab_d50(xyz_d50: np.ndarray) -> np.ndarray:
    """CIE XYZ (D50) → CIELAB (D50)."""
    return np.asarray(
        colour.XYZ_to_Lab(
            np.asarray(xyz_d50, dtype=np.float64), illuminant=_D50_XY,
        ),
        dtype=np.float64,
    )


def compute_delta_e00(
    colors_a_xyz_d65: np.ndarray,
    colors_b_xyz_d65: np.ndarray,
) -> np.ndarray:
    """Per-sample CIEDE2000 between two (N, 3) XYZ-D65 arrays.

    Both inputs are adapted D65 → D50 internally, then converted to
    Lab(D50) before applying CIE 2000.

    Returns
    -------
    1-D float64 array of shape (N,).
    """
    lab_a_d50 = _xyz_to_lab_d50(_adapt_d65_to_d50(colors_a_xyz_d65))
    lab_b_d50 = _xyz_to_lab_d50(_adapt_d65_to_d50(colors_b_xyz_d65))
    de = colour.delta_E(lab_a_d50, lab_b_d50, method="CIE 2000")
    return np.asarray(de, dtype=np.float64).ravel()


def compute_delta_e00_mixed(
    measured_xyz_d65: np.ndarray,
    reference_xyz_d50: np.ndarray,
) -> np.ndarray:
    """ΔE00 when measured is D65 but reference is already D50.

    This is the exact situation for SCK300: measured swatch colours come
    from the sRGB camera pipeline (D65) while darktable reference values
    are stored as XYZ(D50).
    """
    measured_xyz_d50 = _adapt_d65_to_d50(measured_xyz_d65)
    lab_m_d50 = _xyz_to_lab_d50(measured_xyz_d50)
    lab_r_d50 = _xyz_to_lab_d50(reference_xyz_d50)
    de = colour.delta_E(lab_m_d50, lab_r_d50, method="CIE 2000")
    return np.asarray(de, dtype=np.float64).ravel()


def delta_e_stats(de: np.ndarray) -> DeltaEStats:
    """Full Gijsenij-2011 aggregation of a ΔE00 array.

    Returns a DeltaEStats with mean, median, trimean, p95, max,
    best-25%, worst-25%, and the raw per-patch values.
    """
    de = np.asarray(de, dtype=np.float64).ravel()
    q1 = float(np.percentile(de, 25))
    med = float(np.median(de))
    q3 = float(np.percentile(de, 75))
    trimean = (q1 + 2.0 * med + q3) / 4.0

    return DeltaEStats(
        mean=float(np.mean(de)),
        median=med,
        trimean=trimean,
        p95=float(np.percentile(de, 95)),
        max=float(np.max(de)),
        best25=float(np.mean(de[de <= q1])) if np.any(de <= q1) else float(np.min(de)),
        worst25=float(np.mean(de[de >= q3])) if np.any(de >= q3) else float(np.max(de)),
        per_patch=de.tolist(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. ΔE00 component decomposition — ΔL', ΔC', ΔH'
# ═══════════════════════════════════════════════════════════════════════════


def delta_e_components(
    colors_a_xyz_d65: np.ndarray,
    colors_b_xyz_d65: np.ndarray,
) -> DeltaEComponents:
    """Decompose colour error into raw CIELab lightness, chroma, and hue.

    Computes per-patch:
      ΔL = L*_b − L*_a          (signed lightness difference)
      ΔC = C*_b − C*_a          (signed chroma difference)
      Δh = |h°_b − h°_a|        (absolute hue-angle difference, degrees)

    These are diagnostic — they tell you *where* the error is, not its
    perceptual magnitude (use ΔE00 for that).

    Parameters
    ----------
    colors_a_xyz_d65, colors_b_xyz_d65 : (N, 3) XYZ under D65.
    """
    lab_a_d50 = _xyz_to_lab_d50(_adapt_d65_to_d50(colors_a_xyz_d65))
    lab_b_d50 = _xyz_to_lab_d50(_adapt_d65_to_d50(colors_b_xyz_d65))

    La, aa, ba = lab_a_d50[:, 0], lab_a_d50[:, 1], lab_a_d50[:, 2]
    Lb, ab, bb = lab_b_d50[:, 0], lab_b_d50[:, 1], lab_b_d50[:, 2]

    # Lightness difference (signed)
    dL = Lb - La

    # Chroma difference (signed)
    Ca = np.sqrt(aa**2 + ba**2)
    Cb = np.sqrt(ab**2 + bb**2)
    dC = Cb - Ca

    # Hue-angle difference (absolute, degrees)
    ha = np.degrees(np.arctan2(ba, aa))  # [-180, 180]
    hb = np.degrees(np.arctan2(bb, ab))
    dh_raw = hb - ha
    # Wrap to [-180, 180] then take absolute
    dh_raw = (dh_raw + 180.0) % 360.0 - 180.0
    dh = np.abs(dh_raw)

    # Handle achromatic patches (C ≈ 0 → hue undefined)
    achromatic = (Ca < 1e-6) | (Cb < 1e-6)
    dh[achromatic] = 0.0

    return DeltaEComponents(
        delta_L=dL.tolist(),
        delta_C=dC.tolist(),
        delta_h=dh.tolist(),
        mean_abs_dL=float(np.mean(np.abs(dL))),
        mean_abs_dC=float(np.mean(np.abs(dC))),
        mean_dh=float(np.mean(dh)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Angular error — for illuminant-estimation baselines
# ═══════════════════════════════════════════════════════════════════════════


def angular_error(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Recovery angular error between two 3-D illuminant vectors (degrees).

    Both inputs must be in **linear RGB** (not gamma-coded sRGB).

    Returns
    -------
    Angle in degrees ∈ [0, 180].
    """
    e = np.asarray(estimated, dtype=np.float64).ravel()
    g = np.asarray(ground_truth, dtype=np.float64).ravel()
    ne = np.linalg.norm(e)
    ng = np.linalg.norm(g)
    if ne < 1e-12 or ng < 1e-12:
        return 180.0
    cos_theta = np.clip(np.dot(e, g) / (ne * ng), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def reproduction_angular_error(
    estimated_illum: np.ndarray,
    gt_illum: np.ndarray,
) -> float:
    """Reproduction angular error (Finlayson & Zakizadeh 2015).

    Measures the angle between the RGB of a white surface when
    corrected by the estimated vs. ground-truth illuminant.

    Both illuminant vectors in **linear RGB**.

    Returns
    -------
    Angle in degrees ∈ [0, 180].
    """
    e = np.asarray(estimated_illum, dtype=np.float64).ravel()
    g = np.asarray(gt_illum, dtype=np.float64).ravel()
    safe_e = np.where(np.abs(e) > 1e-12, e, 1e-12)
    # Reproduction: white surface under each illuminant, then corrected
    repro_est = g / safe_e  # what you get when you correct with estimated
    repro_gt = np.ones(3)   # perfect correction = [1, 1, 1]
    return angular_error(repro_est, repro_gt)


def angular_error_stats(errors: Sequence[float]) -> dict[str, float]:
    """Gijsenij-2011 aggregation for angular errors.

    Returns dict with mean, median, trimean, best25, worst25, p95.
    """
    e = np.asarray(errors, dtype=np.float64)
    q1 = float(np.percentile(e, 25))
    med = float(np.median(e))
    q3 = float(np.percentile(e, 75))
    return {
        "mean": float(np.mean(e)),
        "median": med,
        "trimean": (q1 + 2.0 * med + q3) / 4.0,
        "best25": float(np.mean(e[e <= q1])) if np.any(e <= q1) else float(np.min(e)),
        "worst25": float(np.mean(e[e >= q3])) if np.any(e >= q3) else float(np.max(e)),
        "p95": float(np.percentile(e, 95)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. SSIM / PSNR — global image similarity (secondary)
# ═══════════════════════════════════════════════════════════════════════════


def compute_ssim(
    image_a_srgb: np.ndarray,
    image_b_srgb: np.ndarray,
    *,
    channel_axis: int = 2,
) -> float:
    """Structural Similarity Index between two sRGB images [0, 1] float32.
    Returns SSIM ∈ [−1, 1]; 1 = identical.
    """
    return float(
        structural_similarity(
            image_a_srgb, image_b_srgb,
            data_range=1.0,
            channel_axis=channel_axis,
        )
    )


def compute_psnr(image_a_srgb: np.ndarray, image_b_srgb: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between two sRGB images [0, 1] float32.

    Returns PSNR in dB.  Higher = more similar.  Returns inf if identical.
    """
    return float(peak_signal_noise_ratio(image_a_srgb, image_b_srgb, data_range=1.0))


# ═══════════════════════════════════════════════════════════════════════════
# 5. Colorfulness — perceptual saturation diagnostic
# ═══════════════════════════════════════════════════════════════════════════


def compute_colorfulness(image_srgb: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colorfulness metric.

    Input: sRGB float32 [0, 1], shape (H, W, 3).

    Returns M ≥ 0.  Higher = more colourful.
    """
    r = image_srgb[:, :, 0].ravel().astype(np.float64)
    g = image_srgb[:, :, 1].ravel().astype(np.float64)
    b = image_srgb[:, :, 2].ravel().astype(np.float64)

    rg = r - g
    yb = 0.5 * (r + g) - b

    sigma_rg = float(np.std(rg))
    sigma_yb = float(np.std(yb))
    mu_rg = float(np.mean(rg))
    mu_yb = float(np.mean(yb))

    return float(
        np.sqrt(sigma_rg**2 + sigma_yb**2)
        + 0.3 * np.sqrt(mu_rg**2 + mu_yb**2)
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. ITA and Chardon classification
# ═══════════════════════════════════════════════════════════════════════════


def compute_ita(L_star: float, b_star: float) -> float:
    """Individual Typology Angle from CIELAB L* and b*.

    ITA = arctan2(L* − 50, b*) × (180/π)
    """
    return float(np.arctan2(L_star - 50.0, b_star) * (180.0 / np.pi))


def classify_chardon(ita: float) -> tuple[str, int]:
    """Chardon 1991 category from ITA value.

    Boundary values stay in the lower category: 55° is Light, 41° is
    Intermediate, 10° is Brown, and −30° is Brown ("Dark" is < −30°).

    Returns (category_name, fitzpatrick_type).
    """
    if ita > 55.0:
        return "Very Light", 1
    if ita > 41.0:
        return "Light", 2
    if ita > 28.0:
        return "Intermediate", 3
    if ita > 10.0:
        return "Tan", 4
    if ita >= -30.0:
        return "Brown", 5
    return _CHARDON_DARKEST


# ═══════════════════════════════════════════════════════════════════════════
# 7. Cross-lighting stability — std(ITA), std(L*), std(a*), std(b*)
# ═══════════════════════════════════════════════════════════════════════════


def cross_lighting_std(values: Sequence[float]) -> CrossLightingStats:
    """Compute consistency statistics for a scalar across lighting conditions.

    Parameters
    ----------
    values : sequence of measurements (e.g. ITA from L1–L5 for one person).

    Returns
    -------
    CrossLightingStats with mean, std, range, n.
    """
    arr = np.asarray(values, dtype=np.float64)
    return CrossLightingStats(
        values=arr.tolist(),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=0)),
        range=float(np.ptp(arr)),
        n=len(arr),
    )


def ita_reduction(std_before: float, std_after: float) -> float:
    """Percentage reduction in ITA std.  Positive = improvement.

    Returns (std_before − std_after) / std_before × 100.
    Returns 0 if std_before is 0 (nothing to improve).
    """
    if std_before <= 0:
        return 0.0
    return (std_before - std_after) / std_before * 100.0


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Coefficient of Variation: std/|mean| × 100%.

    Normalises variability for fair cross-group comparison (e.g. persons
    with different baseline ITA).  Returns ``inf`` when the mean is ≈0.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) < 1e-10:
        return float("inf")
    return float(np.std(arr, ddof=1) / abs(mean) * 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Chardon category consistency
# ═══════════════════════════════════════════════════════════════════════════


def chardon_consistency(categories: Sequence[str]) -> bool:
    """Check if all Chardon categories in the group are identical.

    Parameters
    ----------
    categories : list of Chardon category names for one person+hand
                 across multiple lighting conditions.

    Returns True if all entries are the same category.
    """
    return len(set(categories)) <= 1


def chardon_consistency_rate(groups: Sequence[Sequence[str]]) -> float:
    """Fraction of groups with consistent Chardon category.

    Parameters
    ----------
    groups : list of lists, each inner list being categories for one
             person+hand group across lighting conditions.

    Returns percentage ∈ [0, 100].
    """
    if not groups:
        return 0.0
    consistent = sum(1 for g in groups if chardon_consistency(g))
    return consistent / len(groups) * 100.0


# ═══════════════════════════════════════════════════════════════════════════
# 9. Paired statistical tests
# ═══════════════════════════════════════════════════════════════════════════


def paired_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    """Wilcoxon signed-rank test on paired samples.

    H0: the distribution of a − b is symmetric around zero.
    Rejects when one method consistently outperforms the other.

    Parameters
    ----------
    a, b : paired per-image metric values (e.g. ΔE00 for method A vs B).
    alternative : "two-sided" | "less" | "greater".

    Returns
    -------
    WilcoxonResult with statistic, p_value, n, significant (p<0.05),
    and median of the paired differences (a − b).
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"paired_wilcoxon: a and b must have the same shape, "
            f"got {a_arr.shape} vs {b_arr.shape}"
        )
    # Drop pairs where either value is non-finite
    finite = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr, b_arr = a_arr[finite], b_arr[finite]
    diff = a_arr - b_arr

    # Filter zeros — Wilcoxon is undefined for zero differences
    nonzero = diff[diff != 0]
    n = len(nonzero)

    if n < 10:
        # Too few samples for a meaningful test
        return WilcoxonResult(
            statistic=float("nan"),
            p_value=1.0,
            n=n,
            significant=False,
            median_diff=float(np.median(diff)),
        )

    stat, p = sp_stats.wilcoxon(nonzero, alternative=alternative)
    return WilcoxonResult(
        statistic=float(stat),
        p_value=float(p),
        n=n,
        significant=float(p) < 0.05,
        median_diff=float(np.median(diff)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Aggregation and comparison helpers
# ═══════════════════════════════════════════════════════════════════════════


def improvement_over_baseline(baseline_val: float, method_val: float) -> float:
    """Percentage improvement (positive = method is better / lower).

    (baseline − method) / baseline × 100.
    """
    if baseline_val <= 0:
        return 0.0
    return (baseline_val - method_val) / baseline_val * 100.0


def method_comparison_matrix(
    results: dict[str, Sequence[float]],
) -> dict[tuple[str, str], float]:
    """Pairwise ΔΔ matrix between methods.

    Parameters
    ----------
    results : {method_name: [per_image_metric_values]}.
              All value lists must have the same length (same image set).

    Returns
    -------
    Dict mapping (method_a, method_b) → mean(a_values − b_values).
    Negative values mean method_a is better (lower metric).
    """
    methods = list(results.keys())
    matrix: dict[tuple[str, str], float] = {}
    for a in methods:
        for b in methods:
            if a == b:
                continue
            va = np.asarray(results[a], dtype=np.float64)
            vb = np.asarray(results[b], dtype=np.float64)
            matrix[(a, b)] = float(np.mean(va - vb))
    return matrix


def method_ranking(
    results: dict[str, Sequence[float]],
) -> list[tuple[str, float]]:
    """Rank methods by mean metric value (ascending — lower is better).

    Parameters
    ----------
    results : {method_name: [per_image_metric_values]}.

    Returns list of (method_name, mean_value) sorted ascending.
    """
    ranking = [(name, float(np.mean(vals))) for name, vals in results.items()]
    ranking.sort(key=lambda x: x[1])
    return ranking


def pairwise_wilcoxon_matrix(
    results: dict[str, Sequence[float]],
) -> dict[tuple[str, str], WilcoxonResult]:
    """Wilcoxon signed-rank test for every method pair.

    Parameters
    ----------
    results : {method_name: [per_image_metric_values]}.

    Returns dict mapping (method_a, method_b) → WilcoxonResult.
    """
    methods = list(results.keys())
    matrix: dict[tuple[str, str], WilcoxonResult] = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            w = paired_wilcoxon(results[a], results[b])
            matrix[(a, b)] = w
            # Symmetric entry with flipped sign
            matrix[(b, a)] = WilcoxonResult(
                statistic=w.statistic,
                p_value=w.p_value,
                n=w.n,
                significant=w.significant,
                median_diff=-w.median_diff,
            )
    return matrix


# ═══════════════════════════════════════════════════════════════════════════
# 11. I/O — results CSV read/write
# ═══════════════════════════════════════════════════════════════════════════


def write_results_csv(results: Sequence[ImageResult], path: str | Path) -> Path:
    """Write a list of ImageResult records to CSV.

    Creates parent directories.  Returns the resolved path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_IMAGE_RESULT_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    return out


def read_results_csv(path: str | Path) -> list[ImageResult]:
    """Read ImageResult records from a CSV written by write_results_csv."""
    out: list[ImageResult] = []
    # Build type maps from dataclass annotations
    int_fields: set[str] = set()
    float_fields: set[str] = set()
    for f in fields(ImageResult):
        t = f.type
        if t in ("int | None", "int"):
            int_fields.add(f.name)
        elif t in ("float | None", "float"):
            float_fields.add(f.name)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean: dict[str, Any] = {}
            for k, v in row.items():
                if v == "" or v == "None":
                    clean[k] = None
                elif k in int_fields:
                    clean[k] = int(float(v))  # int("3.0") fails, int(float("3.0")) works
                elif k in float_fields:
                    clean[k] = float(v)
                elif k in ("success", "skin_measured"):
                    clean[k] = v.lower() in ("true", "1", "yes")
                else:
                    clean[k] = v
            out.append(ImageResult(**clean))
    return out


def write_results_json(results: Sequence[ImageResult], path: str | Path) -> Path:
    """Write results as JSON array (for richer consumers)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 12. Convenience — populate ImageResult from pipeline outputs
# ═══════════════════════════════════════════════════════════════════════════


def fill_cc_delta_e(
    result: ImageResult,
    de_before: DeltaEStats | None,
    de_after: DeltaEStats | None,
) -> None:
    """Copy DeltaEStats into the flat ImageResult fields (mutates in place)."""
    if de_before is not None:
        result.cc_de_mean_before = de_before.mean
        result.cc_de_median_before = de_before.median
        result.cc_de_trimean_before = de_before.trimean
        result.cc_de_p95_before = de_before.p95
        result.cc_de_max_before = de_before.max
        result.cc_de_best25_before = de_before.best25
        result.cc_de_worst25_before = de_before.worst25
    if de_after is not None:
        result.cc_de_mean_after = de_after.mean
        result.cc_de_median_after = de_after.median
        result.cc_de_trimean_after = de_after.trimean
        result.cc_de_p95_after = de_after.p95
        result.cc_de_max_after = de_after.max
        result.cc_de_best25_after = de_after.best25
        result.cc_de_worst25_after = de_after.worst25


def fill_skin_tone(
    result: ImageResult,
    skin_before: Any | None,
    skin_after: Any | None,
) -> None:
    """Copy SkinToneValues into the flat ImageResult fields (mutates in place).

    Sets ``result.skin_measured = True`` when skin_before is not None, so
    downstream evaluation can filter rows that lack skin-tone data.

    Accepts any object with L_median, a_median, b_median, ITA,
    chardon_category attributes (duck-typed against SkinToneValues).
    """
    if skin_before is not None:
        result.L_median_before = skin_before.L_median
        result.a_median_before = skin_before.a_median
        result.b_median_before = skin_before.b_median
        result.ITA_before = skin_before.ITA
        result.chardon_before = skin_before.chardon_category
        result.skin_measured = True
    if skin_after is not None:
        result.L_median_after = skin_after.L_median
        result.a_median_after = skin_after.a_median
        result.b_median_after = skin_after.b_median
        result.ITA_after = skin_after.ITA
        result.chardon_after = skin_after.chardon_category


def fill_bn_delta_e(
    result: ImageResult,
    de_before: DeltaEStats | None,
    de_after: DeltaEStats | None,
) -> None:
    """Copy banknote DeltaEStats into the flat ImageResult fields."""
    if de_before is not None:
        result.bn_de_mean_before = de_before.mean
        result.bn_de_median_before = de_before.median
        result.bn_de_trimean_before = de_before.trimean
        result.bn_de_p95_before = de_before.p95
        result.bn_de_max_before = de_before.max
        result.bn_de_best25_before = de_before.best25
        result.bn_de_worst25_before = de_before.worst25
    if de_after is not None:
        result.bn_de_mean_after = de_after.mean
        result.bn_de_median_after = de_after.median
        result.bn_de_trimean_after = de_after.trimean
        result.bn_de_p95_after = de_after.p95
        result.bn_de_max_after = de_after.max
        result.bn_de_best25_after = de_after.best25
        result.bn_de_worst25_after = de_after.worst25


def fill_bn_meta(
    result: ImageResult,
    bn_result: Any,
) -> None:
    """Copy BanknoteResult detection metadata into ImageResult."""
    if bn_result is not None and bn_result.detected:
        result.denomination = bn_result.denomination
        result.bn_n_samples = bn_result.n_samples
        result.bn_confidence = bn_result.confidence
        result.bn_reproj_rmse = bn_result.reproj_rmse
