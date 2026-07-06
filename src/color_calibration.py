"""
color_calibration.py — Unified colour calibration methods.

Consolidates and validates all calibration methods for the thesis:
  - Reference-based: linear 3×3, affine 3×4, poly2, poly3 (Cheung 2004)
  - Baseline (no reference): Gray World, Shades of Gray

All reference-based methods operate in **XYZ colour space** (D65 white point).
Input images must be converted from sRGB → XYZ before calibration, and
converted back XYZ → sRGB afterwards. Utility functions are provided.

Mathematical foundation:
    Given N measured swatch colours M (N×3) and N reference colours R (N×3),
    find a mapping f such that f(M) ≈ R, then apply f to the entire image.

    - Linear:  R ≈ M × Aᵀ           (A is 3×3,  9 params)
    - Affine:  R ≈ [M, 1] × Aᵀ      (A is 3×4, 12 params)
    - Poly2:   R ≈ Φ₂(M) × W        (W is 10×3, 30 params)
    - Poly3:   R ≈ Φ₃(M) × W        (W is 20×3, 60 params)

Ridge regularisation (λ=1e-4) is applied to affine and polynomial methods
to prevent overfitting.  The bias term is NOT regularised (R[-1,-1]=0).

Baselines estimate the scene illuminant from pixel statistics and apply
a diagonal correction — no reference colours needed.

References:
    Cheung et al. (2004). "A comparative study of the characterisation
        of colour cameras." Coloration Technology, 120(1), 19–25.
    van de Weijer et al. (2007). "Edge-Based Color Constancy."
        IEEE Trans. Image Processing.
    Mentor emails (Benčević, M.) — polynomial expansion, sRGB→XYZ,
        patch-based sampling, ridge regularisation.

Usage::

    from src.color_calibration import (
        srgb_to_xyz, xyz_to_srgb, adapt_d50_to_d65,
        calibrate, apply_calibration, compute_delta_e00,
        METHODS,
    )

    image_xyz_d65 = srgb_to_xyz(image_srgb_f32)
    ref_xyz_d65 = adapt_d50_to_d65(ref_xyz_d50)
    params = calibrate("affine", measured_xyz_d65, ref_xyz_d65)
    corrected_xyz_d65 = apply_calibration(image_xyz_d65, params)
    corrected_srgb = xyz_to_srgb(corrected_xyz_d65)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import colour
import numpy as np
from skimage.color import rgb2xyz, xyz2rgb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_D50_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
_D65_XY = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
_D50_XYZ = colour.xy_to_XYZ(_D50_XY)
_D65_XYZ = colour.xy_to_XYZ(_D65_XY)

# Available calibration method names
METHODS: list[str] = [
    "linear",
    "affine",
    "poly2",
    "poly3_cheung",
    "gray_world",
    "shades_of_gray",
]

_DEFAULT_RIDGE_LAMBDA: float = 1e-4
_DEFAULT_SHADES_P: float = 6.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CalibrationParams:
    """Parameters produced by :func:`calibrate`, consumed by :func:`apply_calibration`."""

    method: str
    matrix: np.ndarray          # shape depends on method
    basis: str | None = None    # for poly methods
    ridge_lambda: float = 0.0
    illuminant: np.ndarray | None = None  # for baseline methods
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Colour-space utilities
# ---------------------------------------------------------------------------


def srgb_to_xyz(image_srgb: np.ndarray) -> np.ndarray:
    """Convert float32 sRGB [0,1] image to CIE XYZ (D65).

    Applies inverse sRGB gamma (linearisation) then the sRGB→XYZ matrix.
    Uses ``skimage.color.rgb2xyz`` which handles the full sRGB transfer
    function (linear segment + gamma 2.4).
    """
    return rgb2xyz(image_srgb).astype(np.float32)


def srgb_to_xyz_samples(samples_srgb: np.ndarray) -> np.ndarray:
    """Convert (N, 3) sRGB gamma float32 to CIE XYZ-D65 float64.

    Unlike :func:`srgb_to_xyz` which expects (H, W, 3) images, this
    handles flat (N, 3) sample arrays — e.g. banknote grid samples.

    Internally reshapes to (N, 1, 3) to satisfy ``rgb2xyz``'s image
    shape requirement, then squeezes back.
    """
    fake_image = np.asarray(samples_srgb, dtype=np.float32).reshape(-1, 1, 3)
    xyz = rgb2xyz(fake_image)
    return xyz.reshape(-1, 3).astype(np.float64)


def xyz_to_srgb(image_xyz_d65: np.ndarray) -> np.ndarray:
    """Convert CIE XYZ (D65) image to sRGB [0,1] float32.

    Applies XYZ→linear-sRGB matrix then forward sRGB gamma.
    """
    return np.clip(xyz2rgb(image_xyz_d65), 0.0, 1.0).astype(np.float32)


def adapt_d50_to_d65(xyz_d50: np.ndarray) -> np.ndarray:
    """Chromatically adapt XYZ values from D50 to D65 (Bradford).

    Used to bring SCK300 reference values (stored as D50) into the same
    white-point space as the camera image (sRGB = D65).
    """
    adapted = colour.adaptation.chromatic_adaptation_VonKries(
        xyz_d50, _D50_XYZ, _D65_XYZ, transform="Bradford",
    )
    return np.asarray(adapted, dtype=np.float64)


def adapt_d65_to_d50(xyz_d65: np.ndarray) -> np.ndarray:
    """Chromatically adapt XYZ values from D65 to D50 (Bradford)."""
    adapted = colour.adaptation.chromatic_adaptation_VonKries(
        xyz_d65, _D65_XYZ, _D50_XYZ, transform="Bradford",
    )
    return np.asarray(adapted, dtype=np.float64)


def compute_delta_e00(
    colors_a_xyz_d65: np.ndarray,
    colors_b_xyz_d65: np.ndarray,
) -> np.ndarray:
    """Compute per-sample CIEDE2000 between two sets of XYZ-D65 colours.

    Both inputs are adapted to D50 internally before Lab conversion,
    since CIEDE2000 is defined relative to D50.

    Returns:
        1-D array of ΔE00 values, shape (N,).
    """
    a_xyz_d50 = adapt_d65_to_d50(colors_a_xyz_d65)
    b_xyz_d50 = adapt_d65_to_d50(colors_b_xyz_d65)
    lab_a_d50 = colour.XYZ_to_Lab(np.asarray(a_xyz_d50, dtype=np.float64), illuminant=_D50_XY)
    lab_b_d50 = colour.XYZ_to_Lab(np.asarray(b_xyz_d50, dtype=np.float64), illuminant=_D50_XY)
    de = colour.delta_E(
        np.asarray(lab_a_d50, dtype=np.float64),
        np.asarray(lab_b_d50, dtype=np.float64),
        method="CIE 2000",
    )
    return np.asarray(de, dtype=np.float64).ravel()


def delta_e_stats(de: np.ndarray) -> dict:
    """Aggregate a ΔE array into mean / median / p95 / max."""
    return {
        "mean": float(np.mean(de)),
        "median": float(np.median(de)),
        "p95": float(np.percentile(de, 95)),
        "max": float(np.max(de)),
    }


# ---------------------------------------------------------------------------
# Design matrices (Cheung 2004)
# ---------------------------------------------------------------------------


def _design_matrix(colors_xyz_d65: np.ndarray, basis: str) -> np.ndarray:
    """Build polynomial design matrix Φ(X).

    Cheung, V. et al. (2004) polynomial terms:
        linear:       [x, y, z]                              →  3 features
        affine:       [x, y, z, 1]                           →  4 features
        poly2:        [x, y, z, x², y², z², xy, xz, yz, 1]  → 10 features
        poly3_cheung: poly2 + [x³, y³, z³, x²y, x²z, xy²,
                               xz², y²z, yz², xyz]           → 20 features
    """
    X = colors_xyz_d65.astype(np.float64)
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]

    if basis == "linear":
        return X
    if basis == "affine":
        return np.hstack([x, y, z, np.ones_like(x)])
    if basis == "poly2":
        return np.hstack([
            x, y, z,
            x*x, y*y, z*z,
            x*y, x*z, y*z,
            np.ones_like(x),
        ])
    if basis == "poly3_cheung":
        x2, y2, z2 = x*x, y*y, z*z
        return np.hstack([
            x, y, z,
            x2, y2, z2,
            x*y, x*z, y*z,
            x2*x, y2*y, z2*z,
            x2*y, x2*z, x*y2, x*z2, y2*z, y*z2,
            x*y*z,
            np.ones_like(x),
        ])
    raise ValueError(f"Unknown basis: {basis!r}")


# ---------------------------------------------------------------------------
# Reference-based calibration methods
# ---------------------------------------------------------------------------


def _estimate_linear(
    measured_xyz_d65: np.ndarray, reference_xyz_d65: np.ndarray
) -> np.ndarray:
    """Least-squares 3×3 matrix: reference ≈ measured × Mᵀ. Returns M (3,3).

    No regularisation — well-determined for N ≥ 3.
    No bias term — pure linear mapping through origin.
    """
    M_T, *_ = np.linalg.lstsq(
        measured_xyz_d65.astype(np.float64), reference_xyz_d65.astype(np.float64), rcond=None,
    )
    return M_T.T


def _apply_linear(pixels_xyz_d65: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 3×3 matrix to (N, 3) array."""
    return (pixels_xyz_d65.astype(np.float64) @ M.T)


def _estimate_poly(
    measured_xyz_d65: np.ndarray,
    reference_xyz_d65: np.ndarray,
    basis: str,
    ridge_lambda: float,
) -> np.ndarray:
    """Ridge regression for polynomial mapping: reference ≈ Φ(measured) × W.

    Returns W of shape (F, 3) where F = number of features in basis.
    Ridge regularisation with λ on all features except the bias (last column).
    """
    Phi = _design_matrix(measured_xyz_d65, basis)
    Y = reference_xyz_d65.astype(np.float64)
    F = Phi.shape[1]
    R = np.eye(F, dtype=np.float64) * ridge_lambda
    # Do not regularise the bias term (last row/col)
    if basis in ("affine", "poly2", "poly3_cheung"):
        R[-1, -1] = 0.0
    W = np.linalg.solve(Phi.T @ Phi + R, Phi.T @ Y)
    return W


def _apply_poly(pixels_xyz_d65: np.ndarray, W: np.ndarray, basis: str) -> np.ndarray:
    """Apply polynomial mapping W to (N, 3) array."""
    Phi = _design_matrix(pixels_xyz_d65, basis)
    return Phi @ W


# ---------------------------------------------------------------------------
# Baseline methods (no reference needed)
# ---------------------------------------------------------------------------


def _estimate_illuminant_gray_world(image_xyz_d65: np.ndarray) -> np.ndarray:
    """Gray World: the average colour of the scene is gray (achromatic).

    Returns the raw mean XYZ — NOT L2-normalised — so diagonal correction
    can scale it to the D65 white point directly.
    """
    illum = image_xyz_d65.reshape(-1, 3).mean(axis=0).astype(np.float64)
    return illum


def _estimate_illuminant_shades_of_gray(
    image_xyz_d65: np.ndarray, p: float = 6.0,
) -> np.ndarray:
    """Shades of Gray: Minkowski-norm illuminant estimation.

    Returns the raw Minkowski-p mean — NOT L2-normalised.
    """
    pixels_xyz_d65 = image_xyz_d65.reshape(-1, 3).astype(np.float64)
    mask = pixels_xyz_d65.sum(axis=1) > 1e-6
    pixels_xyz_d65 = pixels_xyz_d65[mask]
    if len(pixels_xyz_d65) == 0:
        return _D65_XYZ.copy()

    illum = np.power(np.mean(np.power(np.abs(pixels_xyz_d65), p), axis=0), 1.0 / p)
    return illum.astype(np.float64)


def _apply_illuminant_correction(
    pixels_xyz_d65: np.ndarray, illuminant: np.ndarray,
) -> np.ndarray:
    """Diagonal correction: scale each XYZ channel so the estimated illuminant
    maps to the D65 white point.

    Previous implementation normalised to equal-energy E (X=Y=Z) via sqrt(3),
    which introduces a tint when converting back to sRGB.  Correct target is
    the D65 white point [0.9505, 1.0, 1.0891] because sRGB is defined under D65.
    """
    # Target: D65 white in XYZ (Y-normalised)
    target = _D65_XYZ  # [0.9505, 1.0, 1.0891]
    # Scale illuminant so it maps to target
    safe = np.where(np.abs(illuminant) > 1e-10, illuminant, 1.0)
    gain = target / safe
    return pixels_xyz_d65.astype(np.float64) * gain


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------


def calibrate(
    method: str,
    measured_xyz_d65: np.ndarray | None = None,
    reference_xyz_d65: np.ndarray | None = None,
    image_xyz_d65: np.ndarray | None = None,
    ridge_lambda: float = _DEFAULT_RIDGE_LAMBDA,
    shades_p: float = _DEFAULT_SHADES_P,
) -> CalibrationParams:
    """Estimate calibration parameters.

    For reference-based methods (linear, affine, poly2, poly3_cheung):
        ``measured_xyz_d65`` and ``reference_xyz_d65`` are required — (N, 3) XYZ-D65 arrays.

    For baseline methods (gray_world, shades_of_gray):
        ``image_xyz_d65`` is required — the full image in XYZ-D65.

    Args:
        method: One of :data:`METHODS`.
        measured_xyz_d65: (N, 3) measured swatch colours in XYZ-D65.
        reference_xyz_d65: (N, 3) reference swatch colours in XYZ-D65.
        image_xyz_d65: Full image in XYZ-D65 (for baselines).
        ridge_lambda: Ridge parameter for affine/poly (default 1e-4).
        shades_p: Minkowski norm for Shades of Gray (default 6).

    Returns:
        :class:`CalibrationParams` ready for :func:`apply_calibration`.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {METHODS}")

    if method == "linear":
        M = _estimate_linear(measured_xyz_d65, reference_xyz_d65)
        return CalibrationParams(method="linear", matrix=M)

    if method == "affine":
        W = _estimate_poly(measured_xyz_d65, reference_xyz_d65, "affine", ridge_lambda)
        return CalibrationParams(
            method="affine", matrix=W, basis="affine",
            ridge_lambda=ridge_lambda,
        )

    if method == "poly2":
        W = _estimate_poly(measured_xyz_d65, reference_xyz_d65, "poly2", ridge_lambda)
        return CalibrationParams(
            method="poly2", matrix=W, basis="poly2",
            ridge_lambda=ridge_lambda,
        )

    if method == "poly3_cheung":
        W = _estimate_poly(measured_xyz_d65, reference_xyz_d65, "poly3_cheung", ridge_lambda)
        return CalibrationParams(
            method="poly3_cheung", matrix=W, basis="poly3_cheung",
            ridge_lambda=ridge_lambda,
        )

    if method == "gray_world":
        illum = _estimate_illuminant_gray_world(image_xyz_d65)
        # Stored matrix approximates the gain for inspection; actual apply
        # uses the illuminant vector via _apply_illuminant_correction.
        gain = _D65_XYZ / np.where(np.abs(illum) > 1e-10, illum, 1.0)
        return CalibrationParams(
            method="gray_world", matrix=np.diag(gain),
            illuminant=illum,
        )

    if method == "shades_of_gray":
        illum = _estimate_illuminant_shades_of_gray(image_xyz_d65, p=shades_p)
        gain = _D65_XYZ / np.where(np.abs(illum) > 1e-10, illum, 1.0)
        return CalibrationParams(
            method="shades_of_gray",
            matrix=np.diag(gain),
            illuminant=illum,
            extra={"p": shades_p},
        )

    raise ValueError(f"Unhandled method: {method!r}")


def apply_calibration(
    image_xyz_d65: np.ndarray,
    params: CalibrationParams,
) -> np.ndarray:
    """Apply calibration to an XYZ-D65 image.

    Args:
        image_xyz_d65: (H, W, 3) float32 XYZ-D65 image.
        params: Output of :func:`calibrate`.

    Returns:
        Corrected (H, W, 3) float32 XYZ-D65 image.  Negative values are
        clamped to zero (physically invalid) but no upper clip is applied
        because valid XYZ-D65 values exceed 1.0 (e.g. sRGB white Z ≈ 1.089).
        Gamut clipping belongs in the subsequent ``xyz_to_srgb()`` call.
    """
    H, W, _ = image_xyz_d65.shape
    pixels_xyz_d65 = image_xyz_d65.reshape(-1, 3)

    if params.method == "linear":
        corrected_xyz_d65 = _apply_linear(pixels_xyz_d65, params.matrix)
    elif params.method in ("affine", "poly2", "poly3_cheung"):
        corrected_xyz_d65 = _apply_poly(pixels_xyz_d65, params.matrix, params.basis)
    elif params.method in ("gray_world", "shades_of_gray"):
        corrected_xyz_d65 = _apply_illuminant_correction(pixels_xyz_d65, params.illuminant)
    else:
        raise ValueError(f"Unknown method in params: {params.method!r}")

    # Clamp negatives only — valid D65 XYZ can exceed 1.0 (white Z≈1.089).
    # sRGB gamut clipping is done by xyz_to_srgb().
    return np.clip(corrected_xyz_d65, 0.0, None).reshape(H, W, 3).astype(np.float32)


def correct_swatches(
    measured_xyz_d65: np.ndarray,
    params: CalibrationParams,
) -> np.ndarray:
    """Apply calibration to swatch colours (N, 3) — for ΔE00 computation.

    Only clamps negatives; no upper bound because valid XYZ-D65 exceeds 1.0.
    """
    if params.method == "linear":
        out = _apply_linear(measured_xyz_d65, params.matrix)
    elif params.method in ("affine", "poly2", "poly3_cheung"):
        out = _apply_poly(measured_xyz_d65, params.matrix, params.basis)
    elif params.method in ("gray_world", "shades_of_gray"):
        out = _apply_illuminant_correction(measured_xyz_d65, params.illuminant)
    else:
        raise ValueError(f"Unknown method: {params.method!r}")
    return np.clip(out, 0.0, None)
