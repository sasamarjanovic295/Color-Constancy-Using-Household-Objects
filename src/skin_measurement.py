"""
skin_measurement.py — Skin colour measurement using Bencevic median+1SD method.

Adapted from Bencevic, M., Sojo, R., & Galic, I. (2025).
"Skin Color Measurement from Dermatoscopic Images."
arXiv:2504.04494  |  github.com/marinbenc/dermatoscopy_colorimetry_eval

Pipeline:
    1. Erode mask (remove boundary artifacts)
    2. Extract skin pixels → standard CIELAB (float32)
    3. Pass 1: median L*, b* + std
    4. Pass 2: discard pixels > 1 SD from median (on L* AND b*)
    5. Pass 3: re-compute median L*, a*, b* of kept pixels
    6. ITA = arctan2(L*−50, b*) × 180/π
    7. Chardon classification (6 categories, Fitzpatrick I–VI)

Corrections vs Bencevic original:
    - Standard CIELAB (float32 input → L* [0,100]) instead of OpenCV uint8
    - arctan2 instead of arctan (correct quadrant, no division by zero)
    - Boolean indexing instead of np.multiply (avoids 0 = valid L* confusion)
    - a* channel tracked alongside L* and b*

Usage::

    from src.skin_measurement import measure_skin_tone

    result = measure_skin_tone(image_srgb, mask_u8,
                               save_steps=True, output_dir=Path("out"))
    print(result.ITA, result.chardon_category)
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ERODE_SIZE: int = 5  # erosion kernel diameter (pixels)

# Chardon 1991 — ITA thresholds and categories (Ly et al. 2020: Dark < −30°)
_CHARDON_THRESHOLDS: list[tuple[float, str, int]] = [
    (55.0, "Very Light", 1),
    (41.0, "Light", 2),
    (28.0, "Intermediate", 3),
    (10.0, "Tan", 4),
    (-30.0, "Brown", 5),
]
_CHARDON_DARKEST: tuple[str, int] = ("Dark", 6)

# Visualisation
_VIZ_SWATCH_RATIO: float = 0.08
_VIZ_FILENAMES: list[str] = [
    "viz_01_mask_erosion.jpg",
    "viz_02_skin_pixels.jpg",
    "viz_03_lab_histograms.jpg",
    "viz_04_kept_rejected.jpg",
    "viz_05_color_on_skin.jpg",
    "viz_06_ita_gauge.jpg",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SkinToneValues:
    """Measured skin colour from a single image + mask pair."""

    L_median: float
    a_median: float
    b_median: float
    L_mean: float
    a_mean: float
    b_mean: float
    ITA: float
    chardon_category: str
    fitzpatrick_type: int
    n_pixels_total: int
    n_pixels_filtered: int
    filter_ratio: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_chardon(ita: float) -> tuple[str, int]:
    """Classify ITA into Chardon 1991 category + Fitzpatrick type."""
    if ita > 55.0:
        return ("Very Light", 1)
    if ita > 41.0:
        return ("Light", 2)
    if ita > 28.0:
        return ("Intermediate", 3)
    if ita > 10.0:
        return ("Tan", 4)
    if ita >= -30.0:
        return ("Brown", 5)
    return _CHARDON_DARKEST


def _lab_to_bgr_swatch(L: float, a: float, b: float) -> np.ndarray:
    """Convert a single L*a*b* colour to a BGR uint8 pixel."""
    pixel_lab = np.array([[[L, a, b]]], dtype=np.float32)
    pixel_bgr = cv.cvtColor(pixel_lab, cv.COLOR_Lab2BGR)
    pixel_bgr = np.clip(pixel_bgr * 255, 0, 255).astype(np.uint8)
    return pixel_bgr[0, 0]


def _skin_centroid(mask: np.ndarray) -> tuple[int, int]:
    """Find the deepest point inside the mask via distance transform."""
    dt = cv.distanceTransform(mask, cv.DIST_L2, 5)
    _, _, _, max_loc = cv.minMaxLoc(dt)
    return max_loc  # (x, y)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _viz_mask_erosion(
    image_bgr: np.ndarray,
    mask_orig_u8: np.ndarray,
    mask_eroded_u8: np.ndarray,
) -> np.ndarray:
    """VIZ 01: original mask contour (blue), eroded overlay (green), diff (red)."""
    vis = image_bgr.copy()
    diff = cv.subtract(mask_orig_u8, mask_eroded_u8)

    # Eroded mask overlay in green
    overlay = vis.copy()
    overlay[mask_eroded_u8 > 0] = (0, 200, 0)
    vis = cv.addWeighted(overlay, 0.25, vis, 0.75, 0)

    # Original mask contour in blue
    contours, _ = cv.findContours(
        mask_orig_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    cv.drawContours(vis, contours, -1, (255, 100, 0), 2)

    # Diff (eroded border) in red
    vis[diff > 0] = (0, 0, 200)
    return vis


def _viz_skin_pixels(
    image_bgr: np.ndarray,
    mask_eroded_u8: np.ndarray,
) -> np.ndarray:
    """VIZ 02: skin pixels on black background."""
    out = np.zeros_like(image_bgr)
    out[mask_eroded_u8 > 0] = image_bgr[mask_eroded_u8 > 0]
    return out


def _viz_lab_histograms(
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    L_med: float,
    b_med: float,
    L_std: float,
    b_std: float,
    keep_mask: np.ndarray,
) -> np.ndarray:
    """VIZ 03: L*/a*/b* histograms with median ±1SD markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), tight_layout=True)
    channels = [
        ("L*", L, L_med, L_std, (0.2, 0.6, 0.2)),
        ("a*", a, None, None, (0.8, 0.2, 0.2)),
        ("b*", b, b_med, b_std, (0.2, 0.2, 0.8)),
    ]
    for ax, (name, vals, med, std, color) in zip(axes, channels):
        ax.hist(vals, bins=100, color=color, alpha=0.4, label="all")
        ax.hist(vals[keep_mask], bins=100, color=color, alpha=0.8, label="kept")
        if med is not None:
            ax.axvline(med, color="black", linewidth=2, label=f"median={med:.1f}")
            ax.axvline(med - std, color="black", linewidth=2, linestyle="--")
            ax.axvline(
                med + std,
                color="black",
                linewidth=2,
                linestyle="--",
                label=f"±1SD={std:.1f}",
            )
        ax.set_title(name, fontsize=14)
        ax.legend(fontsize=8)

    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return cv.cvtColor(buf[:, :, 1:], cv.COLOR_RGB2BGR)


def _viz_kept_rejected(
    image_bgr: np.ndarray,
    mask_eroded_u8: np.ndarray,
    keep_map_u8: np.ndarray,
    reject_map_u8: np.ndarray,
) -> np.ndarray:
    """VIZ 04: green=kept, red=rejected pixels on original image."""
    vis = image_bgr.copy()
    overlay = vis.copy()
    overlay[keep_map_u8 > 0] = (0, 200, 0)
    overlay[reject_map_u8 > 0] = (0, 0, 200)
    vis = cv.addWeighted(overlay, 0.45, vis, 0.55, 0)
    return vis


def _viz_color_on_skin(
    image_bgr: np.ndarray,
    mask_eroded_u8: np.ndarray,
    keep_map_u8: np.ndarray,
    swatch_bgr: np.ndarray,
) -> np.ndarray:
    """VIZ 05: side-by-side — median colour swatch placed on densest skin area."""
    h, w = image_bgr.shape[:2]
    swatch_side = max(int(min(h, w) * _VIZ_SWATCH_RATIO), 40)
    half = swatch_side // 2

    # Find densest skin point
    ys, xs = np.where(keep_map_u8 > 0)
    if len(ys) == 0:
        cx, cy = w // 2, h // 2
    else:
        dt = cv.distanceTransform(keep_map_u8, cv.DIST_L2, 5)
        search_r = min(h, w) // 2
        sy1 = max(0, np.median(ys).astype(int) - search_r)
        sy2 = min(h, np.median(ys).astype(int) + search_r)
        sx1 = max(0, np.median(xs).astype(int) - search_r)
        sx2 = min(w, np.median(xs).astype(int) + search_r)
        roi_dt = dt[sy1:sy2, sx1:sx2]
        _, _, _, local_max = cv.minMaxLoc(roi_dt)
        cx = sx1 + local_max[0]
        cy = sy1 + local_max[1]

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + swatch_side)
    y2 = min(h, y1 + swatch_side)

    # Left = original with skin overlay, Right = swatch overlay
    left = image_bgr.copy()
    left[keep_map_u8 > 0] = left[keep_map_u8 > 0]
    cv.rectangle(left, (x1, y1), (x2, y2), (0, 220, 0), 3)

    right = image_bgr.copy()
    right[y1:y2, x1:x2] = swatch_bgr
    cv.rectangle(right, (x1, y1), (x2, y2), (0, 220, 0), 3)

    return np.hstack([left, right])


def _viz_ita_gauge(ita: float, chardon: str) -> np.ndarray:
    """VIZ 06: horizontal Chardon scale bar with ITA marker."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = [
        ("Dark", -55, -30, "#3B2219"),
        ("Brown", -30, 10, "#6B4226"),
        ("Tan", 10, 28, "#A67B5B"),
        ("Intermediate", 28, 41, "#C8A882"),
        ("Light", 41, 55, "#E8D4B8"),
        ("Very Light", 55, 80, "#F5E6D3"),
    ]

    fig, ax = plt.subplots(figsize=(12, 2.5))
    for name, lo, hi, color in categories:
        ax.barh(
            0,
            hi - lo,
            left=lo,
            height=0.6,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.text(
            (lo + hi) / 2,
            0,
            name,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white" if lo < 10 else "black",
        )

    ita_clamped = np.clip(ita, -55, 80)
    ax.plot(
        ita_clamped,
        0,
        marker="v",
        markersize=18,
        color="red",
        markeredgecolor="black",
        markeredgewidth=1.5,
        zorder=10,
        clip_on=False,
    )
    ax.text(
        ita_clamped,
        -0.55,
        f"ITA={ita:.1f}°\n{chardon}",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlim(-60, 85)
    ax.set_ylim(-1.2, 0.6)
    ax.set_xlabel("ITA (°)", fontsize=11)
    ax.yaxis.set_visible(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return cv.cvtColor(buf[:, :, 1:], cv.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Main measurement function
# ---------------------------------------------------------------------------


def measure_skin_tone(
    image_srgb: np.ndarray,
    mask_u8: np.ndarray,
    erode_size: int = _ERODE_SIZE,
    show_steps: bool = False,
    save_steps: bool = False,
    log: bool = False,
    output_dir: Path | None = None,
) -> SkinToneValues | None:
    """Measure skin tone from *image_srgb* using binary *mask_u8*.

    Parameters
    ----------
    image_srgb : (H, W, 3) float32 sRGB [0, 1].
    mask_u8 : (H, W) uint8 binary mask (255 = skin).
    erode_size : kernel size for boundary erosion.
    show_steps : display intermediate visualisations.
    save_steps : save visualisations to *output_dir*.
    log : print progress to stdout.
    output_dir : where to write ``viz_*.jpg`` files.

    Returns
    -------
    SkinToneValues | None
        ``None`` when fewer than 100 skin pixels survive erosion.
    """
    image_h, image_w = image_srgb.shape[:2]

    # ── 1. Erode mask ─────────────────────────────────────────────────
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erode_size, erode_size))
    mask_eroded_u8 = cv.erode(mask_u8, k)

    n_orig = int(np.count_nonzero(mask_u8))
    n_eroded = int(np.count_nonzero(mask_eroded_u8))

    if log:
        print(f"  skin: mask {n_orig} → eroded {n_eroded} px")

    if n_eroded < 100:
        if log:
            print("  skin: too few pixels after erosion — skipping")
        return None

    # ── 2. Extract skin pixels → Lab ──────────────────────────────────
    # OpenCV COLOR_RGB2Lab handles sRGB gamma decoding for float32 input.
    image_lab = cv.cvtColor(image_srgb, cv.COLOR_RGB2Lab)  # float32 → L*[0,100]
    skin_lab = image_lab[mask_eroded_u8 > 0]  # (N, 3)

    L = skin_lab[:, 0]  # L* [0, 100]
    a = skin_lab[:, 1]  # a*
    b = skin_lab[:, 2]  # b*
    n_total = len(L)

    # ── 3. Pass 1: median + std ───────────────────────────────────────
    L_med = float(np.median(L))
    b_med = float(np.median(b))
    L_std = float(np.std(L))
    b_std = float(np.std(b))

    # ── 4. Pass 2: keep within 1 SD of median on L* AND b* ───────────
    keep = (np.abs(L - L_med) <= L_std) & (np.abs(b - b_med) <= b_std)
    L_kept = L[keep]
    a_kept = a[keep]
    b_kept = b[keep]
    n_filtered = int(np.count_nonzero(keep))
    ratio = n_filtered / n_total if n_total > 0 else 0.0

    if n_filtered < 10:
        if log:
            print("  skin: too few pixels after 1SD filter — skipping")
        return None

    # ── 5. Pass 3: final median on kept pixels ────────────────────────
    L_final = float(np.median(L_kept))
    a_final = float(np.median(a_kept))
    b_final = float(np.median(b_kept))
    L_mean_f = float(np.mean(L_kept))
    a_mean_f = float(np.mean(a_kept))
    b_mean_f = float(np.mean(b_kept))

    # ── 6. ITA ────────────────────────────────────────────────────────
    ita = float(np.arctan2(L_final - 50.0, b_final) * (180.0 / np.pi))

    # ── 7. Chardon classification ─────────────────────────────────────
    chardon, fp = _classify_chardon(ita)

    if log:
        print(
            f"  skin: L*={L_final:.1f} a*={a_final:.1f} b*={b_final:.1f}"
            f"  ITA={ita:.1f}° → {chardon} (FP {fp})"
            f"  [{n_filtered}/{n_total} px, {ratio:.0%}]"
        )

    result = SkinToneValues(
        L_median=L_final,
        a_median=a_final,
        b_median=b_final,
        L_mean=L_mean_f,
        a_mean=a_mean_f,
        b_mean=b_mean_f,
        ITA=ita,
        chardon_category=chardon,
        fitzpatrick_type=fp,
        n_pixels_total=n_total,
        n_pixels_filtered=n_filtered,
        filter_ratio=ratio,
    )

    # ── Visualisations ────────────────────────────────────────────────
    if show_steps or save_steps:
        image_bgr = cv.cvtColor(
            (np.clip(image_srgb, 0, 1) * 255).astype(np.uint8),
            cv.COLOR_RGB2BGR,
        )
        swatch_bgr = _lab_to_bgr_swatch(L_final, a_final, b_final)

        # Build keep/reject spatial maps
        keep_map_u8 = np.zeros((image_h, image_w), dtype=np.uint8)
        reject_map_u8 = np.zeros((image_h, image_w), dtype=np.uint8)
        skin_ys, skin_xs = np.where(mask_eroded_u8 > 0)
        keep_map_u8[skin_ys[keep], skin_xs[keep]] = 255
        reject_map_u8[skin_ys[~keep], skin_xs[~keep]] = 255

        panels = [
            _viz_mask_erosion(image_bgr, mask_u8, mask_eroded_u8),
            _viz_skin_pixels(image_bgr, mask_eroded_u8),
            _viz_lab_histograms(L, a, b, L_med, b_med, L_std, b_std, keep),
            _viz_kept_rejected(image_bgr, mask_eroded_u8, keep_map_u8, reject_map_u8),
            _viz_color_on_skin(image_bgr, mask_eroded_u8, keep_map_u8, swatch_bgr),
            _viz_ita_gauge(ita, chardon),
        ]

        if show_steps:
            titles = [
                "01 Mask erosion",
                "02 Skin pixels",
                "03 Lab histograms",
                "04 Kept/rejected",
                "05 Colour on skin",
                "06 ITA gauge",
            ]
            for title, img in zip(titles, panels):
                cv.imshow(title, img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        if save_steps and output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for fname, img in zip(_VIZ_FILENAMES, panels):
                cv.imwrite(str(out / fname), img)

    return result
