"""
Banknote color correction with geometric alignment, robust sampling,
and model selection (affine vs polynomial). Works in XYZ (linear) space.

Pipeline:
1) Load captured & reference images (sRGB) and convert to XYZ.
2) Find the best banknote orientation by color agreement; warp captured to reference.
3) Sample colors (points, grid, grid-inner, swatch-aware).
4) Fit color-correction models (affine 3×4, polynomial orders per Cheung 2004).
5) Apply models to the full captured image and evaluate ΔE00 on the warped domain.
6) Save all outputs and report the best-performing configuration.
"""

from pathlib import Path
import json
import cv2 as cv
import numpy as np
from skimage.color import rgb2xyz, xyz2rgb, xyz2lab, deltaE_ciede2000
import csv
import math
import matplotlib.pyplot as plt

# ------------------------- Configuration ------------------------- #

data_root = Path(__file__).parent.parent / "data"
cap_id = "162962321"
ref_id = "egp_100"

paths = {
    "captured_ann": data_root / "raw" / "annotations" / f"{cap_id}.json",
    "reference_ann": data_root / "ref" / "annotations" / f"{ref_id}.json",
    "captured_img": data_root / "raw" / "images" / f"{cap_id}.jpg",
    "reference_img": data_root / "ref" / "images" / f"{ref_id}.jpg",
    "output_dir": data_root / "corrected" / "images",
}
paths["output_dir"].mkdir(parents=True, exist_ok=True)

# ------------------------- I/O & Color Utilities ------------------------- #


def load_json(path: Path) -> dict:
    """Load a JSON file from disk."""
    with open(path, "r") as f:
        return json.load(f)


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as float32 RGB in [0,1]."""
    img = cv.imread(str(path)).astype(np.float32) / 255.0
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def to_xyz(img_rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB image to XYZ (float32, [0,1])."""
    return rgb2xyz(img_rgb).astype(np.float32)


def to_srgb(img_xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ image to sRGB (float32, [0,1], clipped)."""
    srgb = xyz2rgb(img_xyz)
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def save_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """Save float32/uint8 RGB image to disk (writes BGR JPEG/PNG)."""
    out = (
        (image_rgb * 255).astype(np.uint8)
        if image_rgb.dtype == np.float32
        else image_rgb
    )
    bgr = cv.cvtColor(out, cv.COLOR_RGB2BGR)
    cv.imwrite(str(path), bgr)


# ------------------------- Geometry & Warping ------------------------- #


def get_bbox_and_points(ann: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the banknote quadrilateral ('bbox') and ordered keypoints ('pt1'..).
    Returns:
        bbox: (4,2) float32 or None if not present
        points: (N,2) float32 ordered by pt index
    """
    bbox_list = [s["points"] for s in ann["shapes"] if s["label"] == "bbox"]
    bbox = np.array(bbox_list[0], dtype=np.float32) if bbox_list else None
    pts = {
        s["label"]: s["points"][0] for s in ann["shapes"] if s["label"].startswith("pt")
    }
    ordered = [pts[f"pt{i}"] for i in range(1, len(pts) + 1)]
    return bbox, np.array(ordered, dtype=np.float32)


def warp_image(
    image: np.ndarray, bbox: np.ndarray, ref_shape_hw: tuple[int, int]
) -> np.ndarray:
    """
    Perspective-warp 'image' from 'bbox' to a full rectangle of size ref_shape_hw.
    ref_shape_hw: (H, W)
    """
    h, w = ref_shape_hw
    dst_rect = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    Hm = cv.getPerspectiveTransform(bbox, dst_rect)
    return cv.warpPerspective(image, Hm, (w, h), flags=cv.INTER_LINEAR)


def sample_colors_from_points(
    image: np.ndarray, points: np.ndarray, region_size: int = 32
) -> np.ndarray:
    """
    Average colors in square windows centered at 'points' on 'image'.
    Returns (N,3) float32.
    """
    h, w, _ = image.shape
    half = region_size // 2
    vals = []
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        x1, x2 = max(x - half, 0), min(x + half, w)
        y1, y2 = max(y - half, 0), min(y + half, h)
        region = image[y1:y2, x1:x2]
        vals.append(region.mean(axis=(0, 1)))
    return np.asarray(vals, dtype=np.float32)


def find_best_bbox_orientation_by_color(
    image_xyz: np.ndarray,
    bbox: np.ndarray,
    reference_xyz: np.ndarray,
    reference_pts: np.ndarray,
    region_size: int = 32,
) -> np.ndarray:
    """
    Choose the bbox orientation that minimizes color MSE against reference samples.
    Rotates the bbox 4 times and picks the lowest MSE after warping.
    """
    ref_shape = reference_xyz.shape[:2]
    ref_samples = sample_colors_from_points(reference_xyz, reference_pts, region_size)
    best_err, best_bbox = float("inf"), bbox.copy()
    b = bbox.copy()
    for _ in range(4):
        warped = warp_image(image_xyz, b, ref_shape)
        meas = sample_colors_from_points(warped, reference_pts, region_size)
        err = float(np.mean((meas - ref_samples) ** 2))
        if err < best_err:
            best_err, best_bbox = err, b.copy()
        b = np.roll(b, 1, axis=0)
    return best_bbox


# ------------------------- Sampling Strategies ------------------------- #


def sample_colors_from_grid(image: np.ndarray, region_size: int = 32) -> np.ndarray:
    """Uniform grid sampling: mean color per region_size tile. Returns (N,3)."""
    h, w, _ = image.shape
    out = []
    for y in range(0, h, region_size):
        for x in range(0, w, region_size):
            x2, y2 = min(x + region_size, w), min(y + region_size, h)
            if x2 > x and y2 > y:
                out.append(image[y:y2, x:x2].mean(axis=(0, 1)))
    return np.asarray(out, dtype=np.float32)


def sample_colors_from_grid_inner(
    image: np.ndarray,
    region_size: int = 32,
    inner: float = 0.7,
    blur_ksize: int = 5,
    robust: str = "median",
    min_side: int = 2,
) -> np.ndarray:
    """
    Grid-inner sampling: mean/median over the central 'inner' window of each grid cell.
    Optional Gaussian blur before sampling. Returns (N,3).
    """
    assert 0.0 < inner <= 1.0
    img = (
        cv.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
        if blur_ksize and blur_ksize % 2 == 1
        else image
    )
    H, W, _ = img.shape
    samples = []
    for y in range(0, H, region_size):
        for x in range(0, W, region_size):
            x2, y2 = min(x + region_size, W), min(y + region_size, H)
            if x2 <= x or y2 <= y:
                continue
            bw, bh = x2 - x, y2 - y
            ix1 = int(round(x + (1 - inner) * 0.5 * bw))
            ix2 = int(round(x2 - (1 - inner) * 0.5 * bw))
            iy1 = int(round(y + (1 - inner) * 0.5 * bh))
            iy2 = int(round(y2 - (1 - inner) * 0.5 * bh))
            ix1 = max(0, min(ix1, W - 1))
            ix2 = max(ix1 + 1, min(ix2, W))
            iy1 = max(0, min(iy1, H - 1))
            iy2 = max(iy1 + 1, min(iy2, H))
            if (ix2 - ix1) < min_side or (iy2 - iy1) < min_side:
                continue
            region = img[iy1:iy2, ix1:ix2]
            val = (
                np.median(region.reshape(-1, 3), axis=0)
                if robust == "median"
                else region.mean(axis=(0, 1))
            )
            samples.append(val.astype(np.float32))
    return (
        np.vstack(samples).astype(np.float32)
        if samples
        else np.empty((0, 3), dtype=np.float32)
    )


def sample_swatches(
    image: np.ndarray,
    sw_h: int,
    sw_v: int,
    inner: float = 0.7,
    blur_ksize: int = 5,
    robust: str = "median",
) -> np.ndarray:
    """
    Swatch-aware sampling on a regular H×V grid; uses only central 'inner' portion.
    Returns (sw_h*sw_v, 3).
    """
    assert 0.0 < inner <= 1.0
    img = (
        cv.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
        if blur_ksize and blur_ksize % 2 == 1
        else image
    )
    H, W, _ = img.shape
    cw, ch = W / sw_h, H / sw_v
    samples = []
    for j in range(sw_v):
        for i in range(sw_h):
            x1f, x2f = i * cw, (i + 1) * cw
            y1f, y2f = j * ch, (j + 1) * ch
            x1 = int(round(x1f + (1 - inner) * 0.5 * cw))
            x2 = int(round(x2f - (1 - inner) * 0.5 * cw))
            y1 = int(round(y1f + (1 - inner) * 0.5 * ch))
            y2 = int(round(y2f - (1 - inner) * 0.5 * ch))
            x1 = max(0, min(x1, W - 1))
            x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(y1 + 1, min(y2, H))
            region = img[y1:y2, x1:x2]
            val = (
                np.median(region.reshape(-1, 3), axis=0)
                if robust == "median"
                else region.mean(axis=(0, 1))
            )
            samples.append(val.astype(np.float32))
    return np.vstack(samples).astype(np.float32)


# ------------------------- Models & Evaluation ------------------------- #


def estimate_correction_matrix(
    measured: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Solve 3×3 linear mapping (no bias): Y ≈ X Mᵀ. Returns (3,3)."""
    M_T, *_ = np.linalg.lstsq(measured, reference, rcond=None)
    return M_T.T.astype(np.float32)


def apply_color_matrix(image_xyz: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 3×3 matrix to an XYZ image."""
    h, w = image_xyz.shape[:2]
    out = (image_xyz.reshape(-1, 3) @ M.T).clip(0, 1)
    return out.reshape(h, w, 3).astype(np.float32)


def estimate_correction_affine(
    measured: np.ndarray, reference: np.ndarray, ridge_lambda: float = 1e-4
) -> np.ndarray:
    """
    Solve affine 3×4 mapping with bias: Y ≈ [X 1] Aᵀ. Returns A (3,4).
    """
    N = measured.shape[0]
    X = np.hstack([measured.astype(np.float32), np.ones((N, 1), dtype=np.float32)])
    Y = reference.astype(np.float32)
    if ridge_lambda > 0:
        XtX = X.T @ X
        R = np.eye(4, dtype=np.float32) * ridge_lambda
        R[-1, -1] = 0.0
        A = np.linalg.solve(XtX + R, X.T @ Y)  # (4,3)
    else:
        A, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return A.T.astype(np.float32)


def apply_color_affine(image_xyz: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Apply affine 3×4 mapping A to an XYZ image."""
    h, w = image_xyz.shape[:2]
    flat = image_xyz.reshape(-1, 3).astype(np.float32)
    flat_aug = np.hstack([flat, np.ones((flat.shape[0], 1), dtype=np.float32)])
    out = (flat_aug @ A.T).clip(0, 1)
    return out.reshape(h, w, 3).astype(np.float32)


def design_matrix_poly(measured: np.ndarray, basis: str = "poly2") -> np.ndarray:
    """
    Build design matrix Φ(X) for polynomial color correction (Cheung-style).
    basis: "affine" | "poly2" | "poly3_cheung".
    """
    X = measured.astype(np.float32)
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    if basis == "affine":
        return np.hstack([x, y, z, np.ones_like(x)]).astype(np.float32)
    if basis == "poly2":
        xy, xz, yz = x * y, x * z, y * z
        return np.hstack(
            [x, y, z, x * x, y * y, z * z, xy, xz, yz, np.ones_like(x)]
        ).astype(np.float32)
    if basis == "poly3_cheung":
        xy, xz, yz = x * y, x * z, y * z
        x2, y2, z2 = x * x, y * y, z * z
        x3, y3, z3 = x2 * x, y2 * y, z2 * z
        x2y, x2z, xy2, xz2, y2z, yz2 = x2 * y, x2 * z, x * y2, x * z2, y2 * z, y * z2
        xyz = x * y * z
        return np.hstack(
            [
                x,
                y,
                z,
                x2,
                y2,
                z2,
                xy,
                xz,
                yz,
                x3,
                y3,
                z3,
                x2y,
                x2z,
                xy2,
                xz2,
                y2z,
                yz2,
                xyz,
                np.ones_like(x),
            ]
        ).astype(np.float32)
    raise ValueError(f"Unknown basis: {basis}")


def estimate_color_correction_poly(
    measured: np.ndarray,
    reference: np.ndarray,
    basis: str = "poly2",
    ridge_lambda: float = 1e-4,
) -> np.ndarray:
    """Ridge regression for polynomial mapping: Y ≈ Φ(X) W. Returns W (F,3)."""
    Phi = design_matrix_poly(measured, basis=basis)
    Y = reference.astype(np.float32)
    F = Phi.shape[1]
    R = np.eye(F, dtype=np.float32) * ridge_lambda
    R[-1, -1] = 0.0
    W = np.linalg.solve(Phi.T @ Phi + R, Phi.T @ Y)  # (F,3)
    return W.astype(np.float32)


def apply_color_correction_poly(
    image_xyz: np.ndarray, W: np.ndarray, basis: str = "poly2"
) -> np.ndarray:
    """Apply polynomial mapping W to an XYZ image."""
    H, Wd, _ = image_xyz.shape
    Phi = design_matrix_poly(image_xyz.reshape(-1, 3), basis=basis)
    out = (Phi @ W).clip(0, 1)
    return out.reshape(H, Wd, 3).astype(np.float32)


def _standardize(Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean, unit-std standardization per feature; returns (Phi_norm, mu, sigma)."""
    mu = Phi.mean(axis=0, keepdims=True)
    sigma = Phi.std(axis=0, keepdims=True) + 1e-8
    return (Phi - mu) / sigma, mu, sigma


def estimate_color_correction_poly_std(
    measured: np.ndarray,
    reference: np.ndarray,
    basis: str = "poly2",
    ridge_lambda: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge on standardized Φ(X). Returns (Wn, mu, sigma)."""
    Phi = design_matrix_poly(measured, basis=basis).astype(np.float32)
    Phi_n, mu, sigma = _standardize(Phi)
    Y = reference.astype(np.float32)
    F = Phi_n.shape[1]
    R = np.eye(F, dtype=np.float32) * ridge_lambda
    Wn = np.linalg.solve(Phi_n.T @ Phi_n + R, Phi_n.T @ Y)
    return Wn.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def apply_color_correction_poly_std(
    image_xyz: np.ndarray,
    Wn: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    basis: str = "poly2",
) -> np.ndarray:
    """Apply Wn learned on standardized Φ(X) to an image."""
    H, Wd, _ = image_xyz.shape
    Phi = design_matrix_poly(image_xyz.reshape(-1, 3), basis=basis).astype(np.float32)
    Phi_n = (Phi - mu) / sigma
    out = (Phi_n @ Wn).clip(0, 1)
    return out.reshape(H, Wd, 3).astype(np.float32)


def deltaE_report(meas_xyz: np.ndarray, ref_xyz: np.ndarray) -> dict:
    """Compute CIEDE2000 statistics (mean/median/p95) on two XYZ point sets."""
    lab_meas = xyz2lab(meas_xyz.reshape(1, -1, 3)).reshape(-1, 3)
    lab_ref = xyz2lab(ref_xyz.reshape(1, -1, 3)).reshape(-1, 3)
    dE = deltaE_ciede2000(lab_meas, lab_ref).astype(np.float32)
    return {
        "mean": float(np.mean(dE)),
        "median": float(np.median(dE)),
        "p95": float(np.percentile(dE, 95)),
        "per_swatch": dE.tolist(),
    }


def write_results_csv(results: list[dict], path: Path) -> None:
    """Write experiment rows to CSV."""
    fieldnames = [
        "sampler",
        "model",
        "lambda",
        "n_samples",
        "deltaE_mean",
        "deltaE_median",
        "deltaE_p95",
        "outfile",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)


def _heatmap_shape(
    sampler_name: str, warped_shape: tuple[int, int, int]
) -> tuple[int, int] | None:
    """
    Infer (rows, cols) to reshape ΔE list for a heatmap, when possible.
    - swatch_6x4 -> (4,6)
    - grid32 / grid32_inner -> ceil(H/32), ceil(W/32)
    - points samplers -> None (no regular grid)
    """
    H, W = warped_shape[0], warped_shape[1]
    if sampler_name == "swatch_6x4":
        return (4, 6)
    if sampler_name in ("grid32", "grid32_inner"):
        return (math.ceil(H / 32), math.ceil(W / 32))
    return None


def save_deltaE_heatmap(
    dE_list: list[float], shape_rc: tuple[int, int], path: Path, title: str
) -> None:
    """Save ΔE heatmap (rows×cols) to disk."""
    arr = np.asarray(dE_list, dtype=np.float32)
    if arr.size != shape_rc[0] * shape_rc[1]:
        return  # skip if counts don't match
    arr = arr.reshape(shape_rc[0], shape_rc[1])
    plt.figure(figsize=(6, 3.5))
    plt.imshow(arr, interpolation="nearest")
    plt.colorbar(label="ΔE00")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ------------------------- Pipeline Execution ------------------------- #

captured_ann = load_json(paths["captured_ann"])
reference_ann = load_json(paths["reference_ann"])

captured_srgb = load_image_rgb(paths["captured_img"])
reference_srgb = load_image_rgb(paths["reference_img"])

captured_xyz = to_xyz(captured_srgb)
reference_xyz = to_xyz(reference_srgb)

bbox, captured_pts = get_bbox_and_points(captured_ann)
_, reference_pts = get_bbox_and_points(reference_ann)

aligned_bbox = find_best_bbox_orientation_by_color(
    captured_xyz, bbox, reference_xyz, reference_pts, region_size=32
)

warped_xyz = warp_image(captured_xyz, aligned_bbox, reference_xyz.shape[:2])
save_image_rgb(paths["output_dir"] / f"{cap_id}_warped.jpg", to_srgb(warped_xyz))

# ------------------------- Automatic Experiment Grid ------------------------- #


def run_experiments():
    """Fit/evaluate all meaningful (sampler, model, lambda) combos; save outputs, CSV, and ΔE heatmaps."""
    results = []

    def s_points4(img):
        return sample_colors_from_points(
            cv.GaussianBlur(img, (5, 5), 0), reference_pts[:4], 32
        )

    def s_points10(img):
        return sample_colors_from_points(
            cv.GaussianBlur(img, (5, 5), 0), reference_pts, 32
        )

    def s_grid32(img):
        return sample_colors_from_grid(cv.GaussianBlur(img, (5, 5), 0), 32)

    def s_grid32_inner(img):
        return sample_colors_from_grid_inner(img, 32, 0.7, 5, "median")

    def s_swatch_6x4(img):
        return sample_swatches(img, 6, 4, 0.7, 5, "median")

    samplers = {
        "points4": s_points4,
        "points10": s_points10,
        "grid32": s_grid32,
        "grid32_inner": s_grid32_inner,
        "swatch_6x4": s_swatch_6x4,
    }

    # add standardized polynomial variants
    model_cfg = {
        "affine": {"lambdas": [1e-4]},
        "poly2": {"lambdas": [1e-4, 3e-4, 1e-3, 3e-3]},
        "poly3_cheung": {"lambdas": [1e-3, 3e-3, 1e-2]},
        "poly2_std": {"lambdas": [1e-4, 3e-4, 1e-3, 3e-3]},
        "poly3_cheung_std": {"lambdas": [1e-3, 3e-3, 1e-2]},
    }

    ref_cache = {}

    for s_name, sampler in samplers.items():
        src = sampler(warped_xyz)
        ref = ref_cache.get(s_name) or sampler(reference_xyz)
        ref_cache[s_name] = ref

        if src.shape != ref.shape or src.size == 0:
            print(
                f"[WARN] Skipping sampler '{s_name}' (shape mismatch: {src.shape} vs {ref.shape})."
            )
            continue

        for m_name, cfg in model_cfg.items():
            for lam in cfg["lambdas"]:
                try:
                    if m_name == "affine":
                        M = estimate_correction_affine(src, ref, ridge_lambda=lam)
                        corrected_full = apply_color_affine(captured_xyz, M)
                        warped_corr = apply_color_affine(warped_xyz, M)

                    elif m_name.endswith("_std"):
                        basis = (
                            "poly2" if m_name.startswith("poly2") else "poly3_cheung"
                        )
                        Wn, mu, sigma = estimate_color_correction_poly_std(
                            src, ref, basis=basis, ridge_lambda=lam
                        )
                        corrected_full = apply_color_correction_poly_std(
                            captured_xyz, Wn, mu, sigma, basis=basis
                        )
                        warped_corr = apply_color_correction_poly_std(
                            warped_xyz, Wn, mu, sigma, basis=basis
                        )

                    else:
                        W = estimate_color_correction_poly(
                            src, ref, basis=m_name, ridge_lambda=lam
                        )
                        corrected_full = apply_color_correction_poly(
                            captured_xyz, W, basis=m_name
                        )
                        warped_corr = apply_color_correction_poly(
                            warped_xyz, W, basis=m_name
                        )

                    out_name = f"{cap_id}_corr_{s_name}_{m_name}_lam{lam:g}.jpg"
                    save_image_rgb(
                        paths["output_dir"] / out_name, to_srgb(corrected_full)
                    )

                    meas_after = sampler(warped_corr)
                    rep = deltaE_report(meas_after, ref)

                    row = {
                        "sampler": s_name,
                        "model": m_name,
                        "lambda": lam,
                        "n_samples": int(src.shape[0]),
                        "deltaE_mean": rep["mean"],
                        "deltaE_median": rep["median"],
                        "deltaE_p95": rep["p95"],
                        "outfile": out_name,
                    }
                    results.append(row)

                    print(
                        f"[OK] {s_name} | {m_name} | λ={lam:g} | N={src.shape[0]} "
                        f"| ΔE00 mean={rep['mean']:.3f}, median={rep['median']:.3f}, p95={rep['p95']:.3f} "
                        f"| saved={out_name}"
                    )

                    # ΔE heatmap where a regular grid is known
                    shape_rc = _heatmap_shape(s_name, warped_xyz.shape)
                    if shape_rc is not None:
                        hm_name = f"{cap_id}_heatmap_{s_name}_{m_name}_lam{lam:g}.png"
                        save_deltaE_heatmap(
                            rep["per_swatch"],
                            shape_rc,
                            paths["output_dir"] / hm_name,
                            title=f"{s_name} | {m_name} | λ={lam:g}",
                        )

                except Exception as e:
                    print(f"[ERR] {s_name} | {m_name} | λ={lam:g}: {e}")

    if not results:
        print("No results collected.")
        return

    # CSV export
    csv_path = paths["output_dir"] / f"{cap_id}_experiments.csv"
    write_results_csv(results, csv_path)
    print(f"\nExperiment table saved: {csv_path.name}")

    # Ranking
    results_sorted = sorted(results, key=lambda r: r["deltaE_mean"])
    print("\n=== RESULTS (sorted by ΔE00 mean) ===")
    for r in results_sorted:
        print(
            f"{r['deltaE_mean']:.3f}  (median {r['deltaE_median']:.3f}, p95 {r['deltaE_p95']:.3f})  "
            f"N={r['n_samples']:4d}  {r['sampler']:12s}  {r['model']:16s}  λ={r['lambda']:g}  -> {r['outfile']}"
        )

    best = results_sorted[0]
    print(
        "\n>>> BEST CONFIGURATION:"
        f" sampler={best['sampler']} | model={best['model']} | λ={best['lambda']:g}"
        f" | N={best['n_samples']} | ΔE00 mean={best['deltaE_mean']:.3f},"
        f" median={best['deltaE_median']:.3f}, p95={best['deltaE_p95']:.3f}"
        f" | file={best['outfile']}"
    )


run_experiments()
