"""
Refactored script for color correction based on annotated banknote images.
Includes:
- Flexible color sampling from annotated points or a full grid.
- Correction matrix estimation (least squares).
- Saving intermediate and final results.
"""

import json
from pathlib import Path

import cv2 as cv
import numpy as np

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

# ------------------------- Utility Functions ------------------------- #


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_image_rgb(path):
    image = cv.imread(str(path)).astype(np.float32) / 255.0
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def save_image_rgb(path, image):
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(str(path), bgr)


def get_bbox_and_points(ann):
    bbox_list = [shape["points"] for shape in ann["shapes"] if shape["label"] == "bbox"]
    bbox = np.array(bbox_list[0], dtype=np.float32) if bbox_list else None

    points = {
        shape["label"]: shape["points"][0]
        for shape in ann["shapes"]
        if shape["label"].startswith("pt")
    }
    sorted_pts = [points[f"pt{i}"] for i in range(1, len(points) + 1)]
    return bbox, np.array(sorted_pts, dtype=np.float32)


def find_best_bbox_orientation(bbox, src_pts, dst_pts, ref_shape):
    min_error = float("inf")
    best_bbox = bbox.copy()
    for _ in range(4):
        transform = cv.getPerspectiveTransform(
            bbox,
            np.array(
                [
                    [0, 0],
                    [ref_shape[1] - 1, 0],
                    [ref_shape[1] - 1, ref_shape[0] - 1],
                    [0, ref_shape[0] - 1],
                ],
                dtype=np.float32,
            ),
        )
        projected = cv.perspectiveTransform(src_pts[None, :, :], transform)[0]
        error = np.mean((projected - dst_pts) ** 2)
        if error < min_error:
            min_error = error
            best_bbox = bbox.copy()
        bbox = np.roll(bbox, 1, axis=0)
    return best_bbox


def warp_image(image, bbox, ref_shape):
    h, w = ref_shape
    dst_rect = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    transform = cv.getPerspectiveTransform(bbox, dst_rect)
    return cv.warpPerspective(image, transform, (w, h), flags=cv.INTER_LINEAR)


def sample_colors_from_points(image, points, region_size=32):
    h, w, _ = image.shape
    half = region_size // 2
    means = []
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        x1, x2 = max(x - half, 0), min(x + half, w)
        y1, y2 = max(y - half, 0), min(y + half, h)
        region = image[y1:y2, x1:x2]
        means.append(region.mean(axis=(0, 1)))
    return np.array(means)


def sample_colors_from_grid(image, region_size=32):
    h, w, _ = image.shape
    samples = []
    for y in range(0, h, region_size):
        for x in range(0, w, region_size):
            x2 = min(x + region_size, w)
            y2 = min(y + region_size, h)
            region = image[y:y2, x:x2]
            if region.size > 0:
                samples.append(region.mean(axis=(0, 1)))
    return np.array(samples)


def estimate_correction_matrix(measured, reference):
    M_T, *_ = np.linalg.lstsq(measured, reference, rcond=None)
    return M_T.T


def apply_color_matrix(image, matrix):
    h, w = image.shape[:2]
    flat = image.reshape(-1, 3)
    corrected = np.clip(flat @ matrix.T, 0, 1)
    return (corrected.reshape(h, w, 3) * 255).astype(np.uint8)


# ------------------------- Pipeline Execution ------------------------- #

captured_ann = load_json(paths["captured_ann"])
reference_ann = load_json(paths["reference_ann"])

captured_img = load_image_rgb(paths["captured_img"])
reference_img = load_image_rgb(paths["reference_img"])

bbox, captured_pts = get_bbox_and_points(captured_ann)
_, reference_pts = get_bbox_and_points(reference_ann)

aligned_bbox = find_best_bbox_orientation(
    bbox, captured_pts[:4], reference_pts[:4], reference_img.shape[:2]
)
warped_img = warp_image(captured_img, aligned_bbox, reference_img.shape[:2])

# --- Save warped image --- #
save_image_rgb(paths["output_dir"] / f"{cap_id}_warped.jpg", warped_img)

# --- Estimate M_est using different strategies --- #

# (1) Using 4 points only
M_4pt = estimate_correction_matrix(
    sample_colors_from_points(warped_img, reference_pts[:4]),
    sample_colors_from_points(reference_img, reference_pts[:4]),
)
corrected_4pt = apply_color_matrix(captured_img, M_4pt)
save_image_rgb(paths["output_dir"] / f"{cap_id}_corrected_4pt.jpg", corrected_4pt)

# (2) Using 10 points
M_10pt = estimate_correction_matrix(
    sample_colors_from_points(warped_img, reference_pts),
    sample_colors_from_points(reference_img, reference_pts),
)
corrected_10pt = apply_color_matrix(captured_img, M_10pt)
save_image_rgb(paths["output_dir"] / f"{cap_id}_corrected_10pt.jpg", corrected_10pt)

# (3) Using grid-based sampling
M_grid = estimate_correction_matrix(
    sample_colors_from_grid(warped_img), sample_colors_from_grid(reference_img)
)
corrected_grid = apply_color_matrix(captured_img, M_grid)
save_image_rgb(paths["output_dir"] / f"{cap_id}_corrected_grid.jpg", corrected_grid)

print("All correction versions saved successfully.")
