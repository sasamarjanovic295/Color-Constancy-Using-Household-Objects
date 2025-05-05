import json
from pathlib import Path

import cv2 as cv
import numpy as np

proj_root = Path(__file__).parent.parent
src_dir = proj_root / "src"

cap_id = "162962321"
ref_id = "egp_100"

captured_ann_path = proj_root / "data" / "raw" / "annotations" / f"{cap_id}.json"
reference_ann_path = proj_root / "data" / "ref" / "annotations" / f"{ref_id}.json"
captured_img_path = proj_root / "data" / "raw" / "images" / f"{cap_id}.jpg"
reference_img_path = proj_root / "data" / "ref" / "images" / f"{ref_id}.jpg"
output_path = proj_root / "data" / "corrected" / "images" / f"{cap_id}_corrected.jpg"

with open(captured_ann_path, "r") as f:
    captured_ann = json.load(f)
with open(reference_ann_path, "r") as f:
    reference_ann = json.load(f)


def show_rgb_image(win_name, rgb_img):
    if rgb_img.dtype == np.float32:
        rgb_img = (rgb_img * 255).astype(np.uint8)
    bgr_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
    cv.imshow(win_name, bgr_img)


def draw_and_show_annotation(
    win_name, img, ann, color_point=(0, 255, 0), color_bbox=(0, 0, 255)
):
    if img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)

    annotated_img = img.copy()

    for shape in ann["shapes"]:
        if shape["label"].startswith("pt"):
            x, y = map(int, shape["points"][0])
            cv.circle(annotated_img, (x, y), 5, color_point, -1)
            cv.putText(
                annotated_img,
                shape["label"],
                (x + 10, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

    for shape in ann["shapes"]:
        if shape["label"] == "bbox":
            pts = np.array(shape["points"], dtype=np.int32).reshape((-1, 1, 2))
            cv.polylines(
                annotated_img, [pts], isClosed=True, color=color_bbox, thickness=3
            )

    show_rgb_image(win_name, annotated_img)


def save_rgb_image(path, rgb_img):
    if rgb_img.dtype == np.float32:
        rgb_img = (rgb_img * 255).astype(np.uint8)

    bgr_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)

    success = cv.imwrite(str(path), bgr_img)
    if not success:
        raise IOError(f"Failed to save image to: {path}")


def get_points(ann, label_prefix="pt"):
    pts = []
    for shape in ann["shapes"]:
        label = shape["label"]
        if label.startswith(label_prefix):
            pts.append((shape["points"][0][0], shape["points"][0][1]))

    pts_sorted = sorted(
        pts,
        key=lambda p: int(
            [
                s["label"][2:]
                for s in ann["shapes"]
                if s["points"][0][0] == p[0] and s["points"][0][1] == p[1]
            ][0]
        ),
    )
    return np.array(pts_sorted)


captured_pts = get_points(captured_ann)
reference_pts = get_points(reference_ann)

ncaptured_img = cv.imread(str(captured_img_path)).astype(np.float32) / 255.0
ncaptured_img = cv.cvtColor(ncaptured_img, cv.COLOR_BGR2RGB)

nreference_img = cv.imread(str(reference_img_path)).astype(np.float32) / 255.0
nreference_img = cv.cvtColor(nreference_img, cv.COLOR_BGR2RGB)

draw_and_show_annotation("Captured Image - Annotated", ncaptured_img, captured_ann)
draw_and_show_annotation("Reference Image - Annotated", nreference_img, reference_ann)


def sample_colors(img, pts, window_size=3):
    coords = np.round(pts).astype(int)
    h, w = img.shape[:2]
    colors = []

    offset = window_size // 2

    for x, y in coords:
        x1 = np.clip(x - offset, 0, w - 1)
        x2 = np.clip(x + offset + 1, 0, w)
        y1 = np.clip(y - offset, 0, h - 1)
        y2 = np.clip(y + offset + 1, 0, h)

        patch = img[y1:y2, x1:x2, :]
        avg_color = patch.reshape(-1, 3).mean(axis=0)
        colors.append(avg_color)

    return np.array(colors)


measured_colors = sample_colors(ncaptured_img, captured_pts)
ref_colors = sample_colors(nreference_img, reference_pts)

# Solve for 3x3 color correction matrix M_est
# measured_colors @ M_est.T ≈ ref_colors
M_est_T, *_ = np.linalg.lstsq(measured_colors, ref_colors, rcond=None)
M_est = M_est_T.T
print("Estimated color correction matrix M_est:")
print(M_est)

print("Measured (captured) colors:\n", measured_colors)
print("Reference colors:\n", ref_colors)
print("Corrected colors:\n", measured_colors @ M_est.T)

diff = np.abs((measured_colors @ M_est.T) - ref_colors)
print("Differences:\n", diff)
print("Mean diff:", diff.mean())

h, w = ncaptured_img.shape[:2]
flat = ncaptured_img.reshape(-1, 3)
corrected_flat = np.clip(flat @ M_est.T, 0, 1)
corrected_img = (corrected_flat.reshape(h, w, 3) * 255).astype(np.uint8)

show_rgb_image("Corrected Image", corrected_img)

save_rgb_image(output_path, corrected_img)

cv.waitKey(0)
cv.destroyAllWindows()
