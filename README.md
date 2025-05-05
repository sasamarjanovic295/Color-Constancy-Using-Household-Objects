# Color-Constancy-Using-Household-Objects

This project explores color constancy using images of household objects and multiple calibration strategies: four-point, multi-point, and grid-based color correction. It aligns RGB values from captured and reference images to improve consistency under different lighting conditions.

---

## 📦 Project Structure

```
data/
├── raw/              # Captured images and their annotations (under unknown lighting)
│   ├── images/
│   └── annotations/
├── ref/              # Reference images and annotations (under ideal lighting)
│   ├── images/
│   └── annotations/
├── corrected/        # Output images after color correction
│   └── images/
src/                  # All scripts and processing code
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/sasamarjanovic295/Color-Constancy-Using-Household-Objects.git
cd Color-Constancy-Using-Household-Objects
```

### 2. Create and activate the Anaconda environment

```bash
conda env create -f environment.yml
conda activate color-constancy
```

> If you haven't created `environment.yml` yet, export your current environment:
>
> ```bash
> conda env export --no-builds | grep -v "prefix:" > environment.yml
> ```

### 3. Add data

- Place **captured images and annotations** in: `data/raw/images/` and `data/raw/annotations/`
- Place **reference images and annotations** in: `data/ref/images/` and `data/ref/annotations/`
- The corrected images will be saved in: `data/corrected/images/`

### 4. Set the image IDs

Edit `src/four_points_calibration.py` and set the following variables:

```python
cap_id = "your_captured_image_name"
ref_id = "your_reference_image_name"
```

Example:
```python
cap_id = "162962321"
ref_id = "egp_100"
```

### 5. Run the script

```bash
python src/four_points_calibration.py
```

This will:
- Load the captured and reference images and annotations
- Warp the captured image using a detected bounding box
- Compute a 3x3 color correction matrix using:
  - 4-point calibration
  - 10-point calibration
  - Grid-based sampling
- Apply and save three versions of corrected images:
  - `{cap_id}_corrected_4pt.jpg`
  - `{cap_id}_corrected_10pt.jpg`
  - `{cap_id}_corrected_grid.jpg`

---

## 🧪 Validation and Evaluation (Upcoming)

The next phase of the project will include:

- 📏 **Validation**: Evaluate how well each calibration method corrects color using error metrics such as MSE or ΔE
- 🤖 **Automatic Detection**: Implement object detection (e.g., YOLO, Detectron2) to find the banknote / color checker automatically
- 🧪 **Grid Size Testing**: Analyze performance with different grid resolutions (e.g., 16x16, 32x32, 64x64)
- 🔁 **Batch Processing**: Enable processing of all images in the dataset automatically
- 📈 **Visualization Tools**: Compare before/after images and show color patches for sample points

---

## 📎 Notes

- All annotations are in [LabelMe](https://github.com/wkentaro/labelme) format (`.json`).
- Captured images should have:
  - Bounding box (`bbox`) of the object (e.g., banknote)
  - At least 4 to 10 point annotations (`pt1`–`ptN`)
- Reference images require only `pt` points (no `bbox` required).

---

## ✅ TODO

- [ ] Add quantitative validation (MSE, ΔE)
- [ ] Test various grid sizes for sampling
- [ ] Implement automatic object/ROI detection
- [ ] Add CLI or GUI for selecting image IDs or processing modes
- [ ] Batch processing support

---

## 🧠 Author

Sasa Marjanovic  
2024–2025 Master's Thesis Project
