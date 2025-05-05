# Color-Constancy-Using-Household-Objects

This project explores color constancy using images of household objects and a simple four-point color calibration method. The current implementation corrects color inconsistencies by aligning RGB values from captured and reference images.

---

## 📦 Project Structure

```
data/
├── raw/              # Place your captured images and their annotations here
│   ├── images/
│   └── annotations/
├── ref/              # Reference images and annotations (ideal lighting)
│   ├── images/
│   └── annotations/
├── corrected/        # Output folder for corrected images
│   └── images/
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

### 4. Set the image IDs (manual for now)

Edit `src/four_point_calibrations.py` and change these two variables to match your image filenames:

```python
cap_id = "your_image_name_without_extension"
ref_id = "your_reference_name_without_extension"
```

Example:

```python
cap_id = "162962321"
ref_id = "egp_100"
```

### 5. Run the script

```bash
python src/four_point_calibrations.py
```

This will:
- Load your images and annotations
- Apply color correction based on 4-point calibration
- Save the corrected image to `data/corrected/images/{cap_id}_corrected.jpg`

---

## 📎 Notes

- All annotations should be in [LabelMe](https://github.com/wkentaro/labelme) format (`.json`).
- Ensure your annotations have `pt1`, `pt2`, `pt3`, `pt4` points labeled in both captured and reference images.
- Bounding box (`bbox`) is optional, but supported for visualization.

---

## ✅ TODO

- [ ] Auto-detect orientation mismatch
- [ ] Support batch processing of images
- [ ] Add GUI or CLI for setting image IDs
- [ ] Automatic loading of matching reference by filename pattern

---

## 🧠 Author

Sasa Marjanovic  
2024–2025 Master's Thesis Project  
