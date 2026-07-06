# Color Constancy Using Household Objects

Master's thesis project exploring whether Euro banknotes can replace professional
ColorChecker cards for color-calibrating photographs, enabling objective skin tone
measurement (ITA) without specialized equipment.

**Author:** Sasa Marjanovic | **Institution:** FERIT Osijek, JJ Strossmayer University | **Year:** 2025/2026

---

## Project Structure

```
calibrate_single.py           # Single-image calibration pipeline
calibrate_dataset.py           # Batch runner over full dataset
evaluate_results.py            # Evaluation: 15+ tables and 14 figures
process_images.py              # Image preprocessing (rotation, crop)
sample_reference_banknotes.py  # Generate reference banknote samples

src/
  colorchecker.py              # SpyderCheckr Photo (SCK300) detection + measurement
  banknote_detection.py        # SIFT+RANSAC Euro banknote detection
  banknote_sampling.py         # Grid-based banknote color sampling
  detection.py                 # Low-level SIFT+RANSAC homography engine
  hand_segmentation_v2.py      # MediaPipe hand/skin segmentation
  skin_measurement.py          # ITA skin tone measurement (Chardon 1991)
  color_calibration.py         # Correction methods (linear, affine, poly2, poly3, etc.)
  evaluation.py                # Metrics, dataclasses, CSV I/O
  reference_data.py            # SCK300 Lab reference values (darktable)

tests/                         # Unit tests
data/
  annotations.json             # Master annotation file (1203 images)
  images/                      # Preprocessed images
  ref/                         # Banknote reference templates per denomination
results/                       # Experiment outputs (CSV + visualizations)
```

---

## Pipeline Overview

Each image contains three objects: a hand, a ColorChecker (SCK300), and a Euro banknote.

```
1. Detect ColorChecker  ->  48 swatch colors (XYZ-D65)
2. Detect banknote      ->  N paired color samples (measured + reference)
3. Segment hand         ->  binary skin mask

4. For each correction method x reference object:
   - Fit calibration matrix (XYZ color space)
   - Apply to full image
   - Measure skin tone (ITA) on corrected image
   - Compute DE00 on ColorChecker swatches (ground truth)
```

**Correction methods:** linear (3x3), affine (3x4), poly2 (30 params), poly3 Cheung (60 params), Gray World, Shades of Gray.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Calibrate a single image

```python
from calibrate_single import calibrate_image

results = calibrate_image(
    image_srgb,                    # float32 sRGB [0,1]
    denomination=20,               # Euro denomination
    side="single_number",          # Banknote side
    ref_dir=Path("data/ref"),      # Reference templates
    methods=["affine", "poly2"],   # Correction methods
)
```

### 3. Run full dataset

```bash
python calibrate_dataset.py \
  --data-dir data/ \
  --output results/run_v3_all \
  --methods linear affine poly2 poly3_cheung gray_world shades_of_gray \
  --resume
```

### 4. Evaluate results

```bash
python evaluate_results.py results/run_v3_all/results.csv \
  --output-dir results/run_v3_all/evaluation
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Dataset

- **1203 images**: 5 persons, 6 lighting conditions (L1-L6), 5 denominations, 2 banknote sides, 2 hands, 2 orientations
- **L2 held out** for generalization testing
- Each image: hand + SpyderCheckr Photo (SCK300) + Euro banknote
- Annotations: `data/annotations.json`

---

## Key Metrics

| Metric | Purpose |
|--------|---------|
| DE00 on CC swatches | Calibration accuracy (ground truth) |
| std(ITA) across lightings | Skin tone measurement stability |
| Chardon consistency | Categorical stability across conditions |
| Wilcoxon signed-rank | Statistical significance of differences |

---

## References

- Chardon, A., Cretois, I., & Hourseau, C. (1991). Skin colour typology and suntanning pathways. *International Journal of Cosmetic Science*.
- Cheung, V., Westland, S., Connah, D., & Ripamonti, C. (2004). A comparative study of the characterisation of colour cameras. *Journal of the Society of Dyers and Colourists*.
- Ly, B. C. K., et al. (2020). Research Techniques Made Simple: Cutaneous Colorimetry. *Journal of Investigative Dermatology*.
