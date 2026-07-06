"""reference_data.py — SCK300 (Spyder Checkr Photo) Lab reference values as XYZ.

Also provides per-panel linear sRGB reference arrays for use with
colour-checker-detection's `sample_colour_checker` MSE rotation search.
"""

from __future__ import annotations

import colour
import numpy as np

# Lab values sourced from darktable colorchecker.h (PR #13174, merged).
# Column-major order: A1-A6, B1-B6, ..., H1-H6 (letters = columns, digits = rows).
_SCK300_LAB = np.array([
    # Column A
    [61.35,  34.81,  18.38],  # A1
    [75.50,   5.84,  50.42],  # A2
    [66.82, -25.10,  23.47],  # A3
    [60.53, -22.62, -20.40],  # A4
    [59.66,  -2.03, -28.46],  # A5
    [59.15,  30.83,  -5.72],  # A6
    # Column B
    [82.68,   5.03,   3.02],  # B1
    [82.25,  -2.42,   3.78],  # B2
    [82.29,   2.20,  -2.04],  # B3
    [24.89,   4.43,   0.78],  # B4
    [25.16,  -3.88,   2.13],  # B5
    [26.13,   2.61,  -5.03],  # B6
    # Column C
    [85.42,   9.41,  14.49],  # C1
    [74.28,   9.05,  27.21],  # C2
    [64.57,  12.39,  37.24],  # C3
    [44.49,  17.23,  26.24],  # C4
    [25.29,   7.95,   8.87],  # C5
    [22.67,   2.11,  -1.10],  # C6
    # Column D
    [92.72,   1.89,   2.76],  # D1
    [88.85,   1.59,   2.27],  # D2
    [73.42,   0.99,   1.89],  # D3
    [57.15,   0.57,   1.19],  # D4
    [41.57,   0.24,   1.45],  # D5
    [25.65,   1.24,   0.05],  # D6
    # Column E
    [96.04,   2.16,   2.60],  # E1
    [80.44,   1.17,   2.05],  # E2
    [65.52,   0.69,   1.86],  # E3
    [49.62,   0.58,   1.56],  # E4
    [33.55,   0.35,   1.40],  # E5
    [16.91,   1.43,  -0.81],  # E6
    # Column F
    [47.12, -32.52, -28.75],  # F1
    [50.49,  53.45, -13.55],  # F2
    [83.61,   3.36,  87.02],  # F3
    [41.05,  60.75,  31.17],  # F4
    [54.14, -40.76,  34.75],  # F5
    [24.75,  13.78, -49.48],  # F6
    # Column G
    [60.94,  38.21,  61.31],  # G1
    [37.80,   7.30, -43.04],  # G2
    [49.81,  48.50,  15.76],  # G3
    [28.88,  19.36, -24.48],  # G4
    [72.45, -23.57,  60.47],  # G5
    [71.65,  23.74,  72.28],  # G6
    # Column H
    [70.19, -31.85,   1.98],  # H1
    [54.38,   8.84, -25.71],  # H2
    [42.03, -15.78,  22.93],  # H3
    [48.82,  -5.11, -23.08],  # H4
    [65.10,  18.14,  18.68],  # H5
    [36.13,  14.15,  15.78],  # H6
], dtype=np.float64)

# D50 is assumed because it is the darktable pipeline default and the ISO standard
# for ICC profiling. The source (colorchecker.h) does not state the illuminant
# explicitly — verify against darktable src/iop/channelmixerrgb.c before publication.
_D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]

SCK300_REFERENCE_XYZ: np.ndarray = colour.Lab_to_XYZ(_SCK300_LAB, illuminant=_D50)

_D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
_D50_XYZ = colour.xy_to_XYZ(_D50)
_D65_XYZ = colour.xy_to_XYZ(_D65)


def _lab_panel_to_linear_srgb_rowmajor(
    lab_colmajor: np.ndarray, n_cols: int, n_rows: int
) -> np.ndarray:
    """Convert a column-major Lab panel array to row-major linear sRGB.

    The row-major order matches what ``swatch_masks`` produces when the warped
    panel image has ``swatches_h=n_cols`` columns and ``swatches_v=n_rows`` rows.
    """
    grid = lab_colmajor.reshape(n_cols, n_rows, 3)
    lab_rowmaj = grid.transpose(1, 0, 2).reshape(n_cols * n_rows, 3)

    xyz_d50 = colour.Lab_to_XYZ(lab_rowmaj, illuminant=_D50)
    xyz_d65 = colour.adaptation.chromatic_adaptation_VonKries(
        np.asarray(xyz_d50, dtype=np.float64),
        _D50_XYZ,
        _D65_XYZ,
        transform="Bradford",
    )
    srgb_linear = colour.XYZ_to_RGB(
        np.asarray(xyz_d65, dtype=np.float64),
        colour.RGB_COLOURSPACES["sRGB"],
        chromatic_adaptation_transform=None,
    )
    return np.asarray(srgb_linear, dtype=np.float32)


# Left panel (columns A-D, indices 0-23): neutral / skin-tone patches.
# Row-major order: A1,B1,C1,D1, A2,B2,C2,D2, ..., A6,B6,C6,D6.
SCK300_LEFT_PANEL_SRGB_LINEAR: np.ndarray = _lab_panel_to_linear_srgb_rowmajor(
    _SCK300_LAB[:24], n_cols=4, n_rows=6
)

# Right panel (columns E-H, indices 24-47): saturated / chromatic patches.
# Row-major order: E1,F1,G1,H1, E2,F2,G2,H2, ..., E6,F6,G6,H6.
SCK300_RIGHT_PANEL_SRGB_LINEAR: np.ndarray = _lab_panel_to_linear_srgb_rowmajor(
    _SCK300_LAB[24:], n_cols=4, n_rows=6
)


SCK300_PATCH_NAMES: list[str] = [
    "A1", "A2", "A3", "A4", "A5", "A6",
    "B1", "B2", "B3", "B4", "B5", "B6",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "D1", "D2", "D3", "D4", "D5", "D6",
    "E1", "E2", "E3", "E4", "E5", "E6",
    "F1", "F2", "F3", "F4", "F5", "F6",
    "G1", "G2", "G3", "G4", "G5", "G6",
    "H1", "H2", "H3", "H4", "H5", "H6",
]
