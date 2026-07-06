#!/usr/bin/env python3
"""Evaluate pipeline results — generate statistical tables and figures.

Usage:
    python evaluate_results.py results.csv [--output-dir results/evaluation] [--held-out L2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from src.evaluation import (
    coefficient_of_variation,
    chardon_consistency,
    chardon_consistency_rate,
    ita_reduction,
    paired_wilcoxon,
    read_results_csv,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS_ORDER = [
    "baseline", "linear", "affine", "poly2",
    "poly3_cheung", "gray_world", "shades_of_gray",
]
CHARDON_CATS = ["Very Light", "Light", "Intermediate", "Tan", "Brown", "Dark"]
CHARDON_CMAP = {
    "Very Light": "#ffe0c0", "Light": "#f5c08a", "Intermediate": "#d4a06a",
    "Tan": "#b07850", "Brown": "#7a5030", "Dark": "#3e2818",
}

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_baseline_after_fields(df: pd.DataFrame) -> None:
    """Use uncalibrated measurements as baseline comparison values in-memory."""
    baseline = df["correction_method"] == "baseline"
    pairs = [
        ("cc_de_mean_before", "cc_de_mean_after"),
        ("cc_de_median_before", "cc_de_median_after"),
        ("cc_de_trimean_before", "cc_de_trimean_after"),
        ("cc_de_p95_before", "cc_de_p95_after"),
        ("cc_de_max_before", "cc_de_max_after"),
        ("cc_de_best25_before", "cc_de_best25_after"),
        ("cc_de_worst25_before", "cc_de_worst25_after"),
        ("bn_de_mean_before", "bn_de_mean_after"),
        ("bn_de_median_before", "bn_de_median_after"),
        ("bn_de_trimean_before", "bn_de_trimean_after"),
        ("bn_de_p95_before", "bn_de_p95_after"),
        ("bn_de_max_before", "bn_de_max_after"),
        ("bn_de_best25_before", "bn_de_best25_after"),
        ("bn_de_worst25_before", "bn_de_worst25_after"),
        ("L_median_before", "L_median_after"),
        ("a_median_before", "a_median_after"),
        ("b_median_before", "b_median_after"),
        ("ITA_before", "ITA_after"),
        ("chardon_before", "chardon_after"),
    ]
    for before, after in pairs:
        mask = baseline & df[after].isna()
        df.loc[mask, after] = df.loc[mask, before]


def _load(path: str, held_out: str) -> pd.DataFrame:
    rows = read_results_csv(path)
    if not rows:
        print("No results found in", path)
        sys.exit(0)
    df = pd.DataFrame([vars(r) for r in rows])
    df = df[df["success"] == True].copy()  # noqa: E712
    if df.empty:
        print("No successful results in", path)
        sys.exit(0)
    _fill_baseline_after_fields(df)
    df["is_train"] = df["lighting_id"] != held_out
    _add_independent_de(df)
    print(f"Loaded {len(df)} successful rows ({df['image_stem'].nunique()} images)")
    return df


def _add_independent_de(df: pd.DataFrame) -> None:
    """Add columns for fair (independent / test-set) ΔE00 evaluation.

    For each row, ``de_eval_type`` is ``"training"`` when the row's CC ΔE00
    was computed on the same swatches used to *fit* the calibration matrix,
    or ``"test"`` when the evaluation reference is independent.

    ``cc_de_eval_type`` / ``bn_de_eval_type`` flag each ΔE column separately.
    """
    cc_eval = pd.Series("test", index=df.index)
    bn_eval = pd.Series("test", index=df.index)

    # CC-trained methods: CC ΔE is training error, BN ΔE is independent
    cc_trained = df["reference_object"] == "colorchecker"
    cc_eval.loc[cc_trained] = "training"

    # BN-trained methods: BN ΔE is training error, CC ΔE is independent
    bn_trained = df["reference_object"] == "banknote"
    bn_eval.loc[bn_trained] = "training"

    df["cc_de_eval_type"] = cc_eval
    df["bn_de_eval_type"] = bn_eval


def _method_label(row):
    m = row["correction_method"]
    r = row["reference_object"]
    if m == "baseline":
        return "baseline"
    if r == "none":
        return m
    return f"{m} ({r[:2].upper()})"


def _save_table(df_table: pd.DataFrame, out: Path, name: str):
    df_table.to_csv(out / f"{name}.csv", float_format="%.2f")


def _save_fig(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png")


def _ita_std_per_group(df: pd.DataFrame) -> pd.DataFrame:
    """Compute std(ITA) across training lightings per person/hand/method."""
    train = df[df["is_train"]].dropna(subset=["ITA_after"]).copy()
    grouped = (
        train.groupby(["person_id", "hand", "hand_side",
                        "correction_method", "reference_object"])
        ["ITA_after"]
        .agg(["std", "mean", "count", list])
        .reset_index()
    )
    grouped.columns = [
        "person_id", "hand", "hand_side",
        "correction_method", "reference_object",
        "ita_std", "ita_mean", "n_lightings", "ita_values",
    ]
    grouped["ita_cv"] = grouped["ita_values"].apply(
        lambda v: coefficient_of_variation(v) if len(v) >= 2 else np.nan
    )
    return grouped


# ═══════════════════════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════════════════════


def table_t1(df: pd.DataFrame, out: Path):
    """T1: ΔE00 Summary by Method."""
    print("\n" + "=" * 72)
    print("T1: ΔE00 Summary by Method")
    print("=" * 72)
    g = df.groupby(["correction_method", "reference_object"])
    t = g.agg(
        de_before_mean=("cc_de_mean_before", "mean"),
        de_before_std=("cc_de_mean_before", "std"),
        de_after_mean=("cc_de_mean_after", "mean"),
        de_after_median=("cc_de_median_after", "median"),
        de_after_p95=("cc_de_p95_after", "mean"),
        n=("cc_de_mean_after", "count"),
    ).reset_index()
    t["improvement_pct"] = (
        (t["de_before_mean"] - t["de_after_mean"]) / t["de_before_mean"] * 100
    )
    print(t.to_string(index=False, float_format="%.2f"))
    _save_table(t, out, "T1_delta_e_summary")



def table_t1b(df: pd.DataFrame, out: Path):
    """T1b: Fair ΔE00 Comparison — independent (test-set) evaluation only.

    CC-trained methods are evaluated on BN ΔE00 (independent).
    BN-trained methods are evaluated on CC ΔE00 (independent).
    Baseline methods are evaluated on CC ΔE00 (always independent).

    This avoids the training-error bias where CC-trained CC ΔE00 is
    artificially low because the matrix was fit on those same swatches.
    """
    print("\n" + "=" * 72)
    print("T1b: Fair ΔE00 — Independent Test-Set Evaluation")
    print("=" * 72)

    rows_out = []
    for (method, ref_obj), grp in df.groupby(
        ["correction_method", "reference_object"]
    ):
        row = {"correction_method": method, "reference_object": ref_obj}

        # Pick the INDEPENDENT ΔE for this method
        if ref_obj == "colorchecker":
            # Trained on CC → CC ΔE is training error → use BN ΔE (test)
            vals = grp["bn_de_mean_after"].dropna()
            row["test_reference"] = "BN"
            row["test_de_mean"] = vals.mean() if len(vals) else None
            row["test_de_median"] = vals.median() if len(vals) else None
            row["n_test"] = len(vals)
            # Also show training error for context
            cc_vals = grp["cc_de_mean_after"].dropna()
            row["training_de_mean"] = cc_vals.mean() if len(cc_vals) else None
        elif ref_obj == "banknote":
            # Trained on BN → BN ΔE is training error → use CC ΔE (test)
            vals = grp["cc_de_mean_after"].dropna()
            row["test_reference"] = "CC"
            row["test_de_mean"] = vals.mean() if len(vals) else None
            row["test_de_median"] = vals.median() if len(vals) else None
            row["n_test"] = len(vals)
            # Also show training error for context
            bn_vals = grp["bn_de_mean_after"].dropna()
            row["training_de_mean"] = bn_vals.mean() if len(bn_vals) else None
        else:
            # Baseline methods (gray_world, shades_of_gray, baseline)
            # No reference used → both CC and BN ΔE are independent
            cc_vals = grp["cc_de_mean_after"].dropna()
            row["test_reference"] = "CC"
            row["test_de_mean"] = cc_vals.mean() if len(cc_vals) else None
            row["test_de_median"] = cc_vals.median() if len(cc_vals) else None
            row["n_test"] = len(cc_vals)
            row["training_de_mean"] = None  # no training — all independent

        rows_out.append(row)

    t = pd.DataFrame(rows_out)
    # Sort by test_de_mean ascending (best first)
    t = t.sort_values("test_de_mean", na_position="last")
    print(t.to_string(index=False, float_format="%.2f"))
    print("\n  NOTE: 'test_de_mean' uses ΔE00 on a reference NOT used for fitting.")
    print("  CC-trained → tested on BN swatches.  BN-trained → tested on CC swatches.")
    print("  'training_de_mean' is shown for context (optimistically biased).")
    _save_table(t, out, "T1b_fair_delta_e")

def table_t2(df: pd.DataFrame, out: Path):
    """T2: ΔE00 by Method × Lighting."""
    print("\n" + "=" * 72)
    print("T2: ΔE00 by Method × Lighting")
    print("=" * 72)
    pivot = df.pivot_table(
        values="cc_de_mean_after",
        index=["correction_method", "reference_object"],
        columns="lighting_id",
        aggfunc="mean",
    )
    # Reorder columns
    cols = sorted(pivot.columns, key=lambda x: (x == "L2", x))
    pivot = pivot[cols]
    print(pivot.to_string(float_format="%.2f"))
    _save_table(pivot, out, "T2_delta_e_by_lighting")


def table_t3(ita_groups: pd.DataFrame, out: Path):
    """T3: std(ITA) Across Lighting per Person × Method."""
    print("\n" + "=" * 72)
    print("T3: std(ITA) per Person × Method (training lightings)")
    print("=" * 72)
    pivot = ita_groups.pivot_table(
        values="ita_std",
        index="person_id",
        columns=["correction_method", "reference_object"],
        aggfunc="mean",
    )
    # Add mean row
    pivot.loc["Mean"] = pivot.mean()
    print(pivot.to_string(float_format="%.2f"))
    _save_table(pivot, out, "T3_std_ita")


def table_t4(ita_groups: pd.DataFrame, out: Path):
    """T4: ITA Reduction %."""
    print("\n" + "=" * 72)
    print("T4: ITA Reduction % vs Baseline")
    print("=" * 72)
    keys = ["person_id", "hand", "hand_side"]
    baseline = (
        ita_groups[ita_groups["correction_method"] == "baseline"]
        [keys + ["ita_std"]]
        .rename(columns={"ita_std": "bl_std"})
    )
    methods = ita_groups[ita_groups["correction_method"] != "baseline"].merge(
        baseline, on=keys, how="left",
    )
    methods["reduction_pct"] = methods.apply(
        lambda r: ita_reduction(r["bl_std"], r["ita_std"])
        if pd.notna(r["bl_std"]) and r["bl_std"] > 0 else np.nan,
        axis=1,
    )
    pivot = methods.pivot_table(
        values="reduction_pct",
        index="person_id",
        columns=["correction_method", "reference_object"],
        aggfunc="mean",
    )
    pivot.loc["Mean"] = pivot.mean()
    print(pivot.to_string(float_format="%.1f"))
    _save_table(pivot, out, "T4_ita_reduction")


def table_t5(ita_groups: pd.DataFrame, out: Path):
    """T5: CV(ITA) per Person × Method."""
    print("\n" + "=" * 72)
    print("T5: CV(ITA) per Person × Method")
    print("=" * 72)
    pivot = ita_groups.pivot_table(
        values="ita_cv",
        index="person_id",
        columns=["correction_method", "reference_object"],
        aggfunc="mean",
    )
    pivot.loc["Mean"] = pivot.mean()
    print(pivot.to_string(float_format="%.2f"))
    _save_table(pivot, out, "T5_cv_ita")


def table_t6(df: pd.DataFrame, out: Path):
    """T6: Chardon Consistency Rate by Method."""
    print("\n" + "=" * 72)
    print("T6: Chardon Consistency Rate (training lightings)")
    print("=" * 72)
    train = df[df["is_train"]].copy()
    rows = []
    for (method, ref), mg in train.groupby(["correction_method", "reference_object"]):
        groups = []
        for _, gg in mg.groupby(["person_id", "hand", "hand_side"]):
            cats = gg["chardon_after"].dropna().tolist()
            if len(cats) >= 2:
                groups.append(cats)
        rate = chardon_consistency_rate(groups) if groups else np.nan
        n_consistent = sum(1 for g in groups if chardon_consistency(g))
        rows.append({
            "method": method, "ref": ref,
            "consistent": n_consistent, "total": len(groups),
            "rate_pct": rate,
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format="%.1f"))
    _save_table(t, out, "T6_chardon_consistency")


def table_t7(df: pd.DataFrame, out: Path):
    """T7: Chardon Transitions — category shift magnitude across lighting."""
    print("\n" + "=" * 72)
    print("T7: Chardon Transitions (training lightings)")
    print("=" * 72)
    chardon_ord = {cat: i for i, cat in enumerate(CHARDON_CATS)}
    train = df[df["is_train"]].copy()
    rows = []
    for (method, ref), mg in train.groupby(["correction_method", "reference_object"]):
        shifts: list[int] = []
        for _, gg in mg.groupby(["person_id", "hand", "hand_side"]):
            cats = gg["chardon_after"].dropna().tolist()
            ords = [chardon_ord[c] for c in cats if c in chardon_ord]
            if len(ords) < 2:
                continue
            shifts.append(max(ords) - min(ords))
        if not shifts:
            continue
        n = len(shifts)
        n_same = sum(1 for s in shifts if s == 0)
        n_1 = sum(1 for s in shifts if s == 1)
        n_2p = sum(1 for s in shifts if s >= 2)
        rows.append({
            "method": method, "ref": ref, "n_groups": n,
            "same": n_same, "same_pct": n_same / n * 100,
            "shift_1": n_1, "shift_1_pct": n_1 / n * 100,
            "shift_2+": n_2p, "shift_2+_pct": n_2p / n * 100,
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format="%.1f"))
    _save_table(t, out, "T7_chardon_transitions")


def table_t8(df: pd.DataFrame, ita_groups: pd.DataFrame, out: Path):
    """T8: Per-Dimension Summary (best correction method only)."""
    print("\n" + "=" * 72)
    print("T8: Per-Dimension Summary")
    print("=" * 72)
    # Determine best non-baseline method by lowest mean std(ITA)
    non_bl = ita_groups[ita_groups["correction_method"] != "baseline"]
    if non_bl.empty:
        print("  No non-baseline methods available.")
        return
    ranking = non_bl.groupby(
        ["correction_method", "reference_object"]
    )["ita_std"].mean()
    best_method, best_ref = ranking.idxmin()
    print(f"  Best method: {best_method} ({best_ref})\n")

    df_best = df[
        (df["correction_method"] == best_method)
        & (df["reference_object"] == best_ref)
    ].copy()
    train_best = df_best[df_best["is_train"]]

    dimensions = [
        ("Person", "person_id"),
        ("Lighting", "lighting_id"),
        ("Hand", "hand"),
        ("Palm orientation", "hand_side"),
        ("Denomination", "denomination"),
        ("Banknote side", "banknote_side"),
    ]
    rows = []
    for dim_name, col in dimensions:
        vals = df_best[col].dropna()
        if vals.empty:
            continue
        for level in sorted(vals.unique()):
            subset = df_best[df_best[col] == level]
            row: dict = {
                "dimension": dim_name,
                "level": level,
                "de_mean_after": subset["cc_de_mean_after"].mean(),
                "n_images": len(subset),
            }
            if col == "lighting_id":
                # std(ITA) not meaningful for a single lighting condition
                row["mean_ita"] = subset["ITA_after"].mean()
                row["std_ita"] = np.nan
                row["cv_ita"] = np.nan
            else:
                train_sub = train_best[train_best[col] == level]
                stds: list[float] = []
                cvs: list[float] = []
                for _, gg in train_sub.groupby(["person_id", "hand", "hand_side"]):
                    v = gg["ITA_after"].dropna().values
                    if len(v) >= 2:
                        stds.append(float(np.std(v, ddof=1)))
                        cvs.append(coefficient_of_variation(v))
                row["mean_ita"] = subset["ITA_after"].mean()
                row["std_ita"] = float(np.mean(stds)) if stds else np.nan
                row["cv_ita"] = float(np.mean(cvs)) if cvs else np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format="%.2f"))
    _save_table(t, out, "T8_per_dimension")


def table_t8b(df: pd.DataFrame, out: Path):
    """T8b: L*, a*, b* Channel Stability across lighting."""
    print("\n" + "=" * 72)
    print("T8b: L*, a*, b* Channel Stability (training lightings)")
    print("=" * 72)
    train = df[df["is_train"]].copy()
    rows = []
    for (person, method, ref), mg in train.groupby(
        ["person_id", "correction_method", "reference_object"]
    ):
        # Average across (hand, hand_side) per lighting, then std across lightings
        per_light = mg.groupby("lighting_id").agg(
            L=("L_median_after", "mean"),
            a=("a_median_after", "mean"),
            b=("b_median_after", "mean"),
        )
        if len(per_light) < 2:
            continue
        rows.append({
            "person_id": person,
            "method": method, "ref": ref,
            "std_L": per_light["L"].std(),
            "std_a": per_light["a"].std(),
            "std_b": per_light["b"].std(),
        })
    t = pd.DataFrame(rows)
    if t.empty:
        print("  No data for channel stability.")
        _save_table(t, out, "T8b_channel_stability")
        return
    # Console: summary per method (mean across persons)
    summary = (
        t.groupby(["method", "ref"])[["std_L", "std_a", "std_b"]]
        .mean()
        .reset_index()
    )
    print(summary.to_string(index=False, float_format="%.2f"))
    _save_table(t, out, "T8b_channel_stability")


def table_t9(df: pd.DataFrame, ita_groups: pd.DataFrame, out: Path):
    """T9: CC vs Banknote head-to-head (affine only)."""
    print("\n" + "=" * 72)
    print("T9: CC vs Banknote (affine method)")
    print("=" * 72)

    affine = df[(df["correction_method"] == "affine")]
    cc = affine[affine["reference_object"] == "colorchecker"]
    bn = affine[affine["reference_object"] == "banknote"]

    if cc.empty or bn.empty:
        print("  Insufficient data for CC vs BN comparison.")
        return

    # Join on image_stem for paired comparison
    merged = cc.merge(bn, on="image_stem", suffixes=("_cc", "_bn"), how="inner")

    metrics = {}
    metrics["ΔE00 CC mean"] = (cc["cc_de_mean_after"].mean(), bn["cc_de_mean_after"].mean())
    metrics["ITA std"] = (
        ita_groups[(ita_groups["correction_method"] == "affine") &
                   (ita_groups["reference_object"] == "colorchecker")]["ita_std"].mean(),
        ita_groups[(ita_groups["correction_method"] == "affine") &
                   (ita_groups["reference_object"] == "banknote")]["ita_std"].mean(),
    )

    rows_out = []
    for name, (cc_val, bn_val) in metrics.items():
        rows_out.append({"Metric": name, "CC (affine)": cc_val, "BN (affine)": bn_val})

    # Wilcoxon on ITA
    if len(merged) >= 10:
        ita_cc = merged["ITA_after_cc"].dropna()
        ita_bn = merged["ITA_after_bn"].dropna()
        common = ita_cc.index.intersection(ita_bn.index)
        if len(common) >= 10:
            w = paired_wilcoxon(ita_cc.loc[common].values, ita_bn.loc[common].values)
            print(f"  H3 Wilcoxon: p={w.p_value:.4f}, significant={w.significant}, "
                  f"median_diff={w.median_diff:.2f}")

    t = pd.DataFrame(rows_out)
    print(t.to_string(index=False, float_format="%.3f"))
    _save_table(t, out, "T9_cc_vs_bn")


def table_t10(df: pd.DataFrame, held_out: str, out: Path):
    """T10: L2 held-out vs training lightings."""
    print("\n" + "=" * 72)
    print(f"T10: {held_out} Held-out vs Training")
    print("=" * 72)
    held = df[~df["is_train"]]
    if held.empty:
        print(f"  No {held_out} data.")
        return
    rows = []
    for (method, ref), mg in df.groupby(["correction_method", "reference_object"]):
        t_data = mg[mg["is_train"]]
        h_data = mg[~mg["is_train"]]
        if t_data.empty or h_data.empty:
            continue
        rows.append({
            "method": method, "ref": ref,
            "ΔE00_train": t_data["cc_de_mean_after"].mean(),
            "ΔE00_held": h_data["cc_de_mean_after"].mean(),
            "ITA_train": t_data["ITA_after"].mean(),
            "ITA_held": h_data["ITA_after"].mean(),
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format="%.2f"))
    _save_table(t, out, "T10_held_out")


def table_t11(ita_groups: pd.DataFrame, out: Path):
    """T11: Method Ranking by mean std(ITA)."""
    print("\n" + "=" * 72)
    print("T11: Method Ranking (lower std(ITA) = better)")
    print("=" * 72)
    agg = (
        ita_groups.groupby(["correction_method", "reference_object"])["ita_std"]
        .mean()
        .reset_index()
        .sort_values("ita_std")
    )
    agg.columns = ["method", "reference", "mean_std_ita"]
    for i, (_, r) in enumerate(agg.iterrows(), 1):
        print(f"  {i}. {r['method']:16s} ({r['reference']:14s})  std={r['mean_std_ita']:.2f}°")
    _save_table(agg, out, "T11_method_ranking")


def table_t12(ita_groups: pd.DataFrame, out: Path):
    """T12: Pairwise Wilcoxon matrix on matched std(ITA) groups."""
    print("\n" + "=" * 72)
    print("T12: Pairwise Wilcoxon (std ITA)")
    print("=" * 72)
    data = ita_groups.dropna(subset=["ita_std"]).copy()
    if data.empty:
        print("  Too few groups for pairwise Wilcoxon.")
        return
    data["method_key"] = (
        data["correction_method"] + "_" + data["reference_object"].str[:2]
    )
    pivot = data.pivot_table(
        values="ita_std",
        index=["person_id", "hand", "hand_side"],
        columns="method_key",
        aggfunc="mean",
    )
    methods = list(pivot.columns)
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            paired = pivot[[a, b]].dropna()
            if len(paired) < 5:
                continue
            w = paired_wilcoxon(paired[a].values, paired[b].values)
            rows.append({"A": a, "B": b, "p": w.p_value, "sig": w.significant,
                         "n": w.n, "median_diff": w.median_diff})
    if not rows:
        print("  Too few paired groups for pairwise Wilcoxon.")
        return
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format="%.4f"))
    _save_table(t, out, "T12_pairwise_wilcoxon")


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════


def figure_v1(df: pd.DataFrame, out: Path):
    """V1: ΔE00 box plot per method."""
    plot_df = df[df["correction_method"] != "baseline"].copy()
    plot_df["method_ref"] = plot_df.apply(_method_label, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_df, x="method_ref", y="cc_de_mean_after",
                hue="reference_object", ax=ax)
    ax.axhline(3, ls="--", color="green", alpha=0.7, label="Good (ΔE00=3)")
    ax.axhline(5, ls="--", color="red", alpha=0.7, label="Acceptable (ΔE00=5)")
    ax.set_ylabel("ΔE00 (mean per image)")
    ax.set_xlabel("Method")
    ax.set_title("V1: Color Calibration Quality (ΔE00 on CC swatches)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right")
    _save_fig(fig, out, "V1_delta_e_boxplot")


def figure_v1b(df: pd.DataFrame, out: Path):
    """V1b: Independent (test-set) ΔE00 — fair comparison across methods.

    Each method is evaluated on the reference it was NOT trained on:
      CC-trained → BN ΔE00,  BN-trained → CC ΔE00,  baselines → CC ΔE00.
    """
    plot_df = df[df["correction_method"] != "baseline"].copy()
    plot_df["method_ref"] = plot_df.apply(_method_label, axis=1)

    # Build the independent ΔE column
    de_test = pd.Series(np.nan, index=plot_df.index)
    cc_mask = plot_df["reference_object"] == "colorchecker"
    bn_mask = plot_df["reference_object"] == "banknote"
    no_mask = plot_df["reference_object"] == "none"
    de_test.loc[cc_mask] = plot_df.loc[cc_mask, "bn_de_mean_after"]
    de_test.loc[bn_mask] = plot_df.loc[bn_mask, "cc_de_mean_after"]
    de_test.loc[no_mask] = plot_df.loc[no_mask, "cc_de_mean_after"]
    plot_df["de_test"] = de_test
    plot_df = plot_df.dropna(subset=["de_test"])

    if plot_df.empty:
        print("  V1b: no independent ΔE data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_df, x="method_ref", y="de_test",
                hue="reference_object", ax=ax)
    ax.axhline(3, ls="--", color="green", alpha=0.7, label="Good (ΔE00=3)")
    ax.axhline(5, ls="--", color="red", alpha=0.7, label="Acceptable (ΔE00=5)")
    ax.set_ylabel("ΔE00 (independent test-set)")
    ax.set_xlabel("Method")
    ax.set_title(
        "V1b: Fair ΔE00 — CC-trained tested on BN, BN-trained tested on CC"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right")
    _save_fig(fig, out, "V1b_independent_delta_e")



def figure_v1c(df: pd.DataFrame, out: Path):
    """V1c: CC ΔE00 for all methods — annotated training vs test.

    Every method is evaluated against CC swatches (same reference),
    but CC-trained rows are hatched to flag training-error bias.
    """
    plot_df = df[df["correction_method"] != "baseline"].copy()
    plot_df = plot_df.dropna(subset=["cc_de_mean_after"])
    plot_df["method_ref"] = plot_df.apply(_method_label, axis=1)
    plot_df["eval_type"] = plot_df["cc_de_eval_type"].map(
        {"training": "training error", "test": "independent test"}
    )

    if plot_df.empty:
        print("  V1c: no CC ΔE data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Combine but use eval_type as hue so boxes are side-by-side
    sns.boxplot(
        data=plot_df,
        x="method_ref",
        y="cc_de_mean_after",
        hue="eval_type",
        palette={"independent test": "#4C72B0", "training error": "#CCCCCC"},
        ax=ax,
    )

    # Add hatching to training-error boxes
    n_methods = plot_df["method_ref"].nunique()
    for i, patch in enumerate(ax.patches):
        if i >= n_methods:  # second hue group = training error
            patch.set_hatch("//")
            patch.set_edgecolor("#888888")

    ax.axhline(3, ls="--", color="green", alpha=0.7, label="Good (ΔE00=3)")
    ax.axhline(5, ls="--", color="red", alpha=0.7, label="Acceptable (ΔE00=5)")
    ax.set_ylabel("ΔE00 on CC swatches (mean per image)")
    ax.set_xlabel("Method")
    ax.set_title(
        "V1c: CC ΔE00 — all methods on same reference "
        "(hatched = training error, solid = independent test)"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right")
    _save_fig(fig, out, "V1c_cc_delta_e_annotated")

def figure_v2(df: pd.DataFrame, held_out: str, out: Path):
    """V2: ITA across lighting — KEY FIGURE."""
    train = df[df["is_train"]].copy()
    # Average across hand/hand_side per (person, lighting, method, ref)
    agg = train.groupby(
        ["person_id", "lighting_id", "correction_method", "reference_object"]
    )["ITA_after"].mean().reset_index()

    persons = sorted(agg["person_id"].dropna().unique())
    n_persons = len(persons)
    if n_persons == 0:
        return

    fig, axes = plt.subplots(1, n_persons, figsize=(4 * n_persons, 5), sharey=True)
    if n_persons == 1:
        axes = [axes]

    refs_to_plot = [("affine", "colorchecker"), ("affine", "banknote"),
                    ("baseline", "none"), ("gray_world", "none")]

    for ax, pid in zip(axes, persons):
        pdata = agg[agg["person_id"] == pid]
        for method, ref in refs_to_plot:
            subset = pdata[(pdata["correction_method"] == method) &
                           (pdata["reference_object"] == ref)]
            if subset.empty:
                continue
            subset = subset.sort_values("lighting_id")
            label = f"{method} ({ref[:2]})" if ref != "none" else method
            ax.plot(subset["lighting_id"], subset["ITA_after"], "o-", label=label, markersize=4)
        ax.set_title(f"Person {pid}")
        ax.set_xlabel("Lighting")
        if ax == axes[0]:
            ax.set_ylabel("ITA (°)")
            ax.legend(fontsize=7)

    fig.suptitle("V2: ITA Across Lighting Conditions (training set)", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out, "V2_ita_across_lighting")


def figure_v3(ita_groups: pd.DataFrame, out: Path):
    """V3: std(ITA) heatmap — persons × methods."""
    pivot = ita_groups.pivot_table(
        values="ita_std", index="person_id",
        columns=["correction_method", "reference_object"], aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2), 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", ax=ax,
                cmap=sns.diverging_palette(120, 10, s=80, l=55, as_cmap=True),
                center=5, vmin=0, vmax=15, linewidths=0.5)
    ax.set_title("V3: std(ITA) Across Training Lightings (°)")
    ax.set_ylabel("Person")
    _save_fig(fig, out, "V3_std_ita_heatmap")


def figure_v4(df: pd.DataFrame, out: Path):
    """V4: Chardon category heatmap."""
    train = df[df["is_train"]].copy()
    # Aggregate to most common chardon per (person, method, ref, lighting)
    agg = (
        train.groupby(["person_id", "correction_method", "reference_object", "lighting_id"])
        ["chardon_after"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
        .reset_index()
    )
    # Focus on key methods
    key_methods = ["baseline", "affine", "gray_world"]
    agg = agg[agg["correction_method"].isin(key_methods)]
    agg["row_label"] = agg.apply(
        lambda r: f"P{int(r['person_id'])}-{r['correction_method']}"
        + (f"({r['reference_object'][:2]})" if r["reference_object"] != "none" else ""),
        axis=1,
    )

    pivot = agg.pivot_table(values="chardon_after", index="row_label",
                            columns="lighting_id", aggfunc="first")
    # Map to numeric for coloring
    cat_to_num = {c: i for i, c in enumerate(CHARDON_CATS)}
    numeric = pivot.map(lambda x: cat_to_num.get(x, np.nan) if pd.notna(x) else np.nan)

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    cmap = sns.color_palette([CHARDON_CMAP[c] for c in CHARDON_CATS], as_cmap=True)
    sns.heatmap(numeric, annot=pivot.values, fmt="", ax=ax, cmap=cmap,
                vmin=0, vmax=5, linewidths=0.5, cbar=False)
    ax.set_title("V4: Chardon Category Stability Across Lighting")
    _save_fig(fig, out, "V4_chardon_heatmap")


def figure_v5(df: pd.DataFrame, out: Path):
    """V5: Before/After ITA scatter."""
    plot_df = df[df["correction_method"] != "baseline"].copy()
    methods = [m for m in METHODS_ORDER if m != "baseline" and m in plot_df["correction_method"].unique()]
    n = len(methods)
    if n == 0:
        return

    fig, axes = plt.subplots(1, min(n, 6), figsize=(5 * min(n, 6), 5), sharey=True, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, method in zip(axes, methods[:6]):
        subset = plot_df[plot_df["correction_method"] == method]
        ax.scatter(subset["ITA_before"], subset["ITA_after"], alpha=0.3, s=8,
                   c=subset["lighting_id"].map(
                       {f"L{i}": plt.cm.tab10(i / 6) for i in range(1, 7)}
                   ))
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
        ax.set_title(method)
        ax.set_xlabel("ITA before (°)")
        if ax == axes[0]:
            ax.set_ylabel("ITA after (°)")

    fig.suptitle("V5: Before vs After ITA", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out, "V5_ita_before_after")


def figure_v6(df: pd.DataFrame, out: Path):
    """V6: ΔE00 heatmap method × lighting."""
    pivot = df.pivot_table(
        values="cc_de_mean_after",
        index=["correction_method", "reference_object"],
        columns="lighting_id",
        aggfunc="mean",
    )
    cols = sorted(pivot.columns, key=lambda x: (x == "L2", x))
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".1f", ax=ax, cmap="YlOrRd",
                linewidths=0.5, vmin=0)
    ax.set_title("V6: ΔE00 After Calibration (Method × Lighting)")
    _save_fig(fig, out, "V6_delta_e_heatmap")


def figure_v7(df: pd.DataFrame, out: Path):
    """V7: Per-dimension ITA box plots."""
    dims = ["person_id", "hand", "hand_side", "denomination", "banknote_side"]
    available = [d for d in dims if d in df.columns and df[d].notna().any()]
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, dim in zip(axes, available):
        plot_data = df[df[dim].notna()].copy()
        plot_data[dim] = plot_data[dim].astype(str)
        sns.boxplot(data=plot_data, x=dim, y="ITA_after", ax=ax)
        ax.set_title(dim)
        ax.set_ylabel("ITA (°)" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("V7: ITA Distribution by Dataset Dimension", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out, "V7_per_dimension")


def figure_v8(df: pd.DataFrame, out: Path):
    """V8: CC vs BN ITA scatter (affine, paired by image)."""
    cc = df[(df["correction_method"] == "affine") &
            (df["reference_object"] == "colorchecker")][["image_stem", "ITA_after", "person_id"]]
    bn = df[(df["correction_method"] == "affine") &
            (df["reference_object"] == "banknote")][["image_stem", "ITA_after"]]

    if cc.empty or bn.empty:
        print("  V8: Insufficient data for CC vs BN scatter.")
        return

    merged = cc.merge(bn, on="image_stem", suffixes=("_cc", "_bn"))
    if len(merged) < 5:
        print("  V8: Too few paired points.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(merged["ITA_after_cc"], merged["ITA_after_bn"],
                         alpha=0.5, s=15, c=merged["person_id"].astype(float),
                         cmap="tab10")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.4, lw=1, label="y=x")

    # Regression
    x, y = merged["ITA_after_cc"].values, merged["ITA_after_bn"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 5:
        slope, intercept, r, p, _ = sp_stats.linregress(x[mask], y[mask])
        rho, rho_p = sp_stats.spearmanr(x[mask], y[mask])
        xs = np.linspace(lims[0], lims[1], 50)
        ax.plot(xs, slope * xs + intercept, "b-", alpha=0.6, lw=1.5)
        ax.text(0.05, 0.92, f"Pearson r={r:.3f} (p={p:.1e})\nSpearman ρ={rho:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("ITA after CC calibration (°)")
    ax.set_ylabel("ITA after Banknote calibration (°)")
    ax.set_title("V8: ColorChecker vs Banknote ITA (affine)")
    plt.colorbar(scatter, label="Person ID")
    _save_fig(fig, out, "V8_cc_vs_bn_scatter")



def figure_v10(df: pd.DataFrame, out: Path):
    """V10: Per-Denomination Banknote Performance."""
    bn = df[df["reference_object"] == "banknote"].copy()
    if bn.empty or "denomination" not in bn.columns:
        print("  V10: No banknote data.")
        return
    bn["denomination"] = bn["denomination"].astype(str) + "€"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ΔE00
    sns.boxplot(data=bn, x="denomination", y="cc_de_mean_after",
                order=["5€", "10€", "20€", "50€", "100€"], ax=axes[0])
    axes[0].axhline(3, ls="--", color="green", alpha=0.7)
    axes[0].axhline(5, ls="--", color="red", alpha=0.7)
    axes[0].set_ylabel("ΔE00 (mean per image)")
    axes[0].set_title("Calibration Quality by Denomination")

    # Right: ITA_after spread
    sns.boxplot(data=bn, x="denomination", y="ITA_after",
                order=["5€", "10€", "20€", "50€", "100€"], ax=axes[1])
    axes[1].set_ylabel("ITA after (°)")
    axes[1].set_title("ITA After Calibration by Denomination")

    fig.suptitle("V10: Per-Denomination Banknote Performance", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out, "V10_denomination_performance")


def figure_v11(df: pd.DataFrame, out: Path):
    """V11: Dorsal vs Palm ITA comparison."""
    if "hand_side" not in df.columns or df["hand_side"].isna().all():
        print("  V11: No hand_side data.")
        return
    plot_df = df[df["hand_side"].isin(["dorsal", "palm"])].copy()
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ITA by person × hand_side
    sns.boxplot(data=plot_df, x="person_id", y="ITA_after", hue="hand_side",
                ax=axes[0])
    axes[0].set_xlabel("Person")
    axes[0].set_ylabel("ITA after (°)")
    axes[0].set_title("ITA by Person × Hand Side")
    axes[0].legend(title="Side")

    # Right: ITA range (max-min across lighting) per hand_side
    train = plot_df[plot_df["is_train"]]
    ranges = (
        train.groupby(["person_id", "hand_side", "correction_method",
                        "reference_object"])["ITA_after"]
        .agg(ita_range=lambda x: x.max() - x.min())
        .reset_index()
    )
    sns.boxplot(data=ranges, x="hand_side", y="ita_range", ax=axes[1])
    axes[1].set_xlabel("Hand Side")
    axes[1].set_ylabel("ITA range across lighting (°)")
    axes[1].set_title("ITA Variability: Dorsal vs Palm")

    fig.suptitle("V11: Dorsal vs Palm ITA Comparison", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out, "V11_dorsal_vs_palm")


def figure_v12(df: pd.DataFrame, held_out: str, out: Path):
    """V12: L2 held-out overlay on ITA-across-lighting plot."""
    # Average across hand/hand_side per (person, lighting, method, ref)
    agg = df.groupby(
        ["person_id", "lighting_id", "correction_method", "reference_object"]
    )["ITA_after"].mean().reset_index()

    persons = sorted(agg["person_id"].dropna().unique())
    n_persons = len(persons)
    if n_persons == 0:
        return

    fig, axes = plt.subplots(1, n_persons, figsize=(4 * n_persons, 5), sharey=True)
    if n_persons == 1:
        axes = [axes]

    refs_to_plot = [("affine", "colorchecker"), ("affine", "banknote"),
                    ("baseline", "none"), ("gray_world", "none")]

    for ax, pid in zip(axes, persons):
        pdata = agg[agg["person_id"] == pid]
        for method, ref in refs_to_plot:
            subset = pdata[(pdata["correction_method"] == method) &
                           (pdata["reference_object"] == ref)]
            if subset.empty:
                continue
            label = f"{method} ({ref[:2]})" if ref != "none" else method

            train_pts = subset[subset["lighting_id"] != held_out].sort_values("lighting_id")
            held_pts = subset[subset["lighting_id"] == held_out]

            line = ax.plot(train_pts["lighting_id"], train_pts["ITA_after"],
                           "o-", label=label, markersize=4)
            if not held_pts.empty:
                colour = line[0].get_color()
                ax.plot(held_pts["lighting_id"], held_pts["ITA_after"],
                        marker="*", markersize=12, color=colour, linestyle="none",
                        zorder=5)

        ax.set_title(f"Person {pid}")
        ax.set_xlabel("Lighting")
        if ax == axes[0]:
            ax.set_ylabel("ITA (°)")
            ax.legend(fontsize=7)

    # Legend entry for the star marker
    axes[-1].plot([], [], marker="*", markersize=12, color="gray",
                  linestyle="none", label=f"{held_out} (held-out)")
    axes[-1].legend(fontsize=7)

    fig.suptitle(f"V12: ITA Across Lighting (★ = {held_out} held-out)", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out, "V12_held_out_overlay")


def figure_v15(df: pd.DataFrame, out: Path):
    """V15: Interaction heatmap — Denomination × Lighting (banknote ΔE00)."""
    bn = df[df["reference_object"] == "banknote"].copy()
    if bn.empty or "denomination" not in bn.columns:
        print("  V15: No banknote data.")
        return

    pivot = bn.pivot_table(
        values="cc_de_mean_after",
        index="denomination",
        columns="lighting_id",
        aggfunc="mean",
    )
    cols = sorted(pivot.columns, key=lambda x: (x == "L2", x))
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", ax=ax, cmap="YlOrRd",
                linewidths=0.5, vmin=0)
    ax.set_ylabel("Denomination (€)")
    ax.set_xlabel("Lighting")
    ax.set_title("V15: Banknote ΔE00 — Denomination × Lighting")
    _save_fig(fig, out, "V15_denomination_x_lighting")


# ═══════════════════════════════════════════════════════════════════════════
# Interaction Effects (A.5.8)
# ═══════════════════════════════════════════════════════════════════════════


def table_interactions(df: pd.DataFrame, ita_groups: pd.DataFrame, out: Path):
    """A.5.8: Interaction effects — 4 analyses in one table set."""
    print("\n" + "=" * 72)
    print("Interaction Effects (A.5.8)")
    print("=" * 72)

    # --- Interaction 1: Denomination × Lighting (ΔE00) ---
    print("\n  Interaction 1: Denomination × Lighting")
    bn = df[(df["reference_object"] == "banknote") & df["is_train"]].copy()
    if not bn.empty and "denomination" in bn.columns:
        i1 = bn.pivot_table(
            values="cc_de_mean_after",
            index="denomination",
            columns="lighting_id",
            aggfunc="mean",
        )
        print(i1.to_string(float_format="%.2f"))
        _save_table(i1, out, "I1_denomination_x_lighting")
    else:
        print("    No banknote data.")

    # --- Interaction 2: Person (skin tone) × Method (CV ITA) ---
    print("\n  Interaction 2: Person × Method (CV ITA)")
    if not ita_groups.empty:
        i2 = ita_groups.pivot_table(
            values="ita_cv",
            index="person_id",
            columns=["correction_method", "reference_object"],
            aggfunc="mean",
        )
        print(i2.to_string(float_format="%.2f"))
        _save_table(i2, out, "I2_person_x_method_cv")

    # --- Interaction 3: Palm orientation × ITA ---
    print("\n  Interaction 3: Palm orientation × ITA")
    if "hand_side" in df.columns:
        train = df[df["is_train"]].copy()
        i3 = (
            train.groupby(["hand_side", "correction_method", "reference_object"])
            .agg(mean_ita=("ITA_after", "mean"), std_ita=("ITA_after", "std"),
                 n=("ITA_after", "count"))
            .reset_index()
        )
        print(i3.to_string(index=False, float_format="%.2f"))
        _save_table(i3, out, "I3_palm_orientation")

    # --- Interaction 4: Denomination × Banknote side ---
    print("\n  Interaction 4: Denomination × Banknote side")
    if not bn.empty and "banknote_side" in bn.columns:
        i4 = bn.pivot_table(
            values=["cc_de_mean_after", "bn_n_samples"],
            index="denomination",
            columns="banknote_side",
            aggfunc="mean",
        )
        print(i4.to_string(float_format="%.2f"))
        _save_table(i4, out, "I4_denomination_x_side")
    else:
        print("    No banknote data.")

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline results.")
    parser.add_argument("csv", help="Path to results.csv")
    parser.add_argument("--output-dir", default="results/evaluation",
                        help="Output directory for tables and figures")
    parser.add_argument("--held-out", default="L2",
                        help="Held-out lighting condition (default: L2)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}/")

    df = _load(args.csv, args.held_out)
    ita_groups = _ita_std_per_group(df)

    # ── Tables ─────────────────────────────────────────────────────
    print("\n\n### TABLES ###\n")
    table_t1(df, out)
    table_t1b(df, out)
    table_t2(df, out)
    table_t3(ita_groups, out)
    table_t4(ita_groups, out)
    table_t5(ita_groups, out)
    table_t6(df, out)
    table_t7(df, out)
    table_t8(df, ita_groups, out)
    table_t8b(df, out)
    table_t9(df, ita_groups, out)
    table_t10(df, args.held_out, out)
    table_t11(ita_groups, out)
    table_t12(ita_groups, out)
    table_interactions(df, ita_groups, out)

    # ── Figures ────────────────────────────────────────────────────
    print("\n\n### FIGURES ###\n")
    figure_v1(df, out)
    figure_v1b(df, out)
    figure_v1c(df, out)
    figure_v2(df, args.held_out, out)
    figure_v3(ita_groups, out)
    figure_v4(df, out)
    figure_v5(df, out)
    figure_v6(df, out)
    figure_v7(df, out)
    figure_v8(df, out)
    figure_v10(df, out)
    figure_v11(df, out)
    figure_v12(df, args.held_out, out)
    figure_v15(df, out)

    print(f"\nDone. Tables and figures saved to {out}/")


if __name__ == "__main__":
    main()
