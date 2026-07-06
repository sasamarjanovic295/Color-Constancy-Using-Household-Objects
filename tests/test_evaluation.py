"""
Tests for src/evaluation.py — all evaluation metrics.

Each test section covers one metric category from the evaluation strategy.
Tests use deterministic synthetic data; no image I/O.
"""

import json

import colour
import numpy as np
import pytest

from src.evaluation import (
    CrossLightingStats,
    DeltaEComponents,
    DeltaEStats,
    ImageResult,
    WilcoxonResult,
    angular_error,
    angular_error_stats,
    chardon_consistency,
    chardon_consistency_rate,
    classify_chardon,
    compute_colorfulness,
    compute_delta_e00,
    compute_delta_e00_mixed,
    compute_ita,
    compute_psnr,
    compute_ssim,
    cross_lighting_std,
    delta_e_components,
    delta_e_stats,
    fill_cc_delta_e,
    fill_skin_tone,
    improvement_over_baseline,
    ita_reduction,
    method_comparison_matrix,
    method_ranking,
    paired_wilcoxon,
    pairwise_wilcoxon_matrix,
    read_results_csv,
    reproduction_angular_error,
    write_results_csv,
    write_results_json,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sck300_reference_xyz_d50():
    """First 4 SCK300 reference XYZ values (D50) from reference_data.py."""
    from src.reference_data import SCK300_REFERENCE_XYZ
    return SCK300_REFERENCE_XYZ[:4].copy()


@pytest.fixture
def identical_xyz():
    """A pair of identical XYZ(D65) arrays — ΔE should be ~0."""
    xyz = np.array([
        [0.20, 0.21, 0.18],
        [0.50, 0.52, 0.55],
        [0.10, 0.11, 0.09],
    ], dtype=np.float64)
    return xyz, xyz.copy()


# ═══════════════════════════════════════════════════════════════════════════
# 1. ΔE00
# ═══════════════════════════════════════════════════════════════════════════


class TestDeltaE00:

    def test_identical_colors_give_zero(self, identical_xyz):
        a, b = identical_xyz
        de = compute_delta_e00(a, b)
        assert de.shape == (3,)
        np.testing.assert_allclose(de, 0.0, atol=1e-10)

    def test_different_colors_give_positive(self):
        a = np.array([[0.50, 0.50, 0.50]], dtype=np.float64)
        b = np.array([[0.30, 0.30, 0.30]], dtype=np.float64)
        de = compute_delta_e00(a, b)
        assert de.shape == (1,)
        assert de[0] > 0

    def test_mixed_illuminant_matches_uniform(self, sck300_reference_xyz_d50):
        """compute_delta_e00_mixed (measured=D65, ref=D50) should equal
        compute_delta_e00 when measured is adapted from the same D50 value."""
        from src.evaluation import _adapt_d65_to_d50  # noqa: F401
        ref_d50 = sck300_reference_xyz_d50
        # "Measured" is ref adapted to D65 — so it should be near-identical
        adapted_d65 = colour.adaptation.chromatic_adaptation_VonKries(
            ref_d50,
            colour.xy_to_XYZ(colour.CCS_ILLUMINANTS[
                "CIE 1931 2 Degree Standard Observer"]["D50"]),
            colour.xy_to_XYZ(colour.CCS_ILLUMINANTS[
                "CIE 1931 2 Degree Standard Observer"]["D65"]),
            transform="Bradford",
        )
        de = compute_delta_e00_mixed(adapted_d65, ref_d50)
        # Round-trip through Bradford should give near-zero ΔE
        np.testing.assert_allclose(de, 0.0, atol=0.1)


class TestDeltaEStats:

    def test_all_zeros(self):
        de = np.zeros(10)
        s = delta_e_stats(de)
        assert isinstance(s, DeltaEStats)
        assert s.mean == 0.0
        assert s.max == 0.0
        assert s.trimean == 0.0
        assert len(s.per_patch) == 10

    def test_known_distribution(self):
        de = np.arange(1.0, 9.0)  # [1, 2, 3, 4, 5, 6, 7, 8]
        s = delta_e_stats(de)
        assert s.mean == pytest.approx(4.5)
        assert s.median == pytest.approx(4.5)
        assert s.max == pytest.approx(8.0)
        # Q1=2.75, Q3=6.25 → trimean = (2.75 + 9.0 + 6.25) / 4 = 4.5
        assert s.trimean == pytest.approx(4.5, abs=0.1)
        # best25: mean of values <= Q1=2.75 → mean([1, 2]) = 1.5
        assert s.best25 == pytest.approx(1.5)
        # worst25: mean of values >= Q3=6.25 → mean([7, 8]) = 7.5
        assert s.worst25 == pytest.approx(7.5)

    def test_per_patch_preserves_values(self):
        de = np.array([1.0, 5.0, 10.0])
        s = delta_e_stats(de)
        assert s.per_patch == [1.0, 5.0, 10.0]


# ═══════════════════════════════════════════════════════════════════════════
# 2. ΔE00 components
# ═══════════════════════════════════════════════════════════════════════════


class TestDeltaEComponents:

    def test_identical_gives_zero_components(self, identical_xyz):
        a, b = identical_xyz
        c = delta_e_components(a, b)
        assert isinstance(c, DeltaEComponents)
        np.testing.assert_allclose(c.delta_L, [0, 0, 0], atol=1e-8)
        np.testing.assert_allclose(c.delta_C, [0, 0, 0], atol=1e-8)
        np.testing.assert_allclose(c.delta_h, [0, 0, 0], atol=1e-8)

    def test_lightness_change_dominates(self):
        """When only lightness changes, ΔL should dominate."""
        # Two colors differing mainly in L*
        a = np.array([[0.20, 0.21, 0.22]], dtype=np.float64)
        b = np.array([[0.40, 0.42, 0.44]], dtype=np.float64)
        c = delta_e_components(a, b)
        assert abs(c.delta_L[0]) > abs(c.delta_C[0])


# ═══════════════════════════════════════════════════════════════════════════
# 3. Angular error
# ═══════════════════════════════════════════════════════════════════════════


class TestAngularError:

    def test_identical_vectors(self):
        v = np.array([1.0, 1.0, 1.0])
        assert angular_error(v, v) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert angular_error(a, b) == pytest.approx(90.0, abs=1e-8)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert angular_error(a, b) == pytest.approx(180.0, abs=1e-8)

    def test_scale_invariance(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])  # same direction, 2x magnitude
        assert angular_error(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_zero_vector_gives_180(self):
        assert angular_error(np.zeros(3), np.ones(3)) == 180.0

    def test_reproduction_angular_error_perfect(self):
        """If estimated illuminant = ground truth, RAE should be 0."""
        illum = np.array([0.8, 1.0, 1.2])
        assert reproduction_angular_error(illum, illum) == pytest.approx(0.0, abs=1e-8)

    def test_reproduction_angular_error_positive(self):
        e = np.array([0.8, 1.0, 1.2])
        g = np.array([1.0, 1.0, 1.0])
        rae = reproduction_angular_error(e, g)
        assert rae > 0


class TestAngularErrorStats:

    def test_known_values(self):
        errors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        s = angular_error_stats(errors)
        assert s["mean"] == pytest.approx(4.5)
        assert s["median"] == pytest.approx(4.5)
        assert "trimean" in s
        assert "best25" in s
        assert "worst25" in s
        assert "p95" in s


# ═══════════════════════════════════════════════════════════════════════════
# 4. SSIM / PSNR
# ═══════════════════════════════════════════════════════════════════════════


class TestSSIMPSNR:

    def test_ssim_identical(self):
        image_srgb = np.random.rand(64, 64, 3).astype(np.float32)
        assert compute_ssim(image_srgb, image_srgb) == pytest.approx(1.0, abs=1e-6)

    def test_ssim_different(self):
        a_srgb = np.zeros((64, 64, 3), dtype=np.float32)
        b_srgb = np.ones((64, 64, 3), dtype=np.float32)
        ssim = compute_ssim(a_srgb, b_srgb)
        assert ssim < 0.1  # very different

    def test_psnr_identical(self):
        image_srgb = np.random.rand(64, 64, 3).astype(np.float32)
        assert compute_psnr(image_srgb, image_srgb) == float("inf")

    def test_psnr_different(self):
        a_srgb = np.zeros((64, 64, 3), dtype=np.float32)
        b_srgb = np.ones((64, 64, 3), dtype=np.float32)
        psnr = compute_psnr(a_srgb, b_srgb)
        assert psnr == pytest.approx(0.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Colorfulness
# ═══════════════════════════════════════════════════════════════════════════


class TestColorfulness:

    def test_gray_image_low_colorfulness(self):
        gray = np.full((32, 32, 3), 0.5, dtype=np.float32)
        assert compute_colorfulness(gray) == pytest.approx(0.0, abs=1e-10)

    def test_colorful_image_higher(self):
        # Image with varied saturated colours — high σ in rg/yb
        np.random.seed(42)
        img = np.random.rand(64, 64, 3).astype(np.float32)
        img[:32, :, 0] = 1.0  # top half red-ish
        img[32:, :, 2] = 1.0  # bottom half blue-ish
        m = compute_colorfulness(img)
        gray = np.full((64, 64, 3), 0.5, dtype=np.float32)
        m_gray = compute_colorfulness(gray)
        assert m > m_gray  # varied image is more colourful than gray
        assert m > 0.1


# ═══════════════════════════════════════════════════════════════════════════
# 6. ITA and Chardon
# ═══════════════════════════════════════════════════════════════════════════


class TestITA:

    def test_formula(self):
        # ITA = arctan2(L*-50, b*) × 180/π
        # L*=70, b*=20 → arctan2(20, 20) = 45°
        ita = compute_ita(70.0, 20.0)
        assert ita == pytest.approx(45.0, abs=1e-8)

    def test_zero_b(self):
        # L*=80, b*=0 → arctan2(30, 0) = 90°
        ita = compute_ita(80.0, 0.0)
        assert ita == pytest.approx(90.0, abs=1e-8)

    def test_negative_ita(self):
        # L*=40, b*=20 → arctan2(-10, 20) ≈ -26.6°
        ita = compute_ita(40.0, 20.0)
        assert ita < 0


class TestChardon:

    def test_very_light(self):
        cat, fp = classify_chardon(60.0)
        assert cat == "Very Light"
        assert fp == 1

    def test_light(self):
        cat, fp = classify_chardon(45.0)
        assert cat == "Light"
        assert fp == 2

    def test_intermediate(self):
        cat, fp = classify_chardon(35.0)
        assert cat == "Intermediate"
        assert fp == 3

    def test_tan(self):
        cat, fp = classify_chardon(15.0)
        assert cat == "Tan"
        assert fp == 4

    def test_brown(self):
        cat, fp = classify_chardon(0.0)
        assert cat == "Brown"
        assert fp == 5

    def test_dark(self):
        cat, fp = classify_chardon(-40.0)
        assert cat == "Dark"
        assert fp == 6

    def test_boundary_values(self):
        # Exactly on threshold — ">55" means 55.0 is NOT Very Light
        cat, _ = classify_chardon(55.0)
        assert cat == "Light"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Cross-lighting stability
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossLighting:

    def test_constant_values(self):
        s = cross_lighting_std([42.0, 42.0, 42.0, 42.0, 42.0])
        assert isinstance(s, CrossLightingStats)
        assert s.std == pytest.approx(0.0)
        assert s.range == pytest.approx(0.0)
        assert s.n == 5

    def test_varying_values(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        s = cross_lighting_std(vals)
        assert s.mean == pytest.approx(30.0)
        assert s.std > 0
        assert s.range == pytest.approx(40.0)

    def test_ita_reduction_positive(self):
        assert ita_reduction(10.0, 5.0) == pytest.approx(50.0)

    def test_ita_reduction_negative(self):
        # After is worse
        assert ita_reduction(5.0, 10.0) == pytest.approx(-100.0)

    def test_ita_reduction_zero_before(self):
        assert ita_reduction(0.0, 5.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 8. Chardon consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestChardonConsistency:

    def test_all_same(self):
        assert chardon_consistency(["Light", "Light", "Light"]) is True

    def test_mixed(self):
        assert chardon_consistency(["Light", "Intermediate", "Light"]) is False

    def test_empty(self):
        assert chardon_consistency([]) is True

    def test_rate_all_consistent(self):
        groups = [
            ["Light", "Light", "Light"],
            ["Tan", "Tan", "Tan"],
        ]
        assert chardon_consistency_rate(groups) == pytest.approx(100.0)

    def test_rate_half_consistent(self):
        groups = [
            ["Light", "Light", "Light"],
            ["Light", "Intermediate", "Light"],
        ]
        assert chardon_consistency_rate(groups) == pytest.approx(50.0)

    def test_rate_empty(self):
        assert chardon_consistency_rate([]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 9. Paired statistical tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWilcoxon:

    def test_identical_gives_no_significance(self):
        vals = list(range(1, 21))  # 20 values
        w = paired_wilcoxon(vals, vals)
        # All differences are zero — too few nonzero to test
        assert not w.significant

    def test_clearly_different(self):
        a = [10.0 + i for i in range(20)]
        b = [5.0 + i for i in range(20)]  # consistently lower
        w = paired_wilcoxon(a, b)
        assert isinstance(w, WilcoxonResult)
        assert w.n == 20
        assert w.significant is True
        assert w.median_diff == pytest.approx(5.0)

    def test_too_few_samples(self):
        w = paired_wilcoxon([1, 2, 3], [4, 5, 6])
        assert w.n < 10
        assert not w.significant  # not enough data


# ═══════════════════════════════════════════════════════════════════════════
# 10. Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestAggregation:

    def test_improvement_over_baseline(self):
        # baseline=10, method=5 → 50% improvement
        assert improvement_over_baseline(10.0, 5.0) == pytest.approx(50.0)

    def test_improvement_negative(self):
        # method is worse
        assert improvement_over_baseline(5.0, 10.0) == pytest.approx(-100.0)

    def test_improvement_zero_baseline(self):
        assert improvement_over_baseline(0.0, 5.0) == 0.0

    def test_method_comparison_matrix(self):
        results = {
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 3.0, 4.0],
        }
        m = method_comparison_matrix(results)
        # A is consistently 1.0 lower than B
        assert m[("A", "B")] == pytest.approx(-1.0)
        assert m[("B", "A")] == pytest.approx(1.0)

    def test_method_ranking(self):
        results = {
            "worst": [10.0, 20.0],
            "best": [1.0, 2.0],
            "mid": [5.0, 6.0],
        }
        ranking = method_ranking(results)
        assert ranking[0][0] == "best"
        assert ranking[-1][0] == "worst"

    def test_pairwise_wilcoxon_matrix(self):
        results = {
            "A": [10.0 + i for i in range(20)],
            "B": [5.0 + i for i in range(20)],
        }
        m = pairwise_wilcoxon_matrix(results)
        assert ("A", "B") in m
        assert ("B", "A") in m
        assert m[("A", "B")].median_diff == pytest.approx(5.0)
        assert m[("B", "A")].median_diff == pytest.approx(-5.0)


# ═══════════════════════════════════════════════════════════════════════════
# 11. I/O — CSV and JSON
# ═══════════════════════════════════════════════════════════════════════════


class TestIO:

    def test_csv_round_trip(self, tmp_path):
        records = [
            ImageResult(
                image_path="test.jpg",
                image_stem="test",
                lighting_id="L1",
                person_id=1,
                hand="left",
                reference_object="colorchecker",
                correction_method="affine",
                cc_de_mean_before=5.5,
                cc_de_mean_after=2.1,
                ITA_before=42.3,
                chardon_before="Light",
                success=True,
                skin_measured=True,
            ),
            ImageResult(
                image_path="test2.jpg",
                image_stem="test2",
                success=False,
                failure_reason="detection: no quad",
            ),
        ]
        csv_path = tmp_path / "results.csv"
        write_results_csv(records, csv_path)
        loaded = read_results_csv(csv_path)
        assert len(loaded) == 2
        assert loaded[0].image_stem == "test"
        assert loaded[0].cc_de_mean_before == pytest.approx(5.5)
        assert loaded[0].cc_de_mean_after == pytest.approx(2.1)
        assert loaded[0].ITA_before == pytest.approx(42.3)
        assert loaded[0].success is True
        assert loaded[0].skin_measured is True
        assert loaded[1].success is False
        assert loaded[1].failure_reason == "detection: no quad"

    def test_csv_person_id_round_trips_as_int(self, tmp_path):
        records = [
            ImageResult(
                image_path="a.jpg", image_stem="a",
                person_id=3, success=True,
            ),
        ]
        csv_path = tmp_path / "round_trip.csv"
        write_results_csv(records, csv_path)
        loaded = read_results_csv(csv_path)
        assert loaded[0].person_id == 3
        assert isinstance(loaded[0].person_id, int)

    def test_json_output(self, tmp_path):
        records = [
            ImageResult(image_path="a.jpg", image_stem="a", success=True),
        ]
        json_path = tmp_path / "results.json"
        write_results_json(records, json_path)
        data = json.loads(json_path.read_text())
        assert len(data) == 1
        assert data[0]["image_stem"] == "a"


# ═══════════════════════════════════════════════════════════════════════════
# 12. Convenience — fill helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestFillHelpers:

    def test_fill_cc_delta_e(self):
        r = ImageResult(image_path="x.jpg", image_stem="x")
        de = DeltaEStats(
            mean=5.0, median=4.0, trimean=4.5, p95=8.0,
            max=10.0, best25=2.0, worst25=9.0, per_patch=[],
        )
        fill_cc_delta_e(r, de, None)
        assert r.cc_de_mean_before == 5.0
        assert r.cc_de_trimean_before == 4.5
        assert r.cc_de_mean_after is None  # not filled

    def test_fill_skin_tone(self):
        r = ImageResult(image_path="x.jpg", image_stem="x")

        class FakeSkin:
            L_median = 68.0
            a_median = 12.0
            b_median = 22.0
            ITA = 42.3
            chardon_category = "Light"

        fill_skin_tone(r, FakeSkin(), None)
        assert r.L_median_before == 68.0
        assert r.ITA_before == 42.3
        assert r.chardon_before == "Light"
        assert r.L_median_after is None
