"""Unit tests for evaluation metrics on synthetic inputs with known answers.

Pure-numpy; no Spyglass/decode/GPU needed.
"""

from __future__ import annotations

import numpy as np

from clusterless_high_d import evaluation as ev


# ------------------------------------------------------------------ divergences
class TestDivergences:
    def test_identical_posteriors_zero(self):
        rng = np.random.default_rng(0)
        p = ev._norm(rng.random((20, 50)))
        assert np.allclose(ev.jensen_shannon(p, p), 0, atol=1e-9)
        assert np.allclose(ev.hellinger(p, p), 0, atol=1e-9)
        assert np.allclose(ev.total_variation(p, p), 0, atol=1e-9)

    def test_disjoint_supports_maximal(self):
        # one bin each, non-overlapping -> JSD=1 bit, Hellinger=1, TV=1
        p = np.array([[1.0, 0.0]])
        q = np.array([[0.0, 1.0]])
        assert np.isclose(ev.jensen_shannon(p, q, base=2.0)[0], 1.0, atol=1e-9)
        assert np.isclose(ev.hellinger(p, q)[0], 1.0, atol=1e-9)
        assert np.isclose(ev.total_variation(p, q)[0], 1.0, atol=1e-9)

    def test_jsd_symmetric(self):
        rng = np.random.default_rng(1)
        p, q = ev._norm(rng.random((5, 8))), ev._norm(rng.random((5, 8)))
        assert np.allclose(ev.jensen_shannon(p, q), ev.jensen_shannon(q, p))

    def test_no_nan_with_zeros(self):
        p = np.array([[0.5, 0.5, 0.0]])
        q = np.array([[0.0, 0.5, 0.5]])
        for fn in (ev.jensen_shannon, ev.hellinger, ev.total_variation):
            assert np.isfinite(fn(p, q)).all()


class TestWasserstein1D:
    def test_zero_for_identical(self):
        p = np.array([[0.0, 1.0, 0.0]])
        c = np.array([0.0, 1.0, 2.0])
        assert np.isclose(ev.wasserstein_1d(p, p, c)[0], 0.0)

    def test_equals_shift_distance(self):
        # point mass at x=0 vs x=3 on a unit grid -> W1 = 3
        centers = np.arange(6, dtype=float)
        p = np.zeros((1, 6))
        p[0, 0] = 1.0
        q = np.zeros((1, 6))
        q[0, 3] = 1.0
        assert np.isclose(ev.wasserstein_1d(p, q, centers)[0], 3.0, atol=1e-9)

    def test_geometry_aware_vs_jsd(self):
        # JSD treats a 1-bin and a 5-bin shift identically; W1 does not.
        centers = np.arange(10, dtype=float)
        base = np.zeros((1, 10))
        base[0, 2] = 1.0
        near = np.zeros((1, 10))
        near[0, 3] = 1.0
        far = np.zeros((1, 10))
        far[0, 7] = 1.0
        assert np.isclose(
            ev.jensen_shannon(base, near)[0], ev.jensen_shannon(base, far)[0]
        )  # both disjoint -> equal
        assert ev.wasserstein_1d(base, near, centers)[0] < ev.wasserstein_1d(base, far, centers)[0]


# ------------------------------------------------------------------ log density
class TestTruePositionLogDensity:
    def test_uniform_is_minus_log_area_1d(self):
        # uniform over [0, 10) with 10 unit bins -> density 1/10 everywhere -> log(0.1)
        axes = [np.arange(10, dtype=float) + 0.5]
        prob = np.full((4, 10), 0.1)
        true_pos = np.array([[3.2], [5.5], [0.1], [9.4]])
        out = ev.true_position_log_density(prob, axes, true_pos)
        assert np.isclose(out["mean"], np.log(0.1), atol=1e-6)

    def test_bin_width_invariant(self):
        # same uniform density at two bin resolutions -> same score
        axes_a = [np.arange(10, dtype=float) + 0.5]  # width 1, 10 bins
        pa = np.full((1, 10), 0.1)
        axes_b = [np.arange(0, 10, 0.5) + 0.25]  # width 0.5, 20 bins
        pb = np.full((1, 20), 1 / 20)
        tp = np.array([[4.3]])
        sa = ev.true_position_log_density(pa, axes_a, tp)["mean"]
        sb = ev.true_position_log_density(pb, axes_b, tp)["mean"]
        assert np.isclose(sa, sb, atol=1e-6)

    def test_peaked_beats_uniform(self):
        axes = [np.arange(10, dtype=float) + 0.5]
        uniform = np.full((1, 10), 0.1)
        peaked = np.full((1, 10), 0.01)
        peaked[0, 4] = 0.91
        tp = np.array([[4.5]])
        s_u = ev.true_position_log_density(uniform, axes, tp)["mean"]
        s_p = ev.true_position_log_density(peaked, axes, tp)["mean"]
        assert s_p > s_u

    def test_2d_uniform(self):
        axes = [np.arange(5, dtype=float) + 0.5, np.arange(4, dtype=float) + 0.5]
        prob = np.full((3, 5, 4), 1 / 20)
        tp = np.array([[2.1, 1.5], [0.3, 3.2], [4.4, 0.6]])
        out = ev.true_position_log_density(prob, axes, tp)
        assert np.isclose(out["mean"], np.log(1 / 20), atol=1e-6)  # density 1/area, area=20


# ------------------------------------------------------------------ HPD coverage
class TestHpdCoverage:
    def test_uniform_posterior_is_calibrated(self):
        # uniform posterior + uniformly random true bin -> coverage ~ level
        rng = np.random.default_rng(3)
        nbins, nt = 50, 20000
        axes = [np.arange(nbins, dtype=float) + 0.5]
        prob = np.full((nt, nbins), 1 / nbins)
        true_pos = rng.integers(0, nbins, nt).astype(float)[:, None] + 0.5
        out = ev.hpd_coverage(prob, axes, true_pos, levels=np.array([0.2, 0.5, 0.8]))
        assert np.allclose(out["coverage"], [0.2, 0.5, 0.8], atol=0.05)

    def test_overconfident_undercovers(self):
        # posterior always peaked at bin 0, truth elsewhere -> low coverage
        nbins, nt = 20, 500
        axes = [np.arange(nbins, dtype=float) + 0.5]
        prob = np.full((nt, nbins), 0.001)
        prob[:, 0] = 1 - 0.001 * (nbins - 1)
        true_pos = np.full((nt, 1), 10.5)  # never at the peak
        out = ev.hpd_coverage(prob, axes, true_pos, levels=np.array([0.5]))
        assert out["coverage"][0] < 0.5


# ------------------------------------------------------------------ state agreement
class TestStateAgreement:
    def test_perfect_agreement(self):
        pi = np.array([0.9, 0.8, 0.1, 0.2, 0.95])
        out = ev.state_probability_agreement(pi, pi)
        assert np.isclose(out["kappa"], 1.0)
        assert np.isclose(out["brier"], 0.0)

    def test_anti_agreement_negative_kappa(self):
        pi_a = np.array([0.9, 0.9, 0.1, 0.1])
        pi_b = np.array([0.1, 0.1, 0.9, 0.9])
        out = ev.state_probability_agreement(pi_a, pi_b)
        assert out["kappa"] < 0


# ------------------------------------------------------------------ decode error / MAP
class TestDecodeError:
    def test_map_recovers_peak_2d(self):
        axes = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
        prob = np.zeros((1, 3, 3))
        prob[0, 2, 1] = 1.0
        mp = ev.map_position(prob, axes)
        assert np.allclose(mp[0], [2.0, 1.0])

    def test_zero_error_when_map_at_truth(self):
        axes = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
        prob = np.zeros((2, 3, 3))
        prob[0, 1, 1] = 1.0
        prob[1, 0, 2] = 1.0
        true_pos = np.array([[1.0, 1.0], [0.0, 2.0]])
        out = ev.decode_error(prob, axes, true_pos)
        assert np.isclose(out["median"], 0.0)
