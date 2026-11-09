"""Unit tests for per-probe mark transforms (no GPU)."""

from __future__ import annotations

import numpy as np

from clusterless_high_d.marks import transform_marks


class TestTransformMarks:
    def setup_method(self):
        rng = np.random.default_rng(0)
        # two probes, correlated channels with different scales
        a = rng.normal(0, 1, (500, 3))
        self.p0 = a @ np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 5.0]]) + 10
        self.p1 = rng.normal(-30, 7, (300, 4))
        self.feats = [self.p0, self.p1]

    def test_raw_unchanged(self):
        out = transform_marks(self.feats, "raw")
        assert np.allclose(out[0], self.p0) and np.allclose(out[1], self.p1)

    def test_zscore_mean0_std1(self):
        out = transform_marks(self.feats, "zscore")
        for o in out:
            assert np.allclose(o.mean(axis=0), 0, atol=1e-6)
            assert np.allclose(o.std(axis=0), 1, atol=1e-6)

    def test_whiten_identity_covariance(self):
        out = transform_marks(self.feats, "whiten")
        for o in out:
            cov = np.cov(o.T)
            assert np.allclose(cov, np.eye(cov.shape[0]), atol=1e-4)

    def test_pca_reduces_dimension(self):
        out = transform_marks(self.feats, "pca", n_components=2)
        assert out[0].shape == (500, 2)
        assert out[1].shape == (300, 2)

    def test_shapes_preserved_for_zscore_whiten(self):
        for kind in ("zscore", "whiten"):
            out = transform_marks(self.feats, kind)
            assert out[0].shape == self.p0.shape
            assert out[1].shape == self.p1.shape

    def test_fit_mask_used_for_stats(self):
        # fit stats on first half only; applied to all. zscore mean over fit-half ~0.
        masks = [np.arange(len(self.p0)) < 250, np.arange(len(self.p1)) < 150]
        out = transform_marks(self.feats, "zscore", fit_masks=masks)
        assert np.allclose(out[0][:250].mean(axis=0), 0, atol=1e-6)

    def test_empty_probe_passthrough(self):
        out = transform_marks([np.empty((0, 3)), self.p1], "whiten")
        assert out[0].shape == (0, 3)

    def test_constant_channel_no_nan(self):
        f = self.p0.copy()
        f[:, 1] = 5.0  # zero-variance channel
        out = transform_marks([f], "zscore")
        assert np.isfinite(out[0]).all()
