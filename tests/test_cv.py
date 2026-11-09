"""Unit tests for the nested-CV fold logic (no GPU / Spyglass)."""

from __future__ import annotations

import numpy as np

from clusterless_high_d.cv import time_series_folds


class TestTimeSeriesFolds:
    def test_kfold_contiguous_test_blocks(self):
        folds = time_series_folds(100, 5, scheme="kfold")
        assert len(folds) == 5
        for _, test in folds:
            assert np.all(np.diff(test) == 1)  # contiguous
        # test blocks partition the range exactly once
        all_test = np.concatenate([t for _, t in folds])
        assert np.array_equal(np.sort(all_test), np.arange(100))

    def test_no_train_test_overlap(self):
        for tr, te in time_series_folds(60, 4, scheme="kfold"):
            assert len(np.intersect1d(tr, te)) == 0

    def test_embargo_removes_adjacent_train(self):
        emb = 5
        for tr, te in time_series_folds(100, 4, embargo=emb, scheme="kfold"):
            lo, hi = te.min() - emb, te.max() + emb
            assert not np.any((tr >= lo) & (tr <= hi))

    def test_timeseries_is_forward_only(self):
        for tr, te in time_series_folds(100, 4, scheme="timeseries"):
            assert tr.max() < te.min()  # train strictly precedes test

    def test_embargo_zero_is_full_complement(self):
        n = 50
        for tr, te in time_series_folds(n, 5, embargo=0, scheme="kfold"):
            assert len(tr) + len(te) == n
