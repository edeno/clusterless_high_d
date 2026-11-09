"""Nested cross-validation for encoding-bandwidth selection.

Selects the encoding-model bandwidth (``waveform_std`` for clusterless, ``position_std``
for both) by **validation** performance and reports an unbiased **test** estimate via
nested CV — implementing the leakage-safe protocol in
``.claude/research/clusterless-high-d/03-evaluation-framework.md``.

Design
------
- **Score = true-position log-density** (a bin-width-invariant proper score; see
  :func:`clusterless_high_d.evaluation.true_position_log_density`), evaluated on held-out,
  movement-restricted bins.
- **Folds are contiguous in time** (sklearn ``KFold(shuffle=False)`` or ``TimeSeriesSplit``)
  with an optional **embargo** that drops training bins adjacent to the test block — neural
  / position data are strongly autocorrelated, so random k-fold leaks.
- **Fit on train, decode all, score the held-out block.** Each fold passes an
  ``is_training`` mask to :func:`run_contfrag_decode`: the encoding model is fit on the
  training bins only; the held-out bins are scored from the resulting posterior. No test
  bin contributes to any encoding model used to score it.

Cost: nested CV runs ``outer * (n_bandwidths * inner + 1)`` full decodes — run under
``tmux`` on a pinned GPU for realistic grids.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold, TimeSeriesSplit

from clusterless_high_d import evaluation as ev
from clusterless_high_d.decode import POSITION_STD, WAVEFORM_STD, run_contfrag_decode

logger = logging.getLogger(__name__)


def time_series_folds(
    n_samples: int,
    n_splits: int,
    embargo: int = 0,
    scheme: str = "kfold",
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Contiguous-in-time train/test index folds via sklearn, with an embargo.

    Parameters
    ----------
    n_samples : int
    n_splits : int
        Number of folds.
    embargo : int, optional
        Drop training bins within ``embargo`` samples of the test block (both sides)
        to break short-range autocorrelation leakage. Default 0.
    scheme : {"kfold", "timeseries"}, optional
        ``"kfold"`` -> ``KFold(shuffle=False)`` (contiguous test blocks, train on both
        sides); ``"timeseries"`` -> ``TimeSeriesSplit`` (forward-only, train precedes test).

    Returns
    -------
    list of (train_idx, test_idx)
        Index arrays into ``range(n_samples)``.
    """
    idx = np.arange(n_samples)
    if scheme == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=False)
    elif scheme == "timeseries":
        splitter = TimeSeriesSplit(n_splits=n_splits)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    folds = []
    for train, test in splitter.split(idx):
        if embargo > 0:
            lo, hi = test.min() - embargo, test.max() + embargo
            train = train[(train < lo) | (train > hi)]
        folds.append((train, test))
    return folds


def _decode_and_score(
    position_info: pd.DataFrame,
    spikes: dict,
    area_match: str,
    modality: str,
    train_idx: NDArray[np.intp],
    eval_idx: NDArray[np.intp],
    bandwidth: dict,
    move: NDArray[np.bool_],
    is_1d: bool,
) -> dict:
    """Fit the encoding model on ``train_idx``, decode, score ``eval_idx`` (movement only).

    Returns dict with ``log_density`` (mean proper score on held-out movement bins),
    ``error`` (median decode error there), and ``n_eval``.
    """
    n = len(position_info)
    mask = np.zeros(n, dtype=bool)
    mask[train_idx] = True
    results, _, _ = run_contfrag_decode(
        position_info, spikes, area_match, modality,
        is_1d=is_1d, is_training=mask,
        position_std=bandwidth.get("position_std", POSITION_STD),
        waveform_std=bandwidth.get("waveform_std", WAVEFORM_STD),
    )
    prob, axes, _ = ev.extract_position_posterior(results)
    true_pos = position_info[["position_x", "position_y"]].to_numpy()
    eval_move = eval_idx[move[eval_idx]]
    if len(eval_move) == 0:
        return {"log_density": np.nan, "error": np.nan, "n_eval": 0}
    ld = ev.true_position_log_density(prob, axes, true_pos)["values"]
    de = ev.decode_error(prob, axes, true_pos)["errors"]
    return {
        "log_density": float(np.mean(ld[eval_move])),
        "error": float(np.median(de[eval_move])),
        "n_eval": int(len(eval_move)),
    }


def nested_cv_bandwidth(
    position_info: pd.DataFrame,
    spikes: dict,
    area_match: str,
    modality: str,
    bandwidth_grid: list[dict],
    outer_splits: int = 3,
    inner_splits: int = 2,
    embargo: int = 0,
    scheme: str = "kfold",
    speed_threshold: float = 4.0,
    is_1d: bool = False,
) -> dict:
    """Nested CV: inner-select bandwidth by validation log-density, report test score.

    Parameters
    ----------
    bandwidth_grid : list of dict
        Each dict gives the bandwidth(s) to try, e.g.
        ``[{"waveform_std": 12}, {"waveform_std": 24}, {"waveform_std": 48}]`` (clusterless)
        or ``[{"position_std": 3}, {"position_std": 6}]`` (either modality).
    outer_splits, inner_splits : int
        Outer (test) and inner (validation) fold counts.
    embargo : int
        Embargo in samples (see :func:`time_series_folds`).
    scheme : {"kfold", "timeseries"}
    speed_threshold : float
        Movement threshold (cm/s) for scoring bins.

    Returns
    -------
    dict with
        ``folds`` : per-outer-fold records (selected bandwidth, inner scores, test scores),
        ``mean_test_log_density``, ``mean_test_error``,
        ``selected`` : list of the bandwidth chosen in each outer fold.
    """
    n = len(position_info)
    move = position_info["speed"].to_numpy() > speed_threshold
    n_decodes = outer_splits * (len(bandwidth_grid) * inner_splits + 1)
    logger.info(
        "nested CV: %d outer x %d inner x %d bandwidths -> %d decodes",
        outer_splits, inner_splits, len(bandwidth_grid), n_decodes,
    )

    fold_records = []
    for k, (otr, ote) in enumerate(
        time_series_folds(n, outer_splits, embargo, scheme)
    ):
        # ---- inner CV on the outer-train block: mean validation log-density per bandwidth
        inner = time_series_folds(len(otr), inner_splits, embargo, scheme)
        val_scores = np.full((len(bandwidth_grid), len(inner)), np.nan)
        for j, (i_tr, i_val) in enumerate(inner):
            for b, bw in enumerate(bandwidth_grid):
                s = _decode_and_score(
                    position_info, spikes, area_match, modality,
                    otr[i_tr], otr[i_val], bw, move, is_1d,
                )
                val_scores[b, j] = s["log_density"]
        mean_val = np.nanmean(val_scores, axis=1)
        best = int(np.nanargmax(mean_val))
        best_bw = bandwidth_grid[best]

        # ---- refit on the full outer-train, score the outer-test block
        test = _decode_and_score(
            position_info, spikes, area_match, modality, otr, ote, best_bw, move, is_1d,
        )
        logger.info(
            "outer fold %d: selected %s (val logdens %.3f) -> test logdens %.3f, err %.1f cm",
            k, best_bw, mean_val[best], test["log_density"], test["error"],
        )
        fold_records.append({
            "fold": k,
            "selected_bandwidth": best_bw,
            "mean_val_log_density": {str(bw): float(v)
                                     for bw, v in zip(bandwidth_grid, mean_val, strict=True)},
            "test_log_density": test["log_density"],
            "test_error": test["error"],
            "n_test": test["n_eval"],
        })

    return {
        "folds": fold_records,
        "selected": [r["selected_bandwidth"] for r in fold_records],
        "mean_test_log_density": float(
            np.mean([r["test_log_density"] for r in fold_records])
        ),
        "mean_test_error": float(np.mean([r["test_error"] for r in fold_records])),
    }
