"""Per-probe mark transforms (raw / z-score / whiten / PCA) for clusterless decoding.

These inject via ``spike_waveform_features`` (one array per probe) with no changes to
``non_local_detector`` — the same standardize/whiten/reduce arms tested on the synthetic
data (see ``.claude/research/clusterless-high-d/clusterless_kde_contact_dimensionality.md``).

Each probe's marks are transformed **independently** (probes are conditionally-independent
electrode groups). Transform statistics are fit on the encoding (training) spikes only when
``fit_masks`` is given, to avoid leakage into the held-out decode.

Conventions: ``features`` is a list of ``(n_spikes, n_channels)`` float arrays, one per probe.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-8


def transform_marks(
    features: list[NDArray],
    kind: str = "zscore",
    n_components: int | None = None,
    fit_masks: list[NDArray] | None = None,
) -> list[NDArray]:
    """Transform per-probe waveform marks.

    Parameters
    ----------
    features : list[NDArray], each shape (n_spikes, n_channels)
        Per-probe marks (e.g. amplitude on each channel).
    kind : {"raw", "zscore", "whiten", "pca"}
        - ``raw``: unchanged.
        - ``zscore``: per-channel standardize (mean 0, unit variance).
        - ``whiten``: ZCA whitening (decorrelate + unit variance, same dimensionality).
        - ``pca``: standardize then project onto the top ``n_components`` principal axes.
    n_components : int or None
        Number of PCA components to keep (``pca`` only). None keeps all channels.
    fit_masks : list[NDArray] or None
        Optional per-probe boolean masks selecting the spikes used to FIT the transform
        statistics (mean/std/covariance). The transform is still APPLIED to all spikes.
        Use to fit on encoding-fold spikes only. None fits on all spikes.

    Returns
    -------
    list[NDArray]
        Transformed per-probe marks (1:1 with input; empty arrays pass through).
    """
    if kind == "raw":
        return [np.asarray(f, dtype=float) for f in features]
    if fit_masks is None:
        fit_masks = [None] * len(features)

    out = []
    for f, m in zip(features, fit_masks, strict=True):
        f = np.asarray(f, dtype=float)
        if f.ndim != 2 or f.shape[0] == 0:
            out.append(f)
            continue
        fit = f if m is None else f[np.asarray(m, dtype=bool)]
        if fit.shape[0] < 2:  # not enough to estimate stats
            out.append(f)
            continue
        mu = fit.mean(axis=0)
        if kind == "zscore":
            sd = fit.std(axis=0)
            sd = np.where(sd > EPS, sd, 1.0)
            out.append((f - mu) / sd)
        elif kind == "whiten":
            cov = np.cov(fit.T) + EPS * np.eye(f.shape[1])
            vals, vecs = np.linalg.eigh(cov)
            vals = np.clip(vals, EPS, None)
            w = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T  # ZCA
            out.append((f - mu) @ w)
        elif kind == "pca":
            sd = fit.std(axis=0)
            sd = np.where(sd > EPS, sd, 1.0)
            fit_s = (fit - mu) / sd
            _, _, vt = np.linalg.svd(fit_s, full_matrices=False)
            k = n_components or f.shape[1]
            out.append(((f - mu) / sd) @ vt[:k].T)
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return out
