"""Evaluation metrics for clusterless / sorted decoding posteriors.

Implements the panel from ``.claude/research/clusterless-high-d/03-evaluation-framework.md``:

Reference-free, single decode (need behavior):
    - ``decode_error``                 median MAP/COM error vs the animal (movement only)
    - ``true_position_log_density``    bin-width-invariant proper score (degrades before RMSE)
    - ``hpd_coverage``                 credible-region coverage / reliability curve

Two-decode agreement (clusterless vs sorted, no behavior needed):
    - ``jensen_shannon`` / ``hellinger`` / ``total_variation``  full-posterior divergence
    - ``wasserstein_1d``               geometry-aware distance (1D / linearized)
    - ``state_probability_agreement``  Continuous-vs-Fragmented kappa + Brier

Plus xarray adapters that pull the position posterior and state probabilities out of a
``run_contfrag_decode`` ``results`` Dataset.

Conventions: position posteriors are probability MASS arrays summing to 1 over the position
grid per time bin, shape ``(n_time, *grid_shape)``; ``axes`` is a list of 1-D bin-center
arrays, one per position dimension; ``true_pos`` is ``(n_time, n_dims)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12


# --------------------------------------------------------------------------- adapters
def extract_position_posterior(results):
    """Pull the per-time position posterior (state marginalized) out of ``results``.

    Parameters
    ----------
    results : xr.Dataset
        Output of :func:`clusterless_high_d.decode.run_contfrag_decode`.

    Returns
    -------
    prob : NDArray, shape (n_time, *grid_shape)
        Posterior probability mass over the position grid, summing to ~1 per time bin.
    axes : list[NDArray]
        Bin-center coordinates, one 1-D array per position dimension.
    time : NDArray, shape (n_time,)
    """
    post = results.acausal_posterior.unstack("state_bins").sum("state")
    pos_dims = [d for d in post.dims if d != "time"]
    post = post.transpose("time", *pos_dims)
    axes = [np.asarray(post[d].values) for d in pos_dims]
    prob = np.asarray(post.values, dtype=float)
    time = np.asarray(post["time"].values)
    return prob, axes, time


def extract_continuous_probability(results) -> NDArray[np.float64]:
    """Per-time probability of the Continuous state (vs Fragmented).

    Returns
    -------
    NDArray, shape (n_time,)
        P(Continuous). Falls back to the first state if no name matches.
    """
    sp = results.acausal_posterior.unstack("state_bins").sum(
        [
            d
            for d in results.acausal_posterior.unstack("state_bins").dims
            if d not in ("time", "state")
        ]
    )
    states = [str(s) for s in np.asarray(sp["state"].values)]
    idx = next((i for i, s in enumerate(states) if "cont" in s.lower()), 0)
    sp = sp.transpose("time", "state")
    return np.asarray(sp.values, dtype=float)[:, idx]


# --------------------------------------------------------------- internal helpers
def _flatten(prob: NDArray) -> NDArray:
    """(n_time, *grid) -> (n_time, n_bins), renormalized over bins."""
    p = prob.reshape(prob.shape[0], -1).astype(float)
    s = p.sum(axis=1, keepdims=True)
    return p / np.where(s > 0, s, 1.0)


def _bin_volume(axes: list[NDArray]) -> float:
    """Product of per-axis bin spacing (assumes ~uniform grid)."""
    vol = 1.0
    for a in axes:
        a = np.asarray(a, dtype=float)
        vol *= float(np.median(np.diff(a))) if a.size > 1 else 1.0
    return vol


def _true_bin_index(axes: list[NDArray], true_pos: NDArray) -> NDArray[np.intp]:
    """Flattened grid index of the bin containing each true position."""
    shape = tuple(a.size for a in axes)
    multi = []
    for d, a in enumerate(axes):
        edges = _centers_to_edges(a)
        idx = np.clip(np.digitize(true_pos[:, d], edges) - 1, 0, a.size - 1)
        multi.append(idx)
    return np.ravel_multi_index(tuple(multi), shape)


def _centers_to_edges(centers: NDArray) -> NDArray:
    c = np.asarray(centers, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])


def _interp_density_at(prob: NDArray, axes: list[NDArray], pts: NDArray) -> NDArray:
    """Multilinear-interpolate the per-time density at per-time points (1-D or 2-D).

    prob: (n_time, *grid) probability mass; density = mass / bin_volume.
    pts:  (n_time, n_dims). Returns (n_time,) density (clipped >= 0).
    """
    vol = _bin_volume(axes)
    dens = prob / vol
    nt = prob.shape[0]
    if len(axes) == 1:
        out = np.empty(nt)
        x = np.asarray(axes[0], dtype=float)
        for t in range(nt):
            out[t] = np.interp(pts[t, 0], x, dens[t], left=dens[t, 0], right=dens[t, -1])
        return np.clip(out, 0, None)
    if len(axes) == 2:
        x, y = np.asarray(axes[0], float), np.asarray(axes[1], float)
        ix = np.clip(np.searchsorted(x, pts[:, 0]) - 1, 0, x.size - 2)
        iy = np.clip(np.searchsorted(y, pts[:, 1]) - 1, 0, y.size - 2)
        tx = np.clip((pts[:, 0] - x[ix]) / np.maximum(x[ix + 1] - x[ix], EPS), 0, 1)
        ty = np.clip((pts[:, 1] - y[iy]) / np.maximum(y[iy + 1] - y[iy], EPS), 0, 1)
        t = np.arange(nt)
        d00, d10 = dens[t, ix, iy], dens[t, ix + 1, iy]
        d01, d11 = dens[t, ix, iy + 1], dens[t, ix + 1, iy + 1]
        out = (1 - tx) * (1 - ty) * d00 + tx * (1 - ty) * d10 + (1 - tx) * ty * d01 + tx * ty * d11
        return np.clip(out, 0, None)
    raise NotImplementedError("only 1-D and 2-D position grids supported")


# --------------------------------------------------------------- single-decode metrics
def map_position(prob: NDArray, axes: list[NDArray]) -> NDArray:
    """Maximum-a-posteriori position per time bin, shape (n_time, n_dims)."""
    nt = prob.shape[0]
    flat_arg = prob.reshape(nt, -1).argmax(axis=1)
    multi = np.unravel_index(flat_arg, tuple(a.size for a in axes))
    return np.stack([axes[d][multi[d]] for d in range(len(axes))], axis=1)


def decode_error(
    prob: NDArray,
    axes: list[NDArray],
    true_pos: NDArray,
    is_movement: NDArray | None = None,
) -> dict:
    """Euclidean MAP-to-true decode error (movement bins).

    Returns dict with ``median``, ``mean``, ``p90``, ``errors`` (movement-restricted),
    and ``frac_within_10cm``.
    """
    mp = map_position(prob, axes)
    err = np.linalg.norm(mp - true_pos, axis=1)
    if is_movement is not None:
        err = err[is_movement]
    return {
        "median": float(np.median(err)),
        "mean": float(np.mean(err)),
        "p90": float(np.percentile(err, 90)),
        "frac_within_10cm": float(np.mean(err < 10)),
        "errors": err,
    }


def true_position_log_density(
    prob: NDArray,
    axes: list[NDArray],
    true_pos: NDArray,
    is_movement: NDArray | None = None,
    floor: float = 1e-6,
) -> dict:
    """Bin-width-invariant proper score: mean log posterior *density* at the truth.

    Higher is better. Reduces to ``-log(area)`` for a uniform posterior, independent of
    bin width, so it is comparable across grids and across KDE/GMM. Degrades before MAP
    RMSE does. Returns dict with ``mean``, ``median``, and per-bin ``values``.
    """
    dens = _interp_density_at(prob, axes, true_pos)
    logd = np.log(np.maximum(dens, floor))
    if is_movement is not None:
        logd = logd[is_movement]
    return {"mean": float(np.mean(logd)), "median": float(np.median(logd)), "values": logd}


def hpd_sharpness(
    prob: NDArray,
    axes: list[NDArray],
    coverage: float = 0.95,
    is_movement: NDArray | None = None,
) -> dict:
    """Sharpness = area of the HPD region (smaller = sharper), via nld's threshold.

    Wraps ``non_local_detector.model_checking.get_highest_posterior_threshold`` to define
    the region, then reports its area (bin count x bin volume) in position units^2.
    Sharpness pairs with :func:`hpd_coverage`: maximize sharpness *subject to* calibration.

    Returns dict with ``mean_area``, ``median_area``, ``mean_frac_bins``.
    """
    from non_local_detector.model_checking.highest_posterior_density import (
        get_highest_posterior_threshold,
    )

    flat = _flatten(prob)
    thr = np.asarray(get_highest_posterior_threshold(flat, coverage=coverage))
    region = flat >= thr[:, None]
    n_in = region.sum(axis=1)
    if is_movement is not None:
        n_in = n_in[is_movement]
    area = n_in * _bin_volume(axes)
    return {
        "mean_area": float(np.mean(area)),
        "median_area": float(np.median(area)),
        "mean_frac_bins": float(np.mean(n_in / flat.shape[1])),
    }


def hpd_coverage(
    prob: NDArray,
    axes: list[NDArray],
    true_pos: NDArray,
    levels: NDArray | None = None,
    is_movement: NDArray | None = None,
) -> dict:
    """Highest-posterior-density coverage / reliability curve.

    For each credible level ``L``, the fraction of (movement) time bins whose smallest
    HPD region of mass ``>= L`` contains the true position. A calibrated decoder lies on
    the diagonal; below = overconfident, above = underconfident. Uses the same HPD-region
    concept as ``non_local_detector.model_checking`` (bins with posterior above the
    coverage threshold) but computes the *expected* coverage with tie-correction (nld's
    raw threshold is for the region/area, not a reliability curve, and over-covers under
    exact ties such as a uniform posterior).

    Returns dict with ``levels`` and ``coverage`` (same length).
    """
    if levels is None:
        levels = np.linspace(0.05, 0.95, 19)
    flat = _flatten(prob)
    tbin = _true_bin_index(axes, true_pos)
    t = np.arange(flat.shape[0])
    p_true = flat[t, tbin][:, None]
    atol = 1e-12
    # HPD adds bins in descending-probability order. Split mass relative to the true bin:
    #   mass_above = mass of bins STRICTLY more probable than the true bin,
    #   mass_tie   = mass of bins equally probable (the true bin's tie group).
    # The true bin enters the level-L region with probability clip((L-above)/tie, 0, 1)
    # (random ordering within the tie group). For a uniform posterior this gives exactly L;
    # for distinct values it reduces to a step at `above`.
    mass_above = np.where(flat > p_true + atol, flat, 0.0).sum(axis=1)
    mass_tie = np.where(np.abs(flat - p_true) <= atol, flat, 0.0).sum(axis=1)
    if is_movement is not None:
        mass_above, mass_tie = mass_above[is_movement], mass_tie[is_movement]
    tie = np.maximum(mass_tie, EPS)
    coverage = np.array([np.mean(np.clip((L - mass_above) / tie, 0.0, 1.0)) for L in levels])
    return {"levels": np.asarray(levels), "coverage": coverage}


# --------------------------------------------------------------- two-decode divergences
def _norm(p: NDArray) -> NDArray:
    s = p.sum(axis=-1, keepdims=True)
    return p / np.where(s > 0, s, 1.0)


def jensen_shannon(p: NDArray, q: NDArray, base: float = 2.0) -> NDArray:
    """Per-time Jensen-Shannon divergence between two flattened posteriors.

    p, q: (n_time, n_bins). Symmetric, finite, bounded [0, 1] in bits (base 2). 0 iff equal.
    """
    p, q = _norm(p.astype(float)), _norm(q.astype(float))
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        out = np.zeros_like(a)
        out[mask] = a[mask] * (np.log(a[mask]) - np.log(b[mask]))
        return out.sum(axis=-1)

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return js / np.log(base)


def hellinger(p: NDArray, q: NDArray) -> NDArray:
    """Per-time Hellinger distance, bounded [0, 1]. No zero-handling needed."""
    p, q = _norm(p.astype(float)), _norm(q.astype(float))
    return np.sqrt(np.clip(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum(axis=-1), 0, 1))


def total_variation(p: NDArray, q: NDArray) -> NDArray:
    """Per-time total-variation distance, 0.5 * L1, bounded [0, 1]."""
    p, q = _norm(p.astype(float)), _norm(q.astype(float))
    return 0.5 * np.abs(p - q).sum(axis=-1)


def wasserstein_1d(p: NDArray, q: NDArray, centers: NDArray) -> NDArray:
    """Per-time 1-D Wasserstein-1 distance (geometry-aware), in position units.

    Closed form: integral of |CDF_p - CDF_q| along the bin centers. For 2-D, linearize
    position first (or compute per-axis as a proxy). p, q: (n_time, n_bins).
    """
    p, q = _norm(p.astype(float)), _norm(q.astype(float))
    centers = np.asarray(centers, dtype=float)
    dx = np.diff(centers)
    cdf_diff = np.cumsum(p - q, axis=-1)[:, :-1]
    return np.abs(cdf_diff * dx).sum(axis=-1)


def posterior_agreement(results_a, results_b) -> dict:
    """Convenience: JSD + Hellinger + TV per time between two decodes' position posteriors.

    Both decodes must share the same position grid and time bins. Returns dict of arrays.
    """
    from non_local_detector.model_checking.posterior_consistency import (
        posterior_consistency_hpd_overlap,
    )

    pa, axes_a, _ = extract_position_posterior(results_a)
    pb, axes_b, _ = extract_position_posterior(results_b)
    fa, fb = _flatten(pa), _flatten(pb)
    out = {
        "jensen_shannon": jensen_shannon(fa, fb),
        "hellinger": hellinger(fa, fb),
        "total_variation": total_variation(fa, fb),
        # nld's HPD-region overlap (intersection / min region size), 95% regions.
        "hpd_overlap_95": np.asarray(posterior_consistency_hpd_overlap(fa, fb, 0.95)),
    }
    if len(axes_a) == 1:
        out["wasserstein_1d"] = wasserstein_1d(fa, fb, axes_a[0])
    return out


# --------------------------------------------------------------- state agreement
def state_probability_agreement(
    pi_a: NDArray, pi_b: NDArray, is_movement: NDArray | None = None
) -> dict:
    """Continuous-vs-Fragmented agreement between two decodes.

    pi_a, pi_b: P(Continuous) per time. Returns Cohen's kappa on the 0.5-thresholded
    label (chance-corrected) and the Brier score ``mean((pi_a - pi_b)**2)``.
    """
    if is_movement is not None:
        pi_a, pi_b = pi_a[is_movement], pi_b[is_movement]
    la, lb = pi_a > 0.5, pi_b > 0.5
    n = len(la)
    po = np.mean(la == lb)
    # chance agreement
    pa1, pb1 = la.mean(), lb.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 1.0
    brier = float(np.mean((pi_a - pi_b) ** 2))
    return {"kappa": float(kappa), "brier": brier, "raw_agreement": float(po), "n": int(n)}
