"""Continuous-Fragmented (ContFrag) decoding for clusterless_high_d.

Wraps ``non_local_detector``'s 2-state ContFrag classifiers (Continuous vs
Fragmented) for both clusterless and sorted-spike data, in 2D (no track graph)
or 1D (Spyglass ``TrackGraph``). Mirrors the recipe in
``continuum_swr_replay.scripts.run_decoding_contfrag``.

Functions
---------
run_contfrag_decode
    Build the environment + classifier and fit/decode for one modality.
gather_area_spikes
    Concatenate spike trains (and clusterless features) across matching areas.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Encoding-likelihood defaults (from continuum_swr_replay.parameters).
PLACE_BIN_SIZE_1D: float = 2.0
PLACE_BIN_SIZE_2D: float = 2.5
POSITION_STD: float = float(np.sqrt(12.5))  # ~3.54 cm
WAVEFORM_STD: float = 24.0  # uV
BLOCK_SIZE: int = 2**10  # 1024


def gather_area_spikes(
    spikes: dict[str, dict],
    area_match: str,
    modality: str,
) -> tuple[list[NDArray], list[NDArray] | None]:
    """Concatenate spike trains across all areas whose name matches.

    Jacob's clusterless is keyed by hemisphere ("Left/Right CA1 Hippocampus")
    while sorted is under "CA1"; matching on a substring ("CA1") pools them.

    Parameters
    ----------
    spikes : dict
        Output of ``load_spikes_by_area``.
    area_match : str
        Case-insensitive substring; areas whose name contains it are pooled.
    modality : str
        ``"sorted"`` or ``"clusterless"``.

    Returns
    -------
    spike_times : list[NDArray]
        One array per unit/probe across matching areas.
    waveform_features : list[NDArray] or None
        Per-unit features for ``"clusterless"``; ``None`` for ``"sorted"``.
    """
    spike_times: list[NDArray] = []
    features: list[NDArray] = []
    for area, area_data in spikes.items():
        if area_match.lower() not in area.lower() or modality not in area_data:
            continue
        spike_times.extend(area_data[modality]["spike_times"])
        if modality == "clusterless":
            features.extend(area_data[modality]["waveform_features"])
    if not spike_times:
        raise ValueError(
            f"No {modality} spikes found for areas matching {area_match!r}"
        )
    return spike_times, (features if modality == "clusterless" else None)


def decoded_map_position(results) -> NDArray[np.float64]:
    """Maximum-a-posteriori decoded position per time bin.

    Marginalizes the ContFrag posterior over the two states, then defers to
    ``non_local_detector.analysis.posterior.maximum_a_posteriori_estimate`` for
    the argmax (handles both 1D and 2D position grids).

    Parameters
    ----------
    results : xr.Dataset
        Output of ``run_contfrag_decode`` (``acausal_posterior`` over a stacked
        ``state_bins`` index with ``state`` + position coordinates).

    Returns
    -------
    NDArray, shape (n_time, n_position_dims)
        MAP position. 2D -> (n_time, 2); 1D -> (n_time, 1).
    """
    from non_local_detector.analysis.posterior import (
        maximum_a_posteriori_estimate,
    )

    posterior = results.acausal_posterior.unstack("state_bins").sum("state")
    return maximum_a_posteriori_estimate(posterior)


def compute_ahead_behind_distance(results, env, position_info):
    """Signed decode distance via ``get_ahead_behind_distance2D``.

    Magnitude is the geodesic distance from the animal's position to the MAP
    decoded position along the environment's N-D bin graph (``track_graphDD``),
    which respects the track interior rather than cutting across gaps; the sign
    is positive when the decode is ahead of the animal's heading, negative when
    behind. ``np.abs`` of the result is the decode error (cm).

    Parameters
    ----------
    results : xr.Dataset
        Output of ``run_contfrag_decode``.
    env : Environment
        The fitted environment; ``track_graphDD`` + ``edges_`` define the 2D
        geodesic.
    position_info : pd.DataFrame
        Needs ``position_x``, ``position_y``, ``velocity_x``, ``velocity_y``.

    Returns
    -------
    NDArray, shape (n_time,)
        Signed ahead/behind distance (cm).
    """
    from non_local_detector.analysis.distance2D import (
        get_ahead_behind_distance2D,
    )

    map_xy = decoded_map_position(results)
    head_position = position_info[["position_x", "position_y"]].to_numpy()
    # head_direction is a heading ANGLE in radians (the function computes
    # cos(head_direction - map_direction)); the docstring's "(n_time, 2)" is
    # misleading. Use the movement heading (orientation is unreliable for these
    # DLC epochs).
    head_direction = np.arctan2(
        position_info["velocity_y"].to_numpy(),
        position_info["velocity_x"].to_numpy(),
    )
    # precomputed_distance=False: the precomputed path in this nld build calls
    # .items() on an nx.shortest_path generator (broken under networkx >=3),
    # so we use the per-time-point path instead.
    return get_ahead_behind_distance2D(
        head_position,
        head_direction,
        map_xy,
        track_graph=env.track_graphDD,
        edges=env.edges_,
        precomputed_distance=False,
    )


def _build_environment(
    is_1d: bool,
    track_graph_name: str | None,
    place_bin_size: float | None = None,
):
    """Construct a non_local_detector Environment (2D bin grid or 1D track)."""
    from non_local_detector.environment import Environment

    if not is_1d:
        return Environment(
            place_bin_size=place_bin_size or PLACE_BIN_SIZE_2D
        )

    if track_graph_name is None:
        raise ValueError("1D decoding requires `track_graph_name`.")
    from spyglass.linearization.v1 import TrackGraph

    tg = TrackGraph & {"track_graph_name": track_graph_name}
    if len(tg) == 0:
        raise ValueError(f"TrackGraph {track_graph_name!r} not found.")
    row = tg.fetch1()
    graph = tg.get_networkx_track_graph()
    edge_spacing = row["linear_edge_spacing"]
    return Environment(
        place_bin_size=PLACE_BIN_SIZE_1D,
        track_graph=graph,
        edge_order=row["linear_edge_order"],
        edge_spacing=edge_spacing,
    )


def run_contfrag_decode(
    position_info: pd.DataFrame,
    spikes: dict[str, dict],
    area_match: str,
    modality: str,
    is_1d: bool = False,
    track_graph_name: str | None = None,
    speed_threshold: float | None = None,
    is_training: NDArray[np.bool_] | None = None,
    position_std: float = POSITION_STD,
    waveform_std: float = WAVEFORM_STD,
    mark_transform: str = "raw",
    mark_n_components: int | None = None,
):
    """Fit a ContFrag classifier and decode one modality over an epoch.

    Parameters
    ----------
    position_info : pd.DataFrame
        Time-indexed position (needs ``position_x``, ``position_y``, ``speed``).
    spikes : dict
        Output of ``load_spikes_by_area``.
    area_match : str
        Substring selecting which areas to pool (e.g. ``"CA1"``).
    modality : str
        ``"sorted"`` or ``"clusterless"``.
    is_1d : bool, optional
        2D bin grid (default) or 1D linearized track.
    track_graph_name : str or None, optional
        Spyglass ``TrackGraph`` name, required when ``is_1d``.
    speed_threshold : float or None, optional
        Used only when ``is_training`` is None. If None (default), ALL samples
        train the encoding model; if a number, only samples faster than it (cm/s).
    is_training : NDArray[bool] or None, optional
        Explicit per-sample mask selecting which samples fit the encoding model.
        Overrides ``speed_threshold`` when given — used by the CV harness to fit on
        a training fold and score the held-out fold. All time bins are still decoded.
    position_std : float, optional
        Position-kernel bandwidth (cm) for the encoding model. Exposed for bandwidth CV.
    waveform_std : float, optional
        Waveform/mark-kernel bandwidth for clusterless. Exposed for bandwidth CV. NOTE: the
        right value depends on ``mark_transform`` — raw amplitude marks want ~24 (uV), but
        z-scored/whitened marks are unit-variance and want ~0.5-2.
    mark_transform : {"raw", "zscore", "whiten", "pca"}, optional
        Per-probe transform applied to clusterless marks before decoding (see
        :func:`clusterless_high_d.marks.transform_marks`). Default "raw".
    mark_n_components : int or None, optional
        PCA components to keep when ``mark_transform="pca"``.

    Returns
    -------
    results : xr.Dataset
        Posterior + state probabilities over time.
    detector : ContFrag{Clusterless,SortedSpikes}Classifier
    env : Environment
    """
    from non_local_detector import (
        ContFragClusterlessClassifier,
        ContFragSortedSpikesClassifier,
    )

    env = _build_environment(is_1d, track_graph_name)

    time = position_info.index.to_numpy()
    position = position_info[["position_x", "position_y"]].to_numpy()
    if is_training is not None:
        is_training = np.asarray(is_training, dtype=bool)
        if len(is_training) != len(time):
            raise ValueError("is_training length must match the number of time bins")
    elif speed_threshold is None:
        is_training = np.ones(len(time), dtype=bool)
    else:
        is_training = position_info["speed"].to_numpy() > speed_threshold
    sampling_frequency = float(1.0 / np.median(np.diff(time)))

    spike_times, features = gather_area_spikes(spikes, area_match, modality)
    if modality == "clusterless" and mark_transform != "raw":
        from .marks import transform_marks

        features = transform_marks(
            features, kind=mark_transform, n_components=mark_n_components
        )
    logger.info(
        "ContFrag %s decode (%s): %d units, %d time bins, %.1f Hz, "
        "%d/%d training samples (pos_std=%.2f, wf_std=%.2f)",
        modality,
        "1D" if is_1d else "2D",
        len(spike_times),
        len(time),
        sampling_frequency,
        int(is_training.sum()),
        len(is_training),
        position_std,
        waveform_std,
    )

    common = {
        "position_time": time,
        "position": position,
        "spike_times": spike_times,
        "time": time,
        "is_training": is_training,
    }

    if modality == "clusterless":
        detector = ContFragClusterlessClassifier(
            environments=env,
            clusterless_algorithm_params={
                "position_std": position_std,
                "waveform_std": waveform_std,
                "block_size": BLOCK_SIZE,
            },
            sampling_frequency=sampling_frequency,
        )
        results = detector.estimate_parameters(
            spike_waveform_features=features, **common
        )
    elif modality == "sorted":
        detector = ContFragSortedSpikesClassifier(
            environments=env,
            sorted_spikes_algorithm_params={
                "position_std": position_std,
                "block_size": BLOCK_SIZE,
            },
            sampling_frequency=sampling_frequency,
        )
        results = detector.estimate_parameters(**common)
    else:
        raise ValueError(f"Unknown modality {modality!r}")

    return results, detector, env
