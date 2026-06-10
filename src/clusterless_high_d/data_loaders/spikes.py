"""Spike data loading from Spyglass, organized by brain area.

Functions
---------
load_spikes_by_area
    Load sorted and clusterless spike data for all brain areas in an NWB file.
filter_spike_times
    Restrict spike times to position time bounds.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import label

from .utils import get_sort_group_brain_areas

logger = logging.getLogger(__name__)


def _load_sorted_spikes_from_group(
    nwb_file_name: str,
    position_time: NDArray[np.float64],
    brain_areas: dict[str, dict] | None = None,
) -> dict[str, list[NDArray[np.float64]]] | None:
    """Load sorted spikes using SortedSpikesGroup.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name.
    position_time : NDArray[np.float64]
        Position timestamps for time-slicing.
    brain_areas : dict[str, dict] or None, optional
        Brain area config from datasets.yaml. Maps area name →
        ``{"sorted_spikes_group_name": "..."}``). If provided, only
        the specified groups are loaded. If None or empty, loads all
        available groups.

    Returns
    -------
    dict[str, list[NDArray]] or None
        Brain area name → list of spike time arrays per unit.
        Returns None if no SortedSpikesGroup entries exist.
    """
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v1.recording import SpikeSortingRecordingSelection
    from spyglass.spikesorting.v1.sorting import SpikeSortingSelection

    ssg = SortedSpikesGroup & {"nwb_file_name": nwb_file_name}
    if len(ssg) == 0:
        return None

    time_slice = [float(position_time[0]), float(position_time[-1])]
    result: dict[str, list[NDArray[np.float64]]] = defaultdict(list)

    # If brain_areas config specifies groups, use those directly
    if brain_areas:
        for area_name, area_cfg in brain_areas.items():
            group_name = area_cfg.get("sorted_spikes_group_name")
            if not group_name:
                continue
            group_entries = ssg & {"sorted_spikes_group_name": group_name}
            if len(group_entries) == 0:
                logger.warning(
                    "SortedSpikesGroup '%s' not found for %s",
                    group_name,
                    nwb_file_name,
                )
                continue
            group_key = group_entries.fetch1("KEY")
            spike_times = SortedSpikesGroup.fetch_spike_data(
                group_key, time_slice=time_slice
            )
            if spike_times:
                result[area_name].extend(spike_times)
                logger.info(
                    "Loaded %d units from SortedSpikesGroup '%s' → %s",
                    len(spike_times),
                    group_name,
                    area_name,
                )
        return dict(result) if result else None

    # No brain_areas config — discover areas from all groups
    area_map = get_sort_group_brain_areas(nwb_file_name)
    sg_to_area: dict[int, str] = {}
    for area_name, sg_ids in area_map.items():
        for sg_id in sg_ids:
            sg_to_area[sg_id] = area_name

    groups = ssg.fetch(as_dict=True)

    for group in groups:
        group_name = group["sorted_spikes_group_name"]

        merge_ids = (
            SortedSpikesGroup.Units
            & {
                "nwb_file_name": nwb_file_name,
                "sorted_spikes_group_name": group_name,
            }
        ).fetch("spikesorting_merge_id")

        if len(merge_ids) == 0:
            continue

        # Trace first merge_id back to sort_group_id to determine brain area
        merge_key = {"merge_id": merge_ids[0]}
        try:
            curation_row = (SpikeSortingOutput.CurationV1 & merge_key).fetch1(
                "sorting_id"
            )
            recording_id = (
                SpikeSortingSelection & {"sorting_id": str(curation_row)}
            ).fetch1("recording_id")
            sort_group_id = int(
                (
                    SpikeSortingRecordingSelection
                    & {"recording_id": str(recording_id)}
                ).fetch1("sort_group_id")
            )
            area_name = sg_to_area.get(sort_group_id, "unknown")
        except Exception:
            logger.warning(
                "Could not determine brain area for group %s, skipping",
                group_name,
            )
            continue

        spike_times = SortedSpikesGroup.fetch_spike_data(
            group, time_slice=time_slice
        )
        if spike_times:
            result[area_name].extend(spike_times)
            logger.info(
                "Loaded %d units from SortedSpikesGroup '%s' → %s",
                len(spike_times),
                group_name,
                area_name,
            )

    return dict(result) if result else None


def _load_sorted_spikes_from_scratch(
    nwb_file_name: str,
    position_time: NDArray[np.float64],
) -> dict[str, list[NDArray[np.float64]]]:
    """Build sorted spike data from the CurationV1 → SpikeSortingOutput chain.

    Used when no SortedSpikesGroup entries exist for the NWB file.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name.
    position_time : NDArray[np.float64]
        Position timestamps for time-slicing.

    Returns
    -------
    dict[str, list[NDArray]]
        Brain area name → list of spike time arrays per unit.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v1.curation import CurationV1
    from spyglass.spikesorting.v1.recording import SpikeSortingRecordingSelection
    from spyglass.spikesorting.v1.sorting import SpikeSortingSelection

    area_map = get_sort_group_brain_areas(nwb_file_name)
    t_start, t_end = float(position_time[0]), float(position_time[-1])

    # Get all recordings for this NWB file
    rec_sel = SpikeSortingRecordingSelection & {"nwb_file_name": nwb_file_name}
    rec_rows = rec_sel.fetch("recording_id", "sort_group_id", as_dict=True)

    # Invert area_map for lookup
    sg_to_area: dict[int, str] = {}
    for area_name, sg_ids in area_map.items():
        for sg_id in sg_ids:
            sg_to_area[sg_id] = area_name

    result: dict[str, list[NDArray[np.float64]]] = defaultdict(list)

    for rec_row in rec_rows:
        sort_group_id = int(rec_row["sort_group_id"])
        area_name = sg_to_area.get(sort_group_id, "unknown")
        recording_id = str(rec_row["recording_id"])

        # Get sorting_ids for this recording
        sort_sel = SpikeSortingSelection & {"recording_id": recording_id}
        sorting_ids = sort_sel.fetch("sorting_id")

        for sorting_id in sorting_ids:
            # Get highest curation_id
            curations = CurationV1 & {"sorting_id": str(sorting_id)}
            if len(curations) == 0:
                continue
            max_curation_id = max(curations.fetch("curation_id"))

            # Find merge entry
            curation_key = {
                "sorting_id": str(sorting_id),
                "curation_id": max_curation_id,
            }
            merge_entries = SpikeSortingOutput.CurationV1 & curation_key
            if len(merge_entries) == 0:
                continue

            merge_id = merge_entries.fetch1("merge_id")
            merge_key = {"merge_id": merge_id}

            # Fetch spike times from NWB
            try:
                nwb_list = SpikeSortingOutput().fetch_nwb(merge_key)
                if not nwb_list:
                    continue
                nwb_file = nwb_list[0]

                # Determine which field has the spike data
                nwb_field = (
                    "object_id"
                    if "object_id" in nwb_file
                    else "units" if "units" in nwb_file else None
                )
                if nwb_field is None:
                    continue

                spike_times_list = nwb_file[nwb_field]["spike_times"].to_list()

                # Filter to position time bounds
                for st in spike_times_list:
                    in_bounds = (st >= t_start) & (st <= t_end)
                    filtered = st[in_bounds]
                    if len(filtered) > 0:
                        result[area_name].append(filtered)

            except Exception as e:
                logger.warning(
                    "Failed to fetch spikes for merge_id %s: %s", merge_id, e
                )
                continue

    return dict(result)


def filter_spike_times(
    spike_times: list[NDArray[np.float64]],
    position_time: NDArray[np.float64],
    waveform_features: list[NDArray[np.float64]] | None = None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]] | None]:
    """Filter spike times to position time bounds.

    Parameters
    ----------
    spike_times : list[NDArray]
        List of spike time arrays, one per unit/sort group.
    position_time : NDArray
        Position timestamps defining the valid time range.
    waveform_features : list[NDArray] or None, optional
        Corresponding waveform features arrays. If provided, filtered
        in lockstep with spike_times.

    Returns
    -------
    filtered_spike_times : list[NDArray]
        Spike times within position bounds. Units with zero spikes dropped.
    filtered_features : list[NDArray] or None
        Corresponding filtered features, or None if input was None.
    """
    t_start, t_end = position_time[0], position_time[-1]
    filtered_times: list[NDArray[np.float64]] = []
    filtered_feats: list[NDArray[np.float64]] = []

    for i, st in enumerate(spike_times):
        mask = (st >= t_start) & (st <= t_end)
        filtered = st[mask]
        if len(filtered) == 0:
            continue
        filtered_times.append(filtered)
        if waveform_features is not None:
            filtered_feats.append(waveform_features[i][mask])

    return filtered_times, filtered_feats if waveform_features is not None else None


def detect_coincident_spikes(
    spike_times: list[NDArray[np.float64]],
    spike_closeness_threshold: float = 0.00004,
    max_coincident_fraction: float = 0.33,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.intp]]]:
    """Detect and remove coincident spikes across sort groups (probes).

    Coincident spikes (within a tight time threshold across more than a
    fraction of spatially-separate sort groups) are almost certainly
    electrical/movement artifacts rather than biological neural activity:
    real neural propagation delays are ~1-10 ms, so threshold crossings within
    tens of microseconds across multiple probes are a shared electrical event.

    Ported from ``continuum_swr_replay.data_loaders.spikes`` (tetrode version);
    here the "sort groups" are high-density probe segments rather than tetrodes,
    but the principle is identical.

    Parameters
    ----------
    spike_times : list[NDArray[np.float64]]
        List of arrays, each containing spike times (seconds) for one sort
        group. Order is preserved 1:1 in the output.
    spike_closeness_threshold : float, optional
        Time threshold for considering spikes coincident (seconds). Default
        40 us (0.00004 s).
    max_coincident_fraction : float, optional
        Fraction of sort groups that must fire together within the threshold to
        flag the cluster as artifact. Default 0.33; real neural synchrony rarely
        exceeds ~20% of channels simultaneously.

    Returns
    -------
    tuple[list[NDArray[np.float64]], list[NDArray[np.intp]]]
        - filtered_spike_times: arrays with coincident spikes removed, one per
          input sort group (1:1 with input). Fully-removed groups get empty
          arrays. Returns ``[]`` if input is empty / all-empty.
        - filtered_time_bin_ind: indices of kept spikes within each original
          input array, one per sort group (1:1 with input), for filtering
          waveform features in lockstep. Returns ``[]`` on empty input.
    """
    if not spike_times or all(len(st) == 0 for st in spike_times):
        return [], []

    concat_spike_times = np.concatenate(spike_times)
    sort_group_id = np.concatenate(
        [
            np.ones(len(spike_time), dtype=int) * i
            for i, spike_time in enumerate(spike_times)
        ]
    )
    time_bin_ind = np.concatenate(
        [np.arange(len(spike_time), dtype=int) for spike_time in spike_times]
    )

    sort_ind = np.argsort(concat_spike_times)
    sorted_spike_times = concat_spike_times[sort_ind]
    sort_group_id = sort_group_id[sort_ind]
    time_bin_ind = time_bin_ind[sort_ind]

    is_close = np.diff(sorted_spike_times) < spike_closeness_threshold
    is_close = np.concatenate([[False], is_close])
    labels, _ = label(is_close)

    # ``scipy.ndimage.label`` leaves the first spike of each close cluster with
    # label=0 because ``np.diff`` is backward-looking. Extend each labeled run
    # backward by one spike so the cluster's leading spike contributes to
    # ``coincident_fraction`` (otherwise a 2-of-4 coincidence evaluates as 1/4
    # and slips through the 0.33 threshold).
    labels_arr = np.asarray(labels)
    next_label = np.concatenate([labels_arr[1:], [0]])
    cluster_labels = np.where(
        (labels_arr == 0) & (next_label > 0), next_label, labels_arr
    )

    spike_events = pd.DataFrame(
        {
            "labels": cluster_labels,
            "sort_group_id": sort_group_id,
            "time_bin_ind": time_bin_ind,
            "spike_times": sorted_spike_times,
        },
    )
    n_sort_groups = len(spike_times)
    coincident_fraction = (
        spike_events.loc[spike_events.labels > 0]
        .groupby("labels")
        .sort_group_id.nunique()
        / n_sort_groups
    )
    artifact_labels = coincident_fraction[
        coincident_fraction > max_coincident_fraction
    ]

    is_artifact = spike_events.labels.isin(artifact_labels.index)
    spike_events = spike_events.loc[~is_artifact]

    grouped = spike_events.groupby("sort_group_id")
    grouped_spike_times = grouped.spike_times.apply(np.array)
    grouped_time_bin_ind = grouped.time_bin_ind.apply(np.array)

    # Preserve all original sort group indices so the output aligns 1:1 with the
    # input. Groups that lost all spikes get empty arrays.
    filtered_spike_times = [
        grouped_spike_times[i]
        if i in grouped_spike_times.index
        else np.array([], dtype=np.float64)
        for i in range(n_sort_groups)
    ]
    filtered_time_bin_ind = [
        grouped_time_bin_ind[i]
        if i in grouped_time_bin_ind.index
        else np.array([], dtype=np.intp)
        for i in range(n_sort_groups)
    ]

    return filtered_spike_times, filtered_time_bin_ind


def _read_waveform_features_from_nwb(
    analysis_file_name: str,
    object_id: str,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Read spike times and waveform features from an analysis NWB file.

    Parameters
    ----------
    analysis_file_name : str
        Spyglass analysis NWB file name.
    object_id : str
        NWB object ID for the units table.

    Returns
    -------
    spike_times : list[NDArray]
        Spike times per unit.
    features : list[NDArray]
        Waveform features per unit, shape ``(n_spikes, n_features)``.
    """
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile

    path = AnalysisNwbfile.get_abs_path(analysis_file_name)
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwbf = io.read()
        units = nwbf.units
        if units is None or len(units) == 0:
            return [], []

        spike_times_raw = units["spike_times"][:]
        feature_cols = [c for c in units.colnames if c not in ("spike_times", "id")]

        spike_times_list = []
        features_list = []
        for i in range(len(spike_times_raw)):
            st = np.asarray(spike_times_raw[i], dtype=np.float64)
            feat_arrays = [np.asarray(units[col][i]) for col in feature_cols]
            feat = np.concatenate(feat_arrays, axis=1) if feat_arrays else np.empty((len(st), 0))
            spike_times_list.append(st)
            features_list.append(feat)

    return spike_times_list, features_list


def _load_clusterless(
    nwb_file_name: str,
    position_time: NDArray[np.float64],
    remove_coincident: bool = True,
    spike_closeness_threshold: float = 0.00004,
    max_coincident_fraction: float = 0.33,
) -> dict[str, dict]:
    """Load clusterless waveform features by brain area.

    Uses ``dj.VirtualModule`` to access the ``UnitWaveformFeatures`` table
    without importing the broken ``spyglass.decoding`` package, then reads
    the analysis NWB files directly with pynwb.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name.
    position_time : NDArray[np.float64]
        Position timestamps for time-bounding.
    remove_coincident : bool, optional
        If True (default), remove cross-probe coincident artifact spikes via
        :func:`detect_coincident_spikes`, evaluated across ALL probes in the
        NWB (not per area), since electrical artifacts span the whole probe.
    spike_closeness_threshold : float, optional
        Coincidence time window (seconds) passed to
        :func:`detect_coincident_spikes`. Default 40 us.
    max_coincident_fraction : float, optional
        Fraction-of-probes threshold passed to
        :func:`detect_coincident_spikes`. Default 0.33.

    Returns
    -------
    dict[str, dict]
        Brain area name → ``{"spike_times": [...], "waveform_features": [...]}``.
        Only areas with waveform feature data are included.
    """
    import datajoint as dj

    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v1.curation import CurationV1
    from spyglass.spikesorting.v1.recording import SpikeSortingRecordingSelection
    from spyglass.spikesorting.v1.sorting import SpikeSortingSelection

    # Access UnitWaveformFeatures via VirtualModule to bypass the broken
    # spyglass.decoding import (non_local_detector API incompatibility).
    wf_schema = dj.VirtualModule("wf", "decoding_waveform_features")
    UnitWaveformFeatures = wf_schema.UnitWaveformFeatures

    area_map = get_sort_group_brain_areas(nwb_file_name)
    sg_to_area: dict[int, str] = {}
    for area_name, sg_ids in area_map.items():
        for sg_id in sg_ids:
            sg_to_area[sg_id] = area_name

    # Get all recordings for this NWB file
    rec_sel = SpikeSortingRecordingSelection & {"nwb_file_name": nwb_file_name}
    rec_rows = rec_sel.fetch("recording_id", "sort_group_id", as_dict=True)

    t_start, t_end = float(position_time[0]), float(position_time[-1])

    # Collect one entry per probe (sort group), tracking its brain area, so
    # coincident-artifact detection can run across ALL probes before binning.
    probe_areas: list[str] = []
    probe_spike_times: list[NDArray[np.float64]] = []
    probe_features: list[NDArray[np.float64]] = []

    for rec_row in rec_rows:
        sort_group_id = int(rec_row["sort_group_id"])
        area_name = sg_to_area.get(sort_group_id, "unknown")
        recording_id = str(rec_row["recording_id"])

        sort_sel = SpikeSortingSelection & {"recording_id": recording_id}
        sorting_ids = sort_sel.fetch("sorting_id")

        for sorting_id in sorting_ids:
            curations = CurationV1 & {"sorting_id": str(sorting_id)}
            if len(curations) == 0:
                continue
            max_curation_id = max(curations.fetch("curation_id"))

            curation_key = {
                "sorting_id": str(sorting_id),
                "curation_id": max_curation_id,
            }
            merge_entries = SpikeSortingOutput.CurationV1 & curation_key
            if len(merge_entries) == 0:
                continue

            merge_id = merge_entries.fetch1("merge_id")

            # Check if waveform features exist for this merge_id
            wf_entries = UnitWaveformFeatures & {
                "spikesorting_merge_id": merge_id
            }
            if len(wf_entries) == 0:
                continue

            row = wf_entries.fetch1()

            try:
                spike_times_list, features_list = _read_waveform_features_from_nwb(
                    row["analysis_file_name"], row["object_id"]
                )
            except Exception as e:
                logger.warning(
                    "Failed to read waveform features for %s: %s",
                    row["analysis_file_name"],
                    e,
                )
                continue

            # Filter to position time bounds, one probe entry per unit.
            for st, feat in zip(spike_times_list, features_list):
                mask = (st >= t_start) & (st <= t_end)
                if mask.any():
                    probe_areas.append(area_name)
                    probe_spike_times.append(st[mask])
                    probe_features.append(feat[mask])

    # Cross-probe coincident artifact removal (over all probes at once).
    if remove_coincident and len(probe_spike_times) >= 2:
        n_before = sum(len(st) for st in probe_spike_times)
        filtered_st, kept_idx = detect_coincident_spikes(
            probe_spike_times,
            spike_closeness_threshold=spike_closeness_threshold,
            max_coincident_fraction=max_coincident_fraction,
        )
        if filtered_st:
            probe_spike_times = filtered_st
            probe_features = [
                feat[idx] for idx, feat in zip(kept_idx, probe_features)
            ]
            n_after = sum(len(st) for st in probe_spike_times)
            logger.info(
                "Coincident-artifact removal: %d/%d spikes dropped (%.2f%%) "
                "across %d probes",
                n_before - n_after,
                n_before,
                100.0 * (n_before - n_after) / max(n_before, 1),
                len(probe_spike_times),
            )

    # Bin probes back to brain areas, dropping any emptied probes.
    result: dict[str, dict] = defaultdict(
        lambda: {"spike_times": [], "waveform_features": []}
    )
    for area_name, st, feat in zip(probe_areas, probe_spike_times, probe_features):
        if len(st) > 0:
            result[area_name]["spike_times"].append(st)
            result[area_name]["waveform_features"].append(feat)

    return {k: v for k, v in result.items() if v["spike_times"]}


def load_spikes_by_area(
    nwb_file_name: str,
    position_time: NDArray[np.float64],
    brain_areas: dict[str, dict] | None = None,
    remove_coincident: bool = True,
    spike_closeness_threshold: float = 0.00004,
    max_coincident_fraction: float = 0.33,
) -> dict[str, dict]:
    """Load spike data for all brain areas in an NWB file.

    For each brain area, returns sorted spikes (from SortedSpikesGroup when
    available, otherwise built from CurationV1 chain) and clusterless
    waveform features (when available).

    Parameters
    ----------
    nwb_file_name : str
        NWB file name, e.g. ``"SC10420250628_.nwb"``.
    position_time : NDArray[np.float64]
        Position timestamps from the position DataFrame index. Spike times
        are bounded to ``[position_time[0], position_time[-1]]``.
    brain_areas : dict[str, dict] or None, optional
        Brain area config from datasets.yaml. Maps area name →
        ``{"sorted_spikes_group_name": "..."}``. When provided, only
        the specified SortedSpikesGroups are loaded. When None, all
        available groups are loaded (or built from scratch).
    remove_coincident : bool, optional
        If True (default), remove cross-probe coincident artifacts from the
        clusterless data (see :func:`detect_coincident_spikes`). Sorted spikes
        are not affected.
    spike_closeness_threshold : float, optional
        Coincidence time window (seconds). Default 40 us.
    max_coincident_fraction : float, optional
        Fraction-of-probes threshold for the coincidence filter. Default 0.33.

    Returns
    -------
    dict[str, dict]
        Brain area name → dict with optional keys:

        - ``"sorted"``: ``{"spike_times": list[NDArray]}``
        - ``"clusterless"``: ``{"spike_times": list[NDArray], "waveform_features": list[NDArray]}``

        Either key may be absent if that data type doesn't exist for the area.
    """
    # Try SortedSpikesGroup first, fall back to building from scratch
    sorted_by_area = _load_sorted_spikes_from_group(
        nwb_file_name, position_time, brain_areas=brain_areas
    )
    if sorted_by_area is None:
        logger.info(
            "No SortedSpikesGroup for %s, building from CurationV1 chain",
            nwb_file_name,
        )
        sorted_by_area = _load_sorted_spikes_from_scratch(
            nwb_file_name, position_time
        )

    # Load clusterless waveform features
    clusterless_by_area = _load_clusterless(
        nwb_file_name,
        position_time,
        remove_coincident=remove_coincident,
        spike_closeness_threshold=spike_closeness_threshold,
        max_coincident_fraction=max_coincident_fraction,
    )

    # Assemble result
    result: dict[str, dict] = {}

    for area_name, spike_times in sorted_by_area.items():
        if spike_times:
            result.setdefault(area_name, {})["sorted"] = {
                "spike_times": spike_times,
            }

    for area_name, clusterless_data in clusterless_by_area.items():
        result.setdefault(area_name, {})["clusterless"] = clusterless_data

    # Log summary
    for area_name, area_data in result.items():
        parts = []
        if "sorted" in area_data:
            parts.append(
                f"{len(area_data['sorted']['spike_times'])} sorted units"
            )
        if "clusterless" in area_data:
            parts.append(
                f"{len(area_data['clusterless']['spike_times'])} clusterless units"
            )
        logger.info("%s: %s", area_name, ", ".join(parts))

    return result
