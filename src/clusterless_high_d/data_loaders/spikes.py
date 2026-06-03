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
from numpy.typing import NDArray

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
    result: dict[str, dict] = defaultdict(
        lambda: {"spike_times": [], "waveform_features": []}
    )

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

            # Filter to position time bounds and add to result
            for st, feat in zip(spike_times_list, features_list):
                mask = (st >= t_start) & (st <= t_end)
                filtered_st = st[mask]
                if len(filtered_st) > 0:
                    result[area_name]["spike_times"].append(filtered_st)
                    result[area_name]["waveform_features"].append(feat[mask])

    # Remove areas with no data
    return {k: v for k, v in result.items() if v["spike_times"]}


def load_spikes_by_area(
    nwb_file_name: str,
    position_time: NDArray[np.float64],
    brain_areas: dict[str, dict] | None = None,
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
    clusterless_by_area = _load_clusterless(nwb_file_name, position_time)

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
