# Data Loaders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build data loading functions that fetch position and spike data (sorted + clusterless) from Spyglass, organized by brain area.

**Architecture:** Three modules under `src/clusterless_high_d/data_loaders/`: `utils.py` for brain area discovery, `position.py` for position loading, `spikes.py` for spike loading. All Spyglass imports are lazy (inside function bodies). Data flows: position loaded first → its time index bounds spike times.

**Tech Stack:** Spyglass (DataJoint), NumPy, Pandas

**Target dataset:** SC10420250628_.nwb (has SortedSpikesGroup, waveform features, and Trodes position)

**Important:** The `spyglass.decoding` module cannot be imported at the top level — it crashes due to a `non_local_detector` API change. The `UnitWaveformFeatures` table lives in `spyglass.decoding.v1.waveform_features` but importing the parent package fails. We must import it carefully (see Task 4).

---

### Task 1: Create module skeleton and utils

**Files:**
- Create: `src/clusterless_high_d/data_loaders/__init__.py`
- Create: `src/clusterless_high_d/data_loaders/utils.py`

- [ ] **Step 1: Create `__init__.py` with lazy imports**

```python
"""Data loading module for clusterless high-d decoding.

Loader functions require a Spyglass/DataJoint connection. They are
exposed via PEP 562 ``__getattr__`` so that importing this package
does not open a connection unless a loader is actually called.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_LAZY_EXPORTS = {
    "load_position": "clusterless_high_d.data_loaders.position",
    "load_spikes_by_area": "clusterless_high_d.data_loaders.spikes",
    "get_sort_group_brain_areas": "clusterless_high_d.data_loaders.utils",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


if TYPE_CHECKING:
    from .position import load_position  # noqa: F401
    from .spikes import load_spikes_by_area  # noqa: F401
    from .utils import get_sort_group_brain_areas  # noqa: F401
```

- [ ] **Step 2: Create `utils.py` with `get_sort_group_brain_areas`**

```python
"""Shared helpers for data loading.

Functions
---------
get_sort_group_brain_areas
    Map sort group IDs to brain area names for a given NWB file.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def get_sort_group_brain_areas(nwb_file_name: str) -> dict[str, list[int]]:
    """Map sort group IDs to brain area names.

    Traces the chain: SortGroup.SortGroupElectrode → ElectrodeGroup → BrainRegion.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name, e.g. ``"SC10420250628_.nwb"``.

    Returns
    -------
    dict[str, list[int]]
        Brain area name → list of sort_group_ids.
        Example: ``{"Cornu ammonis 1 (CA1)": [7, 8, 9, ...], ...}``
    """
    from spyglass.common import BrainRegion, ElectrodeGroup
    from spyglass.spikesorting.v1.recording import SortGroup

    # sort_group_id → electrode_group_name (take first electrode's group)
    sg_electrodes = (
        SortGroup.SortGroupElectrode & {"nwb_file_name": nwb_file_name}
    ).fetch("sort_group_id", "electrode_group_name", as_dict=True)

    sg_to_eg: dict[int, str] = {}
    for row in sg_electrodes:
        sg_id = int(row["sort_group_id"])
        if sg_id not in sg_to_eg:
            sg_to_eg[sg_id] = row["electrode_group_name"]

    # electrode_group_name → region_id
    eg_regions = (
        ElectrodeGroup & {"nwb_file_name": nwb_file_name}
    ).fetch("electrode_group_name", "region_id", as_dict=True)
    eg_to_region = {r["electrode_group_name"]: int(r["region_id"]) for r in eg_regions}

    # region_id → region_name
    all_region_ids = set(eg_to_region.values())
    region_names: dict[int, str] = {}
    for rid in all_region_ids:
        region_names[rid] = (BrainRegion & {"region_id": rid}).fetch1("region_name")

    # Build result: region_name → [sort_group_ids]
    result: dict[str, list[int]] = defaultdict(list)
    for sg_id, eg_name in sorted(sg_to_eg.items()):
        rid = eg_to_region.get(eg_name)
        rname = region_names.get(rid, "unknown")
        result[rname].append(sg_id)

    return dict(result)
```

- [ ] **Step 3: Verify utils works**

Run:
```bash
uv run python -c "
from clusterless_high_d.data_loaders.utils import get_sort_group_brain_areas
areas = get_sort_group_brain_areas('SC10420250628_.nwb')
for area, sgs in areas.items():
    print(f'{area}: {sgs}')
"
```

Expected output (approximate):
```
Cingulate region (CgR): [0, 1, 2, 3, 4, 5, 6, 23]
Cornu ammonis 1 (CA1): [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
```

- [ ] **Step 4: Commit**

```bash
git add src/clusterless_high_d/data_loaders/__init__.py src/clusterless_high_d/data_loaders/utils.py
git commit -m "feat: add data_loaders module skeleton and brain area mapping"
```

---

### Task 2: Position loading

**Files:**
- Create: `src/clusterless_high_d/data_loaders/position.py`

- [ ] **Step 1: Create `position.py`**

```python
"""Position data loading from Spyglass.

Functions
---------
load_position
    Load position tracking data for a given NWB file and epoch.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_position(
    nwb_file_name: str,
    interval_list_name: str,
    source: str = "TrodesPosV1",
) -> pd.DataFrame:
    """Load position data for a given NWB file and epoch.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name, e.g. ``"SC10420250628_.nwb"``.
    interval_list_name : str
        Interval list name selecting the epoch, e.g. ``"pos 0 valid times"``.
    source : str, optional
        Position tracking source. ``"TrodesPosV1"`` or ``"DLCPosV1"``.
        Default ``"TrodesPosV1"``.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with position columns (position_x, position_y,
        orientation, velocity_x, velocity_y, speed). NaN rows dropped.
    """
    from spyglass.position.position_merge import PositionOutput

    restriction = {
        "nwb_file_name": nwb_file_name,
        "source": source,
        "interval_list_name": interval_list_name,
    }

    parts = PositionOutput.merge_get_part(restriction, join_master=True)
    n_parts = len(parts)
    if n_parts == 0:
        raise ValueError(
            f"No PositionOutput entries found for {restriction}"
        )
    if n_parts > 1:
        logger.warning(
            "Found %d PositionOutput entries for %s, using first",
            n_parts,
            restriction,
        )

    merge_id = parts.fetch("merge_id", limit=1)[0]
    position_df = (
        PositionOutput & {"merge_id": merge_id}
    ).fetch1_dataframe()

    return position_df.dropna()
```

- [ ] **Step 2: Smoke test on SC10420250628**

Run:
```bash
uv run python -c "
from clusterless_high_d.data_loaders.position import load_position
pos = load_position('SC10420250628_.nwb', 'pos 0 valid times', source='TrodesPosV1')
print(f'Shape: {pos.shape}')
print(f'Columns: {list(pos.columns)}')
print(f'Time range: {pos.index[0]:.2f} - {pos.index[-1]:.2f}')
print(pos.head(3))
"
```

Expected: DataFrame with position_x, position_y, speed columns, time-indexed, no NaN rows.

- [ ] **Step 3: Commit**

```bash
git add src/clusterless_high_d/data_loaders/position.py
git commit -m "feat: add position data loading from Spyglass"
```

---

### Task 3: Sorted spike loading

**Files:**
- Create: `src/clusterless_high_d/data_loaders/spikes.py`

This task implements `load_spikes_by_area` with sorted spike support only. Clusterless is added in Task 4.

- [ ] **Step 1: Create `spikes.py` with sorted spike loading**

```python
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
) -> dict[str, list[NDArray[np.float64]]] | None:
    """Load sorted spikes using SortedSpikesGroup if available.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name.
    position_time : NDArray[np.float64]
        Position timestamps for time-slicing.

    Returns
    -------
    dict[str, list[NDArray]] or None
        Brain area name → list of spike time arrays per unit.
        Returns None if no SortedSpikesGroup entries exist.
    """
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

    ssg = SortedSpikesGroup & {"nwb_file_name": nwb_file_name}
    if len(ssg) == 0:
        return None

    # Map sort group brain areas to match group names to regions
    area_map = get_sort_group_brain_areas(nwb_file_name)
    # Invert: sort_group_id → area_name
    sg_to_area: dict[int, str] = {}
    for area_name, sg_ids in area_map.items():
        for sg_id in sg_ids:
            sg_to_area[sg_id] = area_name

    groups = ssg.fetch(as_dict=True)
    time_slice = [float(position_time[0]), float(position_time[-1])]

    result: dict[str, list[NDArray[np.float64]]] = defaultdict(list)

    for group in groups:
        group_name = group["sorted_spikes_group_name"]

        # Determine brain area from the group's merge_ids
        from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup as SSG
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
        from spyglass.spikesorting.v1.recording import (
            SpikeSortingRecordingSelection,
        )
        from spyglass.spikesorting.v1.sorting import SpikeSortingSelection

        merge_ids = (
            SSG.Units & {
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


def load_spikes_by_area(
    nwb_file_name: str,
    position_time: NDArray[np.float64],
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
        nwb_file_name, position_time
    )
    if sorted_by_area is None:
        logger.info(
            "No SortedSpikesGroup for %s, building from CurationV1 chain",
            nwb_file_name,
        )
        sorted_by_area = _load_sorted_spikes_from_scratch(
            nwb_file_name, position_time
        )

    # Assemble result
    result: dict[str, dict] = {}

    for area_name, spike_times in sorted_by_area.items():
        if spike_times:
            result.setdefault(area_name, {})["sorted"] = {
                "spike_times": spike_times,
            }

    # Log summary
    for area_name, area_data in result.items():
        parts = []
        if "sorted" in area_data:
            parts.append(f"{len(area_data['sorted']['spike_times'])} sorted units")
        logger.info("%s: %s", area_name, ", ".join(parts))

    return result
```

- [ ] **Step 2: Smoke test sorted spikes on SC10420250628**

Run:
```bash
uv run python -c "
from clusterless_high_d.data_loaders.position import load_position
from clusterless_high_d.data_loaders.spikes import load_spikes_by_area
import logging
logging.basicConfig(level=logging.INFO)

pos = load_position('SC10420250628_.nwb', 'pos 0 valid times')
spikes = load_spikes_by_area('SC10420250628_.nwb', pos.index.to_numpy())

for area, data in spikes.items():
    print(f'{area}:')
    if 'sorted' in data:
        st = data['sorted']['spike_times']
        total = sum(len(s) for s in st)
        print(f'  sorted: {len(st)} units, {total} total spikes')
"
```

Expected: CA1 and CgR areas with sorted units.

- [ ] **Step 3: Commit**

```bash
git add src/clusterless_high_d/data_loaders/spikes.py
git commit -m "feat: add sorted spike loading by brain area"
```

---

### Task 4: Clusterless waveform feature loading

**Files:**
- Modify: `src/clusterless_high_d/data_loaders/spikes.py`

The `spyglass.decoding` package crashes on import due to `non_local_detector` API incompatibility. We need to import `UnitWaveformFeatures` by bypassing the package `__init__.py` using `importlib`.

- [ ] **Step 1: Add `_import_unit_waveform_features` helper and `_load_clusterless` to spikes.py**

Add these functions before `load_spikes_by_area`:

```python
def _import_unit_waveform_features():
    """Import UnitWaveformFeatures bypassing the broken spyglass.decoding __init__.

    The spyglass.decoding package fails to import because
    non_local_detector's API changed. We import the waveform_features
    module directly to avoid triggering that error.

    Returns
    -------
    UnitWaveformFeatures : datajoint.Table
        The UnitWaveformFeatures table class.
    """
    import importlib

    mod = importlib.import_module("spyglass.decoding.v1.waveform_features")
    return mod.UnitWaveformFeatures


def _load_clusterless(
    nwb_file_name: str,
    position_time: NDArray[np.float64],
) -> dict[str, dict]:
    """Load clusterless waveform features by brain area.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name.
    position_time : NDArray[np.float64]
        Position timestamps for time-bounding.

    Returns
    -------
    dict[str, dict]
        Brain area name → {"spike_times": [...], "waveform_features": [...]}.
        Only areas with waveform feature data are included.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v1.curation import CurationV1
    from spyglass.spikesorting.v1.recording import SpikeSortingRecordingSelection
    from spyglass.spikesorting.v1.sorting import SpikeSortingSelection

    try:
        UnitWaveformFeatures = _import_unit_waveform_features()
    except Exception as e:
        logger.warning("Cannot import UnitWaveformFeatures: %s", e)
        return {}

    area_map = get_sort_group_brain_areas(nwb_file_name)
    sg_to_area: dict[int, str] = {}
    for area_name, sg_ids in area_map.items():
        for sg_id in sg_ids:
            sg_to_area[sg_id] = area_name

    # Get all recordings for this NWB file
    rec_sel = SpikeSortingRecordingSelection & {"nwb_file_name": nwb_file_name}
    rec_rows = rec_sel.fetch("recording_id", "sort_group_id", as_dict=True)

    t_start, t_end = float(position_time[0]), float(position_time[-1])
    result: dict[str, dict] = defaultdict(lambda: {"spike_times": [], "waveform_features": []})

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

            # Fetch spike times and features
            try:
                spike_times_list, features_list = wf_entries.fetch_data()
            except Exception as e:
                logger.warning(
                    "Failed to fetch waveform features for merge_id %s: %s",
                    merge_id,
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
```

- [ ] **Step 2: Update `load_spikes_by_area` to include clusterless data**

Replace the `# Assemble result` section and everything after it in `load_spikes_by_area` with:

```python
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
            parts.append(f"{len(area_data['sorted']['spike_times'])} sorted units")
        if "clusterless" in area_data:
            parts.append(
                f"{len(area_data['clusterless']['spike_times'])} clusterless units"
            )
        logger.info("%s: %s", area_name, ", ".join(parts))

    return result
```

- [ ] **Step 3: Test clusterless loading**

Run:
```bash
uv run python -c "
from clusterless_high_d.data_loaders.position import load_position
from clusterless_high_d.data_loaders.spikes import load_spikes_by_area
import logging
logging.basicConfig(level=logging.INFO)

pos = load_position('SC10420250628_.nwb', 'pos 0 valid times')
spikes = load_spikes_by_area('SC10420250628_.nwb', pos.index.to_numpy())

for area, data in spikes.items():
    print(f'{area}:')
    if 'sorted' in data:
        st = data['sorted']['spike_times']
        total = sum(len(s) for s in st)
        print(f'  sorted: {len(st)} units, {total} total spikes')
    if 'clusterless' in data:
        cl = data['clusterless']
        total = sum(len(s) for s in cl['spike_times'])
        feat_shapes = [f.shape for f in cl['waveform_features'][:3]]
        print(f'  clusterless: {len(cl[\"spike_times\"])} units, {total} total spikes')
        print(f'  feature shapes (first 3): {feat_shapes}')
"
```

Expected: At least some areas with clusterless data (SC10420250628 has 7 waveform feature entries).

- [ ] **Step 4: Commit**

```bash
git add src/clusterless_high_d/data_loaders/spikes.py
git commit -m "feat: add clusterless waveform feature loading by brain area"
```

---

### Task 5: Update package __init__.py and integration test

**Files:**
- Modify: `src/clusterless_high_d/__init__.py`

- [ ] **Step 1: Update package `__init__.py`**

The current `__init__.py` imports from deleted modules. Replace it with imports from the new data_loaders:

```python
"""Clusterless high-dimensional decoding for hippocampal datasets."""

__version__ = "0.1.0"
```

Keep it minimal — the data_loaders module uses lazy imports, so users import directly:
```python
from clusterless_high_d.data_loaders import load_position, load_spikes_by_area
```

- [ ] **Step 2: Full integration test**

Run:
```bash
uv run python -c "
from clusterless_high_d.data_loaders import load_position, load_spikes_by_area
import logging
logging.basicConfig(level=logging.INFO)

# Load position
pos = load_position('SC10420250628_.nwb', 'pos 0 valid times')
print(f'Position: {pos.shape}, time range: {pos.index[0]:.2f} - {pos.index[-1]:.2f}')
print(f'Columns: {list(pos.columns)}')
print()

# Load spikes
spikes = load_spikes_by_area('SC10420250628_.nwb', pos.index.to_numpy())
print(f'Brain areas: {list(spikes.keys())}')
for area, data in spikes.items():
    print(f'\n{area}:')
    if 'sorted' in data:
        st = data['sorted']['spike_times']
        total = sum(len(s) for s in st)
        print(f'  sorted: {len(st)} units, {total} total spikes')
        # Verify all spikes within position bounds
        for s in st:
            assert s.min() >= pos.index[0], 'spike before position start'
            assert s.max() <= pos.index[-1], 'spike after position end'
        print('  all spikes within position bounds ✓')
    if 'clusterless' in data:
        cl = data['clusterless']
        total = sum(len(s) for s in cl['spike_times'])
        print(f'  clusterless: {len(cl[\"spike_times\"])} units, {total} total spikes')
        # Verify alignment
        for s, f in zip(cl['spike_times'], cl['waveform_features']):
            assert len(s) == f.shape[0], f'misaligned: {len(s)} vs {f.shape[0]}'
        print('  spike/feature alignment ✓')

print('\nIntegration test passed')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/clusterless_high_d/__init__.py
git commit -m "feat: update package init for new data_loaders module"
```
