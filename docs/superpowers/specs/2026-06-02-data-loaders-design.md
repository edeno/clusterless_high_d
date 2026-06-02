# Data Loaders Design Spec

## Goal

Build data loading functions that fetch all data needed for decoding (clusterless and sorted) from Spyglass, organized by brain area. Target dataset: SC10420250628_.nwb.

## Module Layout

```
src/clusterless_high_d/data_loaders/
├── __init__.py          # Lazy imports (no DJ connection at import time)
├── position.py          # load_position()
├── spikes.py            # load_spikes_by_area()
└── utils.py             # Shared helpers (brain area mapping, interval lookup)
```

## Functions

### `load_position(nwb_file_name, interval_list_name, source="TrodesPosV1")`

**Module**: `position.py`

**Parameters**:
- `nwb_file_name`: e.g. `"SC10420250628_.nwb"`
- `interval_list_name`: e.g. `"pos 0 valid times"` — used to select the position epoch
- `source`: `"TrodesPosV1"` or `"DLCPosV1"` — position tracking source

**Returns**: `pd.DataFrame` — time-indexed with columns: `position_x`, `position_y`, `orientation`, `velocity_x`, `velocity_y`, `speed`. NaN rows dropped.

**Implementation**:
1. `PositionOutput.merge_get_part({"nwb_file_name": ..., "source": ..., "interval_list_name": ...}, join_master=True)`
2. Verify `len() == 1` before `fetch1_dataframe()`
3. Drop NaN rows, return DataFrame

**Spyglass tables**: `PositionOutput` (merge), `TrodesPosV1` or `DLCPosV1` (part).

---

### `load_spikes_by_area(nwb_file_name, position_time)`

**Module**: `spikes.py`

**Parameters**:
- `nwb_file_name`: e.g. `"SC10420250628_.nwb"`
- `position_time`: `np.ndarray` of timestamps from position DataFrame index — used to bound spike times to `[position_time[0], position_time[-1]]`

**Returns**: `dict[str, AreaSpikeData]` keyed by brain area name (e.g. `"Cornu ammonis 1 (CA1)"`, `"Cingulate region (CgR)"`).

```python
AreaSpikeData = {
    "sorted": {                              # present if sorted spikes exist
        "spike_times": list[np.ndarray],     # per unit
    },
    "clusterless": {                         # present if waveform features exist
        "spike_times": list[np.ndarray],     # per sort group/unit
        "waveform_features": list[np.ndarray],  # per sort group/unit, shape (n_spikes, n_features)
    },
}
```

Either key may be absent if that data type doesn't exist for the area.

**Implementation**:

#### Step 1: Map sort groups to brain areas

Chain: `SortGroup.SortGroupElectrode` → `ElectrodeGroup` → `BrainRegion`.

```
sort_group_id → electrode_group_name → region_id → region_name
```

Result: `dict[str, list[int]]` mapping region name → list of sort_group_ids.

Helper: `get_sort_group_brain_areas(nwb_file_name)` in `utils.py`.

#### Step 2: Load sorted spikes

**If `SortedSpikesGroup` entries exist for this NWB file**: use them directly.

1. Query `SortedSpikesGroup & {"nwb_file_name": nwb_file_name}`
2. Group names encode brain area (e.g. `"CA1_only_accept"`, `"PFC_exclude_all_other"`)
3. For each group, fetch spike times via the group's `fetch_spike_data()` method (or equivalent)
4. Map group names to canonical brain area names using the sort group → brain area mapping from Step 1

**If no `SortedSpikesGroup` entries exist**: build from scratch.

1. For each brain area's sort groups, trace:
   `SpikeSortingRecordingSelection` → `SpikeSortingSelection` → `CurationV1` → `SpikeSortingOutput.CurationV1`
2. Fetch spike times from the NWB objects referenced by the merge entries
3. Use the highest `curation_id` per sorting (most curated version)

#### Step 3: Load clusterless waveform features

1. From the `SpikeSortingOutput` merge IDs obtained in Step 2 (or traced fresh), check which have entries in `UnitWaveformFeatures`
2. For those that do, fetch the waveform features (spike times + feature arrays) from the referenced NWB analysis files

#### Step 4: Filter spike times to position bounds

Same pattern as continuum-swr's `filter_spike_times`:
- Keep only spikes where `position_time[0] <= t <= position_time[-1]`
- Drop units/sort groups with zero spikes after filtering
- Maintain alignment between `spike_times` and `waveform_features` lists

---

### `get_sort_group_brain_areas(nwb_file_name)`

**Module**: `utils.py`

**Returns**: `dict[str, list[int]]` — brain area name → sort_group_ids.

**Implementation**: `SortGroup.SortGroupElectrode` → `ElectrodeGroup` → `BrainRegion` join.

---

## Design Decisions

1. **Lazy Spyglass imports**: All `spyglass.*` and `datajoint` imports inside function bodies, not at module top level. Allows importing constants and type hints without a DB connection.

2. **Position loaded separately**: Position is independent of brain area and always required. Caller passes `position_time` to `load_spikes_by_area` for time-bounding.

3. **SortedSpikesGroup preferred when available**: Already curated, brain-area-organized. Fall back to building from scratch when absent.

4. **Brain areas discovered, not hardcoded**: Different datasets have different regions. The sort group → electrode group → brain region chain handles this generically.

5. **Both sorted and clusterless per area**: Enables decoder comparison. Either may be absent for a given area.

6. **spyglass.decoding import avoided**: The module is broken with the latest non_local_detector. We import only from `spyglass.spikesorting`, `spyglass.position`, and `spyglass.common`.

## Constraints

- The `spyglass.decoding` module cannot be imported (broken by non_local_detector API change). All clusterless waveform feature access must go through direct DB queries or the `UnitWaveformFeatures` table imported carefully.
- SC10420250628 is the primary development target. SC10420250616 lacks position and waveform features. L1420240604 has waveform features but no SortedSpikesGroup.

## Testing Plan

1. Smoke test `load_position` on SC10420250628 with `"pos 0 valid times"` / `TrodesPosV1`
2. Smoke test `load_spikes_by_area` on SC10420250628 — verify both CA1 and CgR areas returned with sorted spikes
3. Verify clusterless data present for areas that have waveform features
4. Verify spike time filtering matches position bounds
5. Test fallback path (no SortedSpikesGroup) on SC10420250616 or L1420240604
