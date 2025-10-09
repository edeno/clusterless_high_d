## Overview

Spyglass is a neuroscience data analysis framework that:

- Uses **DataJoint** for reproducible data pipeline management
- Stores data in **NWB (Neurodata Without Borders)** format
- Provides standardized pipelines for common neuroscience analyses
- Ensures reproducible research through automated dependency tracking
- Supports collaborative data sharing and analysis

### DataJoint Foundation

- All tables inherit from DataJoint table types: `Manual`, `Lookup`,
  `Computed`, `Imported`
- Tables are organized into **schemas** (MySQL databases) by topic
- Primary keys establish relationships and dependencies between tables
- The `@schema` decorator associates tables with their schema

## Table Types and Patterns

### 1. Parameters Tables

Store analysis parameters for reproducible computations.

**Conventions:**

- Name ends with `Parameters` or `Params`
- Usually `dj.Lookup` (for predefined values) or `dj.Manual`
- Primary key: `{pipeline}_params_name: varchar(32)`
- Parameters stored as: `{pipeline}_params: blob` (Python dictionary)

**Example:**

```python
@schema
class MyAnalysisParameters(SpyglassMixin, dj.Lookup):
    definition = """
    analysis_params_name: varchar(32)
    ---
    analysis_params: blob
    """

    contents = [
        ["default", {"threshold": 0.5, "window_size": 100}],
        ["strict", {"threshold": 0.8, "window_size": 50}],
    ]

    @classmethod
    def insert_default(cls):
        cls().insert(rows=cls.contents, skip_duplicates=True)
```

### 2. Selection Tables

Pair data with parameters for analysis.

**Conventions:**

- Name ends with `Selection`
- Usually `dj.Manual`
- Foreign keys to data source and parameters tables
- Sets up what will be computed

**Example:**

```python
@schema
class MyAnalysisSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> DataSourceTable
    -> MyAnalysisParameters
    ---
    """
```

### 3. Computed Tables

Store analysis results.

**Conventions:**

- Usually `dj.Computed`
- Foreign key to corresponding Selection table
- Include `analysis_file_name` from `AnalysisNwbfile`
- Implement `make()` method for computation

**Example:**

```python
@schema
class MyAnalysisOutput(SpyglassMixin, dj.Computed):
    definition = """
    -> MyAnalysisSelection
    ---
    -> AnalysisNwbfile
    """

    def make(self, key):
        # Get parameters and data
        params = (MyAnalysisParameters & key).fetch1("analysis_params")

        # Perform analysis
        result = perform_analysis(data, **params)

        # Save to NWB file and get analysis_file_name
        analysis_file_name = create_analysis_nwb_file(result, key)

        # Insert result
        self.insert1({**key, "analysis_file_name": analysis_file_name})
```

### 4. NWB Ingestion Tables

Import data from NWB files.

**Conventions:**

- Usually `dj.Imported`
- Include `object_id` to track NWB objects
- Implement `make()` and `fetch_nwb()` methods

### 5. Merge Tables

Combine outputs from different pipeline versions.

**Conventions:**

- Name ends with `Output`
- Inherit from custom `Merge` class
- Enable unified downstream processing

## Table-Joining Protocol

When connecting two tables:

1. Use `.describe()` or `.heading` to view primary keys (PK) and foreign keys (FK).
2. Look for direct FK. If none, check for linkers via `.parents()`/`.children()`.
3. Chain joins, confirming keys at each step.
4. Smoke-test: `.fetch(limit=1, as_dict=True)`.
5. If stuck, check for overlapping PK/FKs among all tables; explain attempts; suggest checking docs.

## Knowledge Base

### CORE API - use these commands to access data in Spyglass

| Command / Method           | Origin        | 1-line purpose                                                                        | Tutorial-style example\*                                                                                            |
| -------------------------- | ------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `&` (restriction)          | **DataJoint** | Filter rows by key-dict or SQL.                                                       | `(Session & {'nwb_file_name': 'j1620210710_.nwb'}).fetch1()`                                                        |
| `.proj()`                  | **DataJoint** | Rename / drop attrs before join.                                                      | `(Session * Subject).proj(session_date='session_start_time').fetch(limit=3)`                                        |
| `.fetch()` / `.fetch1()`   | **DataJoint** | Materialise query (arrays / dicts / DataFrames).                                      | `(IntervalList & key).fetch('valid_times', limit=2)`                                                                |
| `.describe()`              | **DataJoint** | Show schema, PKs, docstring.                                                          | `PositionOutput.describe()`                                                                                         |
| `.aggr()`                  | **DataJoint** | On-the-fly aggregation.                                                               | `Spikes.aggr(IntervalList, n='count(*)').fetch(limit=1)`                                                            |
| `dj.U()`                   | **DataJoint** | Union of two (or more) relations.                                                     | `task_or_sleep = dj.U((IntervalList & 'interval_list_name="task"'), (IntervalList & 'interval_list_name="sleep"'))` |
| `dj.AndList()`             | **DataJoint** | OR-restrict using list of dicts.                                                      | `files = Session & dj.AndList([{'nwb_file_name': 'fileA.nwb'}, {'nwb_file_name': 'fileB.nwb'}])`                    |
| `.heading`                 | **DataJoint** | Python attr: dict of all columns.                                                     | `Session.heading`                                                                                                   |
| `<<` (up-stream)           | **Spyglass**  | Restrict by **ancestor** attribute (shorthand for `restrict_by(direction="up")`).     | `PositionOutput() << "nwb_file_name = 'j1620210710_.nwb'"`                                                          |
| `>>` (down-stream)         | **Spyglass**  | Restrict by **descendant** attribute (shorthand for `restrict_by(direction="down")`). | `Session() >> 'trodes_pos_params_name="default"'`                                                                   |
| `.restrict_by()`           | **Spyglass**  | Explicit long-distance restrict; choose `"up"`/`"down"`.                              | `PositionOutput().restrict_by("nwb_file_name = 'j1620210710_.nwb'", direction="up")`                                |
| `.merge_restrict()`        | **Spyglass**  | Union of master + parts.                                                              | `PositionOutput.merge_restrict()`                                                                                       |
| `.merge_fetch()`           | **Spyglass**  | Fast cross-part fetch.                                                                | `PositionOutput.merge_fetch({'nwb_file_name': 'j1620210710_.nwb'})`                                                 |
| `.fetch_nwb()`             | **Spyglass**  | Load raw/analysis NWB as `h5py.File`.                                                 | `(LFPOutput & part_key).fetch_nwb()`                                                                                |
| `.fetch1_dataframe()`      | **Spyglass**  | First matching row → tidy `pandas.DataFrame`.                                         | `(PositionOutput & part_key).fetch1_dataframe()`                                                                    |
| `.fetch_pose_dataframe()`  | **Spyglass**  | Pose key-points DF.                                                                   | `(PositionOutput & part_key).fetch_pose_dataframe(bodypart='nose')`                                                 |
| `fetch_results`            | **Spyglass**  | Return decoder results dict/array.                                                    | `results = (DecodingOutput & part_key).fetch_results()`                                                             |
| `get_restricted_merge_ids` | **Spyglass**  | Map friendly keys → merge\_ids (spikes).                                              | `ids = SpikeSortingOutput.get_restricted_merge_ids(key)`                                                            |

### Spyglass — Common-schema “must-know” tables & key fields

| Schema.Table                                | Primary key(s)                                         | Why a beginner needs it                                          |
| ------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------- |
| **common\_nwbfile.Nwbfile**                 | `nwb_file_name`                                        | Registry of every *raw* NWB file — all pipelines anchor here.    |
| **common\_nwbfile.AnalysisNwbfile**         | `analysis_file_abs_path`                               | Tracks derived NWB files (filtered LFP, decoding, …).            |
| **common\_session.Session**                 | `nwb_file_name`                                        | One row per recording; first thing you restrict on.              |
| └─ *Session.Experimenter*                   | `nwb_file_name`, `lab_member_name`                     | Maps each session to its **LabMember** experimenters             |
| └─ *Session.DataAcquisitionDevice*          | `nwb_file_name`, `data_acquisition_device_name`        | Lists headstages / DAQs used in that session                     |
| **common\_interval.IntervalList**           | `nwb_file_name`, `interval_list_name`                  | Time-windows (task, sleep, artefact) that gate every analysis.   |
| **common\_subject.Subject**                 | `subject_id`                                           | Animal metadata; auto-added from NWB.                            |
| **common\_lab.LabMember**                   | `lab_member_name`                                      | People registry; used in Session.Experimenter and permissions    |
| **common\_lab.LabTeam** (+ *LabTeamMember*) | `team_name`                                            | Group members so collaborators can curate/delete their own data. |
| **common\_lab.Institution**                 | `institution_name`                                     | Lookup referenced in `Session.institution_name`.                 |
| **common\_lab.Lab**                         | `lab_name`                                             | Lookup referenced in `Session.lab_name`.                         |
| **common\_device.DataAcquisitionDevice**    | `data_acquisition_device_name`                         | Amplifier / digitiser catalogue.                                 |
| **common\_device.CameraDevice**             | `camera_name`                                          | Camera hardware; referenced by **TaskEpoch**.                    |
| **common\_device.ProbeType**                | `probe_type`                                           | Defines shank count, site spacing, manufacturer.                 |
| **common\_device.Probe**                    | `probe_id`                                             | Physical probe instances linked to sessions.                     |
| **common\_region.BrainRegion**              | `region_id`                                            | Standardised anatomical labels                                   |
| **common\_ephys.ElectrodeGroup**            | `nwb_file_name`, `electrode_group_name`                | Groups channels on a probe                                       |
| **common\_ephys.Electrode**                 | `nwb_file_name`, `electrode_id`                        | Channel-level metadata (coords, region)                          |
| **common\_ephys.Raw**                       | `nwb_file_name`, `interval_list_name`, `raw_object_id` | Entry point for raw ElectricalSeries upstream of LFP & spikes    |
| **common\_filter.FirFilterParameters**      | `filter_name`, `filter_sampling_rate`                  | Library of standard FIR kernels (θ, γ, ripple)                   |
| **common\_position.IntervalPositionInfo**   | `nwb_file_name`, `interval_list_name`                  | Links raw pose series to analysis epochs.                        |
| **common\_sensors.SensorData**              | `nwb_file_name`, `sensor_data_object_id`               | Generic analog/IMU channels.                                     |
| **common\_dio.DIOEvents**                   | `dio_event_name`                        | TTL pulses & sync lines for behavior timestamping.              |
| **common\_task.Task**                       | `task_name`                                            | Lookup of behavioral task definitions.                          |
| **common\_task.TaskEpoch**                  | `nwb_file_name`, `epoch`                               | Maps each epoch to a `Task`, `CameraDevice`, and `IntervalList`  |

| Table (module path)                                          | Primary key(s) you **always** need                                     | Description                                                 |
| ------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **position.PositionOutput** *(merge master)*                 | `merge_id`                                                   | Final XY/θ trajectories; part tables `TrodesPosV1`, `DLCPosV1`.                |
| **lfp.LFPOutput** *(merge master)*                           | `merge_id`                                                   | Band-limited LFP; main part `LFPV1`.                                           |
| **spikesorting.spikesorting_merge.SpikeSortingOutput** *(merge master)*         | `merge_id`                                                   | Curated spike times; part `CurationV1`.                                        |
| **spikesorting.analysis.v1.group.SortedSpikesGroup**                               | `nwb_file_name`, `sorted_spikes_group_name`, `unit_filter_params_name` | Bundles curated units for ensemble analyses or decoding.                       |
| **ripple.v1.ripple.RippleTimesV1**                                  | `lfp_merge_id`,`filter_name`,`filter_sampling_rate`,`nwb_file_name`,`target_interval_list_name`,`lfp_band_sampling_rate`,`group_name`,`ripple_param_name`,`pos_merge_id`                     | Start/stop of hippocampal sharp-wave ripples (needs LFP ripple-band + speed).  |
| **mua.v1.mua.MuaEventsV1**                                   | `mua_param_name`, `nwb_file_name`,`unit_filter_params_name`,`sorted_spikes_group_name`, `pos_merge_id`, `detection_interval`                 | Multi-unit burst events (+ helper to fetch firing-rate & speed).               |
| **decoding.decoding\_merge.DecodingOutput** *(merge master)* | `merge_id`                                                   | Decoding posteriors; parts `ClusterlessDecodingV1`, `SortedSpikesDecodingV1`.  |
| **behavior.v1.core.PoseGroup**                               | `nwb_file_name`, `pose_group_name`                                     | Key-point cohorts that drive MoSeq or other pose analyses.
| **sharing.AnalysisNwbfileKachery**                   | `kachery_zone_name`, `analysis_file_name`                                              | Analysis NWB files stored in Kachery; used for fetching NWB data.               |

### Mini Glossary (symbols tutor should recognize)

- **Output table** – merge table ending in `Output`; single, versioned endpoint for downstream analysis. A ‘master’ table with a DataJoint ‘part’ table connected to the endpoint of each available pipeline e.g. `LFPOutput`
- **Group table** – table ending in `Group` that groups rows for another table for easy usage e.g. `SortedSpikesGroup` groups a set of spike sorting analyses.
- **Long-distance restriction** – `<<` (up-stream), `>>` (down-stream) operators that filter based on attributes several joins away.
- **`fetch_nwb`** – returns an `h5py.File`-like NWBFile; auto-detects raw vs analysis files.
- **`fetch1_dataframe`** – returns a `pandas.DataFrame` for the first matching row.

### Database Exploration (for tutor use)

For tables outside the knowledge base, you can internally use:

- `table.children(as_objects=bool)` : returns a list of all tables (or table names) with a foreign key reference to `table`
- `table.parents(as_objects=bool)` : returns a list of all upstream tables (or table names) on which `table` depends on through a foreign key reference
- `table.heading` : return the list of keys defining the entries of a table.
- `dj.FreeTable(dj.conn(), full_table_name)` : returns a table object corresponding to the database table `full_table_name`. Allows access to table operations without requiring access to the python class.

## Common Linkers (memorize!)

- Session ↔ ProbeType: via ElectrodeGroup, Probe
- Session ↔ BrainRegion: via ElectrodeGroup, Electrode
- Session ↔ Task: via TaskEpoch
- Session ↔ CameraDevice: via TaskEpoch
- Session ↔ LabMember: via Session.Experimenter (part table)

*(“via” = join through these)*

### Schema pre-fixes

The Spyglass pipeline is organized into schemas, each with a specific focus. Here are the main schemas and their purposes:

- `common_*` – contains common data structures and utilities used across the pipeline, such as raw electrophysiology data.
- `lfp_*` – contains tables related to local field potentials (LFP) analysis
- `position_*` – contains tables related to position data analysis, including raw and processed position data.
- `spikesorting_*` – contains tables related to spike sorting, including raw spike data
- `decoding_*` – contains tables related to decoding analyses, such as decoding neural activity
- `position_linearization_*` – contains tables related to linearizing position data for analysis along a track
- `ripple_*` – contains tables related to ripple analysis, which is often used in conjunction with LFP data.

### Merge tables

Analogy: PositionOutput (any *Output) is a Table-of-Contents; each part table is a chapter, and merge_id is the bookmark.
Discover chapters: dir(PositionOutput) → pick one, then PositionOutput.TrodesPosV1.describe() to see required keys.
Build your search: key = {"nwb_file_name": "...", ...}.
Get the bookmark: merge_key = PositionOutput.merge_get_part(key).fetch1("KEY"); fetch data with (PositionOutput & merge_key).fetch1_dataframe().
Never type merge_id by hand—lookup with merge_get_part (or get_restricted_merge_ids) and preview everything with merge_view().

- **`merge_view()`**: "To quickly peek at the combined data from all pipelines and see what columns are available, use `merge_view()`."
- **`get_restricted_merge_ids()`**: "For some tables like `SpikeSortingOutput`, there are powerful shortcuts that do the lookup for you. `get_restricted_merge_ids(key)` can often replace steps 3 and 4."

### Quick examples for grabbing data (for tutor use)

```python
from spyglass.common import Session, IntervalList
from spyglass.position import PositionOutput
from spyglass.lfp import LFPOutput
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

nwb_file_name = "j1620210710_.nwb"
Session & {"nwb_file_name": nwb_file_name}
(IntervalList & {"nwb_file_name": nwb_file_name, "interval_list_name": "02_r1"}).fetch(
    "valid_times"
)
key = {
    "nwb_file_name": nwb_file_name,
    "interval_list_name": "pos 1 valid times",
    "trodes_pos_params_name": "default",
}

merge_key = (PositionOutput.merge_get_part(key)).fetch1("KEY")
# use this to restrict PositionOutput and fetch the data
position_info = (
    (PositionOutput & merge_key).fetch1_dataframe().loc[:, ["position_x", "position_y"]]
)
key = {
    "nwb_file_name": nwb_file_name,
    "lfp_electrode_group_name": "lfp_tets_j16",
    "target_interval_list_name": "02_r1 noPrePostTrialTimes",
    "filter_name": "LFP 0-400 Hz",
    "filter_sampling_rate": 30000,
}

merge_key = (LFPOutput.merge_get_part(key)).fetch1("KEY")
lfp_data = (LFPOutput & merge_key).fetch1_dataframe()

SpikeSortingOutput().get_spike_times(
    {"merge_id": "0164f4ef-8f78-c9a7-d50e-72c699bbbffc"}
)[0]

spike_times, unit_ids = SortedSpikesGroup.fetch_spike_data(
    {
        "nwb_file_name": nwb_file_name,
        "unit_filter_params_name": "all_units",
        "sorted_spikes_group_name": "HPC_02_r1_clusterless",
    },
    return_unit_ids=True,
)

```
