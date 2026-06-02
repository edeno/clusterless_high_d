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
