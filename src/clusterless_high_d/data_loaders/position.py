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
