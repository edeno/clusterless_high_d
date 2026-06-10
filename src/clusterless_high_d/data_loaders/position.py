"""Position data loading from Spyglass.

Functions
---------
load_position
    Load position tracking data for a given NWB file and epoch, from either
    TrodesPosV1 (keyed by interval list name) or DLCPosV1 (keyed by epoch).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_trodes_merge_id(
    nwb_file_name: str,
    interval_list_name: str | None,
    epoch: int | None,
) -> str:
    """Resolve a TrodesPosV1 PositionOutput merge_id.

    TrodesPosV1 is keyed by ``interval_list_name`` (e.g. ``"pos 0 valid
    times"``). If only ``epoch`` is given, it is converted to the matching
    position interval name via Spyglass.
    """
    from spyglass.common.common_behav import (
        convert_epoch_interval_name_to_position_interval_name,
    )
    from spyglass.position.position_merge import PositionOutput

    if interval_list_name is None:
        if epoch is None:
            raise ValueError(
                "TrodesPosV1 requires `interval_list_name` (or `epoch`)."
            )
        interval_list_name = convert_epoch_interval_name_to_position_interval_name(
            {"nwb_file_name": nwb_file_name, "epoch": epoch}
        )
        if not interval_list_name:
            raise ValueError(
                f"No TrodesPosV1 position interval for {nwb_file_name} "
                f"epoch {epoch} (epoch may not have processed position)."
            )

    restriction = {
        "nwb_file_name": nwb_file_name,
        "source": "TrodesPosV1",
        "interval_list_name": interval_list_name,
    }
    parts = PositionOutput.merge_get_part(restriction, join_master=True)
    if parts is None or len(parts) == 0:
        raise ValueError(f"No PositionOutput entries found for {restriction}")
    n_parts = len(parts)
    if n_parts > 1:
        logger.warning(
            "Found %d PositionOutput entries for %s, using first",
            n_parts,
            restriction,
        )
    return parts.fetch("merge_id", limit=1)[0]


def _resolve_dlc_merge_id(nwb_file_name: str, epoch: int | None) -> str:
    """Resolve a DLCPosV1 PositionOutput merge_id.

    DLCPosV1 is keyed by ``epoch`` (an int), not by ``interval_list_name``.
    If multiple DLC configurations exist for the epoch (different models or
    parameter sets), the first is used with a warning.
    """
    if epoch is None:
        raise ValueError("DLCPosV1 requires `epoch`.")

    from spyglass.position.position_merge import PositionOutput
    from spyglass.position.v1.position_dlc_selection import DLCPosV1

    dlc = DLCPosV1 & {"nwb_file_name": nwb_file_name, "epoch": epoch}
    n_dlc = len(dlc)
    if n_dlc == 0:
        raise ValueError(
            f"No DLCPosV1 entry for {nwb_file_name} epoch {epoch}."
        )
    if n_dlc > 1:
        logger.warning(
            "Found %d DLCPosV1 entries for %s epoch %d, using first",
            n_dlc,
            nwb_file_name,
            epoch,
        )
    dlc_key = dlc.fetch("KEY", limit=1)[0]
    return (PositionOutput.DLCPosV1 & dlc_key).fetch1("merge_id")


def load_position(
    nwb_file_name: str,
    interval_list_name: str | None = None,
    source: str = "TrodesPosV1",
    epoch: int | None = None,
    dropna_columns: tuple[str, ...] = ("position_x", "position_y"),
    drop_origin: bool = True,
    origin_tol: float = 1e-6,
    max_speed: float | None = None,
    interpolate: bool = False,
    max_pts_to_interp: int = 20,
    max_cm_to_interp: float = 20.0,
) -> pd.DataFrame:
    """Load position data for a given NWB file and epoch.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name, e.g. ``"SC10420250628_.nwb"``.
    interval_list_name : str or None, optional
        Interval list name selecting the epoch for ``TrodesPosV1``, e.g.
        ``"pos 0 valid times"``. Required for ``TrodesPosV1`` unless ``epoch``
        is given. Ignored for ``DLCPosV1``.
    source : str, optional
        Position tracking source: ``"TrodesPosV1"`` or ``"DLCPosV1"``.
        Default ``"TrodesPosV1"``.
    epoch : int or None, optional
        Epoch number. Required for ``DLCPosV1`` (which is keyed by epoch, not
        by interval list name). For ``TrodesPosV1`` it is an alternative to
        ``interval_list_name`` and is converted internally.
    dropna_columns : tuple of str, optional
        Columns that must be non-NaN; rows with NaN in any of them are
        dropped. Default ``("position_x", "position_y")``. Note: dropping on
        all columns is wrong here because ``orientation`` is entirely NaN for
        some Trodes single-LED epochs, which would empty the DataFrame.
    drop_origin : bool, optional
        If True (default), flag samples within ``origin_tol`` of the origin as
        tracking dropouts. DLC encodes dropped/untracked frames as a (0, 0)
        sentinel (exactly 0 for some pipelines, ~1e-14 numerical noise for
        others). Real cm-scale positions never land within a micron of the
        origin, so this is a safe dropout test. Runs of consecutive dropouts
        have ~zero internal speed and so are NOT caught by ``max_speed`` alone.
    origin_tol : float, optional
        Distance from the origin (cm) below which a sample is treated as a
        dropout sentinel. Default ``1e-6`` — tight enough to never match a
        real position, loose enough to catch numerical-noise zeros.
    max_speed : float or None, optional
        If set, flag samples whose ``speed`` exceeds this value (cm/s). DLC
        dropout transitions (and any other tracking jumps) produce physically
        implausible speeds (hundreds of cm/s vs. a rat's ~80-100 cm/s max).
        A threshold such as 100-120 removes those jumps. Default None.
    interpolate : bool, optional
        If True, flagged samples (origin + high-speed) are set to NaN and the
        kinematic columns are linearly interpolated over time across short
        gaps (mirrors Spyglass ``interp_pos``), rather than simply dropped.
        Gaps longer than ``max_pts_to_interp`` (and leading/trailing gaps) are
        left as NaN and then dropped. Default False (drop the flagged samples).
    max_pts_to_interp : int, optional
        Maximum length (in samples) of a consecutive dropout gap to
        interpolate across when ``interpolate=True``. Longer gaps are dropped
        instead of bridged (avoids inventing a straight-line trajectory across
        a long tracking loss). Default 20.
    max_cm_to_interp : float, optional
        Maximum distance (cm) between the samples bracketing a dropout gap for
        that gap to be interpolated when ``interpolate=True``. If the animal
        moved farther than this during the tracking loss, a straight-line
        bridge would be a fabricated diagonal trajectory, so the gap is left
        NaN (and dropped) instead. Mirrors Spyglass ``interp_pos``. Default 20.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with position columns (position_x, position_y,
        orientation, velocity_x, velocity_y, speed). Dropout/jump samples are
        either dropped or interpolated per the options above.
    """
    from spyglass.position.position_merge import PositionOutput

    if source == "DLCPosV1":
        merge_id = _resolve_dlc_merge_id(nwb_file_name, epoch)
    elif source == "TrodesPosV1":
        merge_id = _resolve_trodes_merge_id(
            nwb_file_name, interval_list_name, epoch
        )
    else:
        raise ValueError(f"Unsupported position source: {source!r}")

    position_df = (PositionOutput & {"merge_id": merge_id}).fetch1_dataframe()

    # Flag tracking artifacts: origin sentinels and implausible-speed jumps.
    bad = pd.Series(False, index=position_df.index)
    if drop_origin:
        is_origin = (
            np.hypot(position_df["position_x"], position_df["position_y"])
            < origin_tol
        )
        bad |= is_origin
        if int(is_origin.sum()):
            logger.info(
                "Flagged %d origin dropout-sentinel samples", int(is_origin.sum())
            )
    if max_speed is not None:
        if "speed" not in position_df.columns:
            raise ValueError(
                "max_speed filtering requested but no 'speed' column present."
            )
        is_fast = position_df["speed"] > max_speed
        bad |= is_fast
        if int(is_fast.sum()):
            logger.info(
                "Flagged %d samples with speed > %.1f cm/s", int(is_fast.sum()),
                max_speed,
            )

    kinematic = [
        c
        for c in ("position_x", "position_y", "velocity_x", "velocity_y", "speed")
        if c in position_df.columns
    ]
    position_df.loc[bad, kinematic] = np.nan

    if interpolate and bad.any():
        position_df = _interpolate_dropout_gaps(
            position_df, kinematic, max_pts_to_interp, max_cm_to_interp
        )

    if dropna_columns:
        position_df = position_df.dropna(subset=list(dropna_columns))

    return position_df


def _interpolate_dropout_gaps(
    position_df: pd.DataFrame,
    columns: list[str],
    max_pts_to_interp: int,
    max_cm_to_interp: float,
) -> pd.DataFrame:
    """Linearly interpolate NaN dropout gaps over time, span by span.

    Mirrors Spyglass ``interp_pos``: a gap is bridged only when it is short
    enough (``max_pts_to_interp`` samples) and the samples bracketing it are
    close enough (``max_cm_to_interp`` cm) that a straight line is a reasonable
    estimate. Edge gaps (no bracketing sample) and gaps failing either test are
    left NaN, to be dropped downstream.
    """
    t = position_df.index.to_numpy(dtype=float)
    x = position_df["position_x"].to_numpy()
    y = position_df["position_y"].to_numpy()
    is_nan = np.isnan(x)
    if not is_nan.any():
        return position_df

    # Consecutive-NaN spans as (start, stop) inclusive index pairs.
    nan_idx = np.where(is_nan)[0]
    spans: list[tuple[int, int]] = []
    start = prev = nan_idx[0]
    for i in nan_idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            spans.append((start, prev))
            start = prev = i
    spans.append((start, prev))

    n = len(position_df)
    arrays = {c: position_df[c].to_numpy().copy() for c in columns}
    n_interp = n_skip = 0
    for s1, s2 in spans:
        if s1 == 0 or s2 == n - 1:  # edge gap: no bracket
            n_skip += 1
            continue
        if (s2 - s1 + 1) > max_pts_to_interp:
            n_skip += 1
            continue
        jump = np.hypot(x[s1 - 1] - x[s2 + 1], y[s1 - 1] - y[s2 + 1])
        if jump > max_cm_to_interp:  # animal moved too far during the loss
            n_skip += 1
            continue
        span_t = t[s1 : s2 + 1]
        for c in columns:
            arr = arrays[c]
            arr[s1 : s2 + 1] = np.interp(
                span_t, [t[s1 - 1], t[s2 + 1]], [arr[s1 - 1], arr[s2 + 1]]
            )
        n_interp += 1

    for c in columns:
        position_df[c] = arrays[c]

    logger.info(
        "Interpolated %d dropout gaps, left %d un-bridged (too long/far)",
        n_interp,
        n_skip,
    )
    return position_df
