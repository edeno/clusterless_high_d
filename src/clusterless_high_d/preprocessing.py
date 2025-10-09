"""Preprocessing pipeline for extracting clusterless marks from raw ephys data.

This module handles the extraction and grouping of waveform features (marks)
from raw electrophysiology data using the Spyglass pipeline. It populates:
- UnitWaveformFeatures: Individual spike events with waveform features
- UnitWaveformFeaturesGroup: Grouped marks ready for decoder input

Once preprocessing is complete, use spyglass_io.fetch_waveform_features()
to retrieve the marks for decoding.
"""

from typing import List, Union

import datajoint as dj

# --- Spyglass Imports ---
from spyglass.common import IntervalList, Nwbfile
from spyglass.decoding.v1.clusterless import (
    PositionGroup,
    UnitWaveformFeatures,
    UnitWaveformFeaturesGroup,
)
from spyglass.utils import logger


def populate_waveform_features(
    nwb_file_name: str,
    waveform_features_group_name: str,
    position_group_name: str,
    encoding_interval: str,
    decoding_interval: Union[str, List[str]],
    skip_duplicates: bool = True,
    **kwargs,
) -> None:
    """Extract and group waveform features (marks) for clusterless decoding.

    This function populates the Spyglass UnitWaveformFeatures and
    UnitWaveformFeaturesGroup tables, which extract spike waveforms and
    compute features (e.g., PCA, amplitude) from raw electrophysiology data.

    The grouped features can later be fetched for decoding using:
        spike_times, waveform_features = (
            UnitWaveformFeatures & waveform_keys
        ).fetch_data()

    Parameters
    ----------
    nwb_file_name : str
        The name of the source NWB file.
    waveform_features_group_name : str
        Name for the waveform features group to create in
        UnitWaveformFeaturesGroup. This identifies the set of marks
        for downstream decoding.
    position_group_name : str
        The name of the position group in PositionGroup. Used to
        align marks with behavioral data.
    encoding_interval : str
        The name of the interval in IntervalList used for encoding/training.
        Marks will be extracted from this time window.
    decoding_interval : Union[str, List[str]]
        The name of the interval(s) in IntervalList for decoding/prediction.
        Can be a single interval name or list of interval names.
    skip_duplicates : bool, optional
        If True, skips insertion if matching entries already exist.
        Defaults to True.
    **kwargs : dict
        Additional keyword arguments passed to the populate call
        (e.g., display_progress=True, reserve_jobs=True).

    Raises
    ------
    ValueError
        If required upstream entries do not exist in Spyglass tables.
    datajoint.errors.DataJointError
        If insertion or population fails.

    Examples
    --------
    ```python
    # Prerequisites: Ensure these exist in Spyglass
    # - NWB file imported (Nwbfile table)
    # - Position data processed (PositionGroup table)
    # - Intervals defined (IntervalList table)

    from clusterless_high_d.preprocessing import populate_waveform_features

    # Extract marks for a single interval
    populate_waveform_features(
        nwb_file_name='mediumnwb20230802_.nwb',
        waveform_features_group_name='high_d_ca1_group',
        position_group_name='default_position',
        encoding_interval='pos 0 valid times',
        decoding_interval='pos 0 valid times',
        display_progress=True
    )

    # Extract marks for multiple intervals
    populate_waveform_features(
        nwb_file_name='mediumnwb20230802_.nwb',
        waveform_features_group_name='high_d_ca1_group',
        position_group_name='default_position',
        encoding_interval='pos 0 valid times',
        decoding_interval=['interval_1', 'interval_2', 'interval_3'],
        display_progress=True
    )
    ```

    Notes
    -----
    This function wraps Spyglass's mark extraction pipeline. The actual
    mark extraction parameters (filter settings, threshold, PCA components)
    are configured in Spyglass's parameter tables.

    After preprocessing completes, fetch marks for decoding with:
        from clusterless_high_d.spyglass_io import fetch_waveform_features
        spike_times, features = fetch_waveform_features(waveform_keys)
    """
    # --- Input Validation ---
    logger.info(f"Starting waveform feature extraction for {nwb_file_name}")

    # Validate NWB file exists
    if not (Nwbfile & {"nwb_file_name": nwb_file_name}):
        raise ValueError(
            f"NWB file not found in Spyglass: {nwb_file_name}\n"
            "Ensure the file has been imported using Spyglass ingestion pipeline."
        )

    # Validate position group exists
    pos_group_key = {
        "nwb_file_name": nwb_file_name,
        "position_group_name": position_group_name,
    }
    if not (PositionGroup & pos_group_key):
        raise ValueError(
            f"PositionGroup entry not found for: {nwb_file_name}, {position_group_name}\n"
            "Create a PositionGroup entry before extracting marks."
        )

    # Validate encoding interval exists
    encoding_interval_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": encoding_interval,
    }
    if not (IntervalList & encoding_interval_key):
        raise ValueError(
            f"Encoding IntervalList not found: {nwb_file_name}, {encoding_interval}\n"
            "Define the interval in IntervalList table first."
        )

    # Validate and collect decoding intervals
    if isinstance(decoding_interval, str):
        decoding_intervals = [decoding_interval]
    elif isinstance(decoding_interval, list):
        decoding_intervals = decoding_interval
    else:
        raise TypeError(
            "decoding_interval must be a string or list of strings."
        )

    valid_decoding_intervals = []
    for interval_name in decoding_intervals:
        interval_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
        if not (IntervalList & interval_key):
            raise ValueError(
                f"Decoding IntervalList not found: {nwb_file_name}, {interval_name}\n"
                "Define all intervals before preprocessing."
            )
        valid_decoding_intervals.append(interval_name)

    if not valid_decoding_intervals:
        logger.error("No valid decoding intervals found. Aborting.")
        return

    logger.info(
        f"Validated inputs: {len(valid_decoding_intervals)} decoding interval(s)"
    )

    # --- Base Key for Group ---
    group_key = {
        "nwb_file_name": nwb_file_name,
        "waveform_features_group_name": waveform_features_group_name,
        "position_group_name": position_group_name,
        "encoding_interval": encoding_interval,
    }

    # --- Process Each Interval ---
    successful_intervals = 0
    failed_intervals = 0

    for current_interval in valid_decoding_intervals:
        pipeline_description = (
            f"{nwb_file_name} | WF Group: {waveform_features_group_name} | "
            f"Interval: {current_interval}"
        )

        features_key = {
            **group_key,
            "decoding_interval": current_interval,
        }

        try:
            # --- 1. Populate UnitWaveformFeatures ---
            # This extracts spike waveforms and computes features
            logger.info(
                f"---- Extracting waveform features | {pipeline_description} ----"
            )

            if not (UnitWaveformFeatures & features_key):
                UnitWaveformFeatures.populate(
                    features_key,
                    reserve_jobs=True,
                    **kwargs
                )
                logger.info(f"Waveform features extracted for {current_interval}")
            else:
                logger.info(
                    f"Waveform features already exist for {current_interval}"
                )

            # Verify extraction succeeded
            if not (UnitWaveformFeatures & features_key):
                raise dj.errors.DataJointError(
                    f"UnitWaveformFeatures population failed for {pipeline_description}"
                )

            # --- 2. Populate UnitWaveformFeaturesGroup ---
            # This groups features for easy fetching during decoding
            logger.info(
                f"---- Grouping waveform features | {pipeline_description} ----"
            )

            if not (UnitWaveformFeaturesGroup & features_key):
                UnitWaveformFeaturesGroup.populate(
                    features_key,
                    reserve_jobs=True,
                    **kwargs
                )
                logger.info(f"Waveform features grouped for {current_interval}")
            else:
                logger.info(
                    f"Waveform features group already exists for {current_interval}"
                )

            # Verify grouping succeeded
            if not (UnitWaveformFeaturesGroup & features_key):
                raise dj.errors.DataJointError(
                    f"UnitWaveformFeaturesGroup population failed for {pipeline_description}"
                )

            successful_intervals += 1
            logger.info(f"✓ Completed preprocessing for {current_interval}")

        except dj.errors.DataJointError as e:
            logger.error(
                f"DataJoint error processing interval {current_interval}: {e}"
            )
            failed_intervals += 1
        except Exception as e:
            logger.error(
                f"Error processing interval {current_interval}: {e}",
                exc_info=True,
            )
            failed_intervals += 1

    # --- Final Summary ---
    logger.info(f"---- Preprocessing complete for {nwb_file_name} ----")
    logger.info(f"    Successfully processed: {successful_intervals} intervals")
    logger.info(f"    Failed: {failed_intervals} intervals")

    if successful_intervals > 0:
        logger.info(
            f"\nTo fetch marks for decoding, use:"
            f"\n    from clusterless_high_d.spyglass_io import fetch_waveform_features"
            f"\n    waveform_keys = {{"
            f"\n        'nwb_file_name': '{nwb_file_name}',"
            f"\n        'waveform_features_group_name': '{waveform_features_group_name}'"
            f"\n    }}"
            f"\n    spike_times, features = fetch_waveform_features(waveform_keys)"
        )


def validate_preprocessing_inputs(
    nwb_file_name: str,
    position_group_name: str,
    interval_list_names: List[str],
) -> bool:
    """Validate that required upstream Spyglass entries exist.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name to check
    position_group_name : str
        Position group name to check
    interval_list_names : List[str]
        List of interval names to check

    Returns
    -------
    bool
        True if all inputs are valid, raises ValueError otherwise
    """
    # Check NWB file
    if not (Nwbfile & {"nwb_file_name": nwb_file_name}):
        raise ValueError(f"NWB file not found: {nwb_file_name}")

    # Check position group
    pos_key = {
        "nwb_file_name": nwb_file_name,
        "position_group_name": position_group_name,
    }
    if not (PositionGroup & pos_key):
        raise ValueError(
            f"PositionGroup not found: {nwb_file_name}, {position_group_name}"
        )

    # Check all intervals
    for interval_name in interval_list_names:
        interval_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
        if not (IntervalList & interval_key):
            raise ValueError(
                f"IntervalList not found: {nwb_file_name}, {interval_name}"
            )

    return True
