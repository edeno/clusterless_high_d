"""Spyglass I/O functions for loading hippocampal datasets."""

from pathlib import Path
from typing import Dict, Any, Tuple
import yaml

import numpy as np
import pandas as pd

# --- Spyglass Imports ---
from spyglass.common import Session, IntervalList
from spyglass.position import PositionOutput
from spyglass.decoding.v1.clusterless import (
    PositionGroup,
    UnitWaveformFeatures,
    UnitWaveformFeaturesGroup,
)


def load_dataset(dataset_id: str, registry_path: Path = Path("configs/datasets.yaml")) -> Dict[str, Any]:
    """Load a dataset from Spyglass.

    This function queries Spyglass based on the dataset_id and returns
    the features, position, and metadata needed for decoding.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (looked up in registry)
    registry_path : Path
        Path to datasets.yaml registry

    Returns
    -------
    dict
        Dictionary with keys:
        - features: np.ndarray (n_time, n_features) - neural features
        - position: np.ndarray (n_time, 2) - actual position [x, y]
        - time: np.ndarray (n_time,) - time stamps
        - metadata: dict - additional info (n_contacts, region, etc.)
        - position_grid_x: np.ndarray (n_x_bins,) - x coordinates of position grid
        - position_grid_y: np.ndarray (n_y_bins,) - y coordinates of position grid
    """
    # Load registry
    if registry_path.exists():
        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        if dataset_id not in registry:
            raise ValueError(
                f"Dataset {dataset_id} not found in registry {registry_path}. "
                f"Available datasets: {list(registry.keys())}"
            )

        dataset_info = registry[dataset_id]
        spyglass_query = dataset_info["spyglass_query"]
        metadata = {
            "region": dataset_info.get("region", "unknown"),
            "n_contacts": dataset_info.get("n_contacts", 0),
            "description": dataset_info.get("description", ""),
        }
    else:
        # No registry - treat dataset_id as direct identifier
        spyglass_query = {"identifier": dataset_id}
        metadata = {"region": "unknown", "n_contacts": 0, "description": ""}

    # Fetch data using Spyglass
    print(f"  Loading from Spyglass: {spyglass_query}")

    # Fetch waveform features (marks) if available
    waveform_group_name = spyglass_query.get("waveform_features_group_name")
    if waveform_group_name:
        waveform_key = {
            "nwb_file_name": spyglass_query["nwb_file_name"],
            "waveform_features_group_name": waveform_group_name,
        }
        spike_times, features = fetch_waveform_features(waveform_key)
    else:
        # Use placeholder if marks haven't been extracted yet
        print("  [PLACEHOLDER] Waveform features not specified, using synthetic data")
        data = _generate_placeholder_data(metadata)
        return data

    # Fetch position data
    position_group_name = spyglass_query.get("position_group_name")
    if position_group_name:
        position_key = {
            "nwb_file_name": spyglass_query["nwb_file_name"],
            "position_group_name": position_group_name,
        }
        position, time = fetch_position_group(position_key)
    else:
        print("  [PLACEHOLDER] Position group not specified, using synthetic data")
        data = _generate_placeholder_data(metadata)
        return data

    # Fetch interval for time windowing
    interval_name = spyglass_query.get("interval_list_name")
    if interval_name:
        interval_times = fetch_interval_times(
            spyglass_query["nwb_file_name"], interval_name
        )
    else:
        interval_times = None

    # Create position grid for posterior
    position_grid_x = np.linspace(position[:, 0].min(), position[:, 0].max(), 50)
    position_grid_y = np.linspace(position[:, 1].min(), position[:, 1].max(), 50)

    return {
        "spike_times": spike_times,
        "features": features,
        "position": position,
        "time": time,
        "interval_times": interval_times,
        "metadata": metadata,
        "position_grid_x": position_grid_x,
        "position_grid_y": position_grid_y,
    }


def _generate_placeholder_data(metadata: dict) -> Dict[str, Any]:
    """Generate placeholder data for testing.

    Replace this with actual Spyglass data loading.

    Parameters
    ----------
    metadata : dict
        Metadata with n_contacts, region, etc.

    Returns
    -------
    dict
        Placeholder dataset
    """
    # Create synthetic data for demonstration
    n_time = 1000
    n_contacts = metadata.get("n_contacts", 16)
    n_features_per_contact = 10  # e.g., waveform features

    # Synthetic features
    features = np.random.randn(n_time, n_contacts * n_features_per_contact)

    # Synthetic position (circular trajectory)
    t = np.linspace(0, 4 * np.pi, n_time)
    center_x, center_y = 50, 50
    radius = 30
    position = np.column_stack([
        center_x + radius * np.cos(t),
        center_y + radius * np.sin(t),
    ])

    # Time stamps
    time = np.linspace(0, 100, n_time)  # 100 seconds

    # Position grid for posterior
    position_grid_x = np.linspace(0, 100, 50)
    position_grid_y = np.linspace(0, 100, 50)

    return {
        "features": features,
        "position": position,
        "time": time,
        "metadata": metadata,
        "position_grid_x": position_grid_x,
        "position_grid_y": position_grid_y,
    }


# === Fetch Functions for Preprocessed Data ===


def fetch_waveform_features(waveform_key: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch preprocessed waveform features (marks) from Spyglass.

    This function retrieves spike times and waveform features that were
    previously extracted using the preprocessing pipeline. Use this after
    running preprocessing.populate_waveform_features().

    Parameters
    ----------
    waveform_key : dict
        Restriction key containing at minimum:
        - nwb_file_name: str
        - waveform_features_group_name: str

    Returns
    -------
    spike_times : np.ndarray
        Array of spike times (n_spikes,)
    waveform_features : np.ndarray
        Array of waveform features (n_spikes, n_features)
        Features typically include PCA components, amplitudes, etc.

    Examples
    --------
    ```python
    waveform_key = {
        'nwb_file_name': 'mediumnwb20230802_.nwb',
        'waveform_features_group_name': 'high_d_ca1_group',
    }
    spike_times, features = fetch_waveform_features(waveform_key)
    ```

    Notes
    -----
    This follows the Spyglass pattern:
        (UnitWaveformFeatures & key).fetch_data()

    See SPYGLASS.md for more details on UnitWaveformFeatures table.
    """
    # Query UnitWaveformFeatures table
    if not (UnitWaveformFeaturesGroup & waveform_key):
        raise ValueError(
            f"No waveform features group found for key: {waveform_key}\n"
            "Run preprocessing.populate_waveform_features() first."
        )

    # Fetch spike times and features using Spyglass's fetch_data method
    spike_times, waveform_features = (
        UnitWaveformFeatures & waveform_key
    ).fetch_data()

    return spike_times, waveform_features


def fetch_position_group(position_key: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch position data from Spyglass PositionGroup.

    Parameters
    ----------
    position_key : dict
        Restriction key containing:
        - nwb_file_name: str
        - position_group_name: str

    Returns
    -------
    position : np.ndarray
        Position array (n_time, 2) with [x, y] coordinates
    time : np.ndarray
        Time stamps (n_time,)

    Examples
    --------
    ```python
    position_key = {
        'nwb_file_name': 'mediumnwb20230802_.nwb',
        'position_group_name': 'default_position',
    }
    position, time = fetch_position_group(position_key)
    ```

    Notes
    -----
    This fetches from Spyglass PositionGroup table, which groups position
    data for decoding. The position data is typically derived from
    PositionOutput merge table.

    See SPYGLASS.md for details on PositionGroup and PositionOutput tables.
    """
    if not (PositionGroup & position_key):
        raise ValueError(
            f"No position group found for key: {position_key}\n"
            "Ensure PositionGroup entry exists in Spyglass."
        )

    # Fetch position data using Spyglass method
    position_df = (PositionGroup & position_key).fetch1_dataframe()

    # Extract position and time
    position = position_df[["position_x", "position_y"]].values
    time = position_df.index.values  # Assuming time is the index

    return position, time


def fetch_interval_times(nwb_file_name: str, interval_list_name: str) -> np.ndarray:
    """Fetch time intervals from Spyglass IntervalList.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name
    interval_list_name : str
        Name of the interval list

    Returns
    -------
    np.ndarray
        Array of interval times, shape (n_intervals, 2) with [start, stop]

    Examples
    --------
    ```python
    intervals = fetch_interval_times(
        'mediumnwb20230802_.nwb',
        'pos 0 valid times'
    )
    ```

    Notes
    -----
    IntervalList defines time windows for analysis (e.g., task periods,
    sleep epochs, valid tracking times). All Spyglass analyses are
    windowed by IntervalList entries.

    See SPYGLASS.md for details on IntervalList table.
    """
    interval_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": interval_list_name,
    }

    if not (IntervalList & interval_key):
        raise ValueError(
            f"IntervalList not found: {nwb_file_name}, {interval_list_name}"
        )

    # Fetch valid times
    valid_times = (IntervalList & interval_key).fetch1("valid_times")

    return valid_times


def validate_spyglass_connection() -> bool:
    """Validate that Spyglass database connection is working.

    Returns
    -------
    bool
        True if connection is valid, False otherwise

    Examples
    --------
    ```python
    if validate_spyglass_connection():
        print("Spyglass connection OK")
    else:
        print("Spyglass connection failed")
    ```

    Notes
    -----
    This performs a simple test query to verify DataJoint connection
    and database access. If this fails, check:
    - Database credentials in dj.config
    - Network connectivity
    - Database server status
    """
    try:
        # Try a simple query to test connection
        Session.fetch(limit=1)
        return True
    except Exception as e:
        print(f"Spyglass connection failed: {e}")
        return False
