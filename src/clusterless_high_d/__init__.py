"""Clusterless high-dimensional decoding for hippocampal datasets.

Workflow:
---------
1. Preprocessing: Extract marks from raw ephys
   >>> from clusterless_high_d.preprocessing import populate_waveform_features
   >>> populate_waveform_features(nwb_file_name, waveform_group_name, ...)

2. Decoding: Fetch marks and run decoder
   >>> from clusterless_high_d import run_decode
   >>> run_decode(dataset_id, config_path, ...)

3. Analysis: Load results and compare decoders
   >>> results = np.load('results/.../posterior.npz')
"""

__version__ = "0.1.0"

from .decode import run_decode
from .preprocessing import populate_waveform_features, validate_preprocessing_inputs
from .registry import load_dataset_registry
from .spyglass_io import (
    fetch_interval_times,
    fetch_position_group,
    fetch_waveform_features,
    load_dataset,
    validate_spyglass_connection,
)

__all__ = [
    # Core workflow
    "run_decode",
    "populate_waveform_features",
    # Data loading
    "load_dataset",
    "fetch_waveform_features",
    "fetch_position_group",
    "fetch_interval_times",
    # Registry management
    "load_dataset_registry",
    # Validation
    "validate_preprocessing_inputs",
    "validate_spyglass_connection",
]
