"""Core decoding logic for clusterless high-dimensional analysis."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .plotting import plot_posterior, plot_summary
from .spyglass_io import load_dataset


def run_decode(
    dataset_id: str,
    config_path: Path,
    plots_config_path: Optional[Path] = None,
    outdir: Path = Path("results"),
) -> dict:
    """Run clusterless decoding for a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    config_path : Path
        Path to base configuration YAML
    plots_config_path : Path, optional
        Path to plotting configuration YAML
    outdir : Path
        Output directory (results will be in outdir/dataset_id/timestamp/)

    Returns
    -------
    dict
        Dictionary with paths to outputs and summary statistics
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    plots_config = None
    if plots_config_path and plots_config_path.exists():
        with open(plots_config_path) as f:
            plots_config = yaml.safe_load(f)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = outdir / dataset_id / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Save resolved config for provenance
    config_resolved = {
        "dataset_id": dataset_id,
        "timestamp": timestamp,
        "config": config,
        "plots_config": plots_config,
    }
    with open(output_dir / "config.resolved.yaml", "w") as f:
        yaml.dump(config_resolved, f, default_flow_style=False)

    # Load dataset from Spyglass
    print(f"Loading dataset {dataset_id}...")
    data = load_dataset(dataset_id)

    # Extract data components
    spike_times = data.get("spike_times")
    waveform_features = data.get("features")
    position = data["position"]
    time = data["time"]
    metadata = data["metadata"]

    if spike_times is not None:
        print(
            f"  Loaded {len(spike_times)} spikes with {waveform_features.shape[1]} features"
        )
    else:
        print(f"  Using placeholder data")
    print(f"  Position data: {len(time)} time bins")
    print(
        f"  Metadata: {metadata['n_contacts']} contacts, region: {metadata.get('region', 'unknown')}"
    )

    # Check if high-dimensional
    n_contacts_threshold = config["high_dimensional"]["n_contacts_threshold"]
    is_high_d = metadata["n_contacts"] >= n_contacts_threshold
    print(f"  High-dimensional: {is_high_d} (threshold: {n_contacts_threshold})")

    # Run decoding
    # TODO: Replace placeholder with actual non_local_detector call
    print("Running decoder...")
    decoder_type = config.get("decoder", {}).get("type", "placeholder")

    if spike_times is not None and waveform_features is not None:
        # Have real marks - ready for actual decoder
        print(f"  Decoder type: {decoder_type}")
        posterior = _run_decoder_placeholder(
            spike_times=spike_times,
            waveform_features=waveform_features,
            position=position,
            time=time,
            config=config,
            is_high_d=is_high_d,
        )
    else:
        # Fallback to synthetic data for testing
        print("  Using placeholder decoder with synthetic data")
        posterior = _run_decoder_placeholder(
            spike_times=None,
            waveform_features=None,
            position=position,
            time=time,
            config=config,
            is_high_d=is_high_d,
        )

    # Save outputs
    print("Saving outputs...")
    np.savez_compressed(
        output_dir / "posterior.npz",
        time=time,
        posterior=posterior,
        position_grid_x=data.get("position_grid_x"),
        position_grid_y=data.get("position_grid_y"),
        actual_position=position,
    )

    # Generate plots
    if plots_config:
        print("Generating plots...")
        plot_posterior(
            time=time,
            posterior=posterior,
            actual_position=position,
            config=plots_config,
            output_path=plots_dir
            / f"posterior_over_time.{plots_config['general']['format']}",
        )

        plot_summary(
            time=time,
            posterior=posterior,
            actual_position=position,
            config=plots_config,
            output_path=plots_dir / f"summary.{plots_config['general']['format']}",
        )

    print(f"✓ Results saved to {output_dir}")

    return {
        "output_dir": output_dir,
        "dataset_id": dataset_id,
        "timestamp": timestamp,
        "n_time_bins": len(time),
        "n_contacts": metadata["n_contacts"],
        "is_high_d": is_high_d,
    }


def _run_decoder_placeholder(
    spike_times: np.ndarray,
    waveform_features: np.ndarray,
    position: np.ndarray,
    time: np.ndarray,
    config: dict,
    is_high_d: bool,
) -> np.ndarray:
    """Placeholder decoder - replace with actual non_local_detector call.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times (n_spikes,) or None for synthetic data
    waveform_features : np.ndarray
        Waveform features (n_spikes, n_features) or None for synthetic data
    position : np.ndarray
        Actual positions (n_time, 2)
    time : np.ndarray
        Time stamps (n_time,)
    config : dict
        Configuration dictionary
    is_high_d : bool
        Whether this is high-dimensional data

    Returns
    -------
    np.ndarray
        Posterior probability (n_time, n_x_bins, n_y_bins)

    Notes
    -----
    TODO: Replace with actual decoder implementation:

    For non_local_detector integration:
    ```python
    from non_local_detector import ClusterlessDecoder

    # Initialize decoder
    decoder = ClusterlessDecoder(
        position_range=config['decoding']['position_range'],
        spatial_resolution=config['decoding']['spatial_resolution'],
        **config['decoder']
    )

    # Fit on encoding data
    decoder.fit(spike_times, waveform_features, position)

    # Decode
    posterior = decoder.predict(spike_times, waveform_features, time)
    ```

    For GPU-accelerated or custom decoders:
    - Branch on config['decoder']['type']
    - Implement different decoder classes
    - Compare performance/accuracy across implementations
    """
    # For now, return a placeholder posterior
    n_time = len(time)
    n_x_bins = 50
    n_y_bins = 50

    if spike_times is not None:
        print(f"  [PLACEHOLDER] Would decode {len(spike_times)} spikes")
        print(f"  [PLACEHOLDER] Feature dimensions: {waveform_features.shape}")
    print("  [PLACEHOLDER] Using dummy decoder - replace with non_local_detector")

    # Create random posterior for demonstration
    posterior = np.random.rand(n_time, n_x_bins, n_y_bins)
    # Normalize across space for each time bin
    posterior = posterior / posterior.sum(axis=(1, 2), keepdims=True)

    return posterior
