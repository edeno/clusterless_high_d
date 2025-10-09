"""Dataset registry management for batch processing."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_dataset_registry(
    registry_path: Path = Path("configs/datasets.yaml"),
) -> Dict[str, Any]:
    """Load the dataset registry from YAML.

    Parameters
    ----------
    registry_path : Path
        Path to datasets.yaml file

    Returns
    -------
    dict
        Dictionary mapping dataset_id to metadata/query info

    Raises
    ------
    FileNotFoundError
        If registry file doesn't exist
    ValueError
        If registry is malformed
    """
    registry_path = Path(registry_path)

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Dataset registry not found: {registry_path}\n"
            "Create configs/datasets.yaml or provide a valid path."
        )

    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    if not isinstance(registry, dict):
        raise ValueError(f"Registry must be a dictionary, got {type(registry)}")

    # Validate each entry
    for dataset_id, info in registry.items():
        if not isinstance(info, dict):
            raise ValueError(
                f"Dataset entry '{dataset_id}' must be a dictionary, got {type(info)}"
            )

        if "spyglass_query" not in info:
            raise ValueError(
                f"Dataset entry '{dataset_id}' missing required 'spyglass_query' field"
            )

    return registry


def get_dataset_info(
    dataset_id: str, registry_path: Path = Path("configs/datasets.yaml")
) -> Dict[str, Any]:
    """Get metadata for a specific dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier
    registry_path : Path
        Path to datasets.yaml file

    Returns
    -------
    dict
        Dataset metadata and query info

    Raises
    ------
    KeyError
        If dataset_id not found in registry
    """
    registry = load_dataset_registry(registry_path)

    if dataset_id not in registry:
        available = list(registry.keys())
        raise KeyError(
            f"Dataset '{dataset_id}' not found in registry.\n"
            f"Available datasets: {available}"
        )

    return registry[dataset_id]


def list_datasets(registry_path: Path = Path("configs/datasets.yaml")) -> list:
    """List all dataset IDs in the registry.

    Parameters
    ----------
    registry_path : Path
        Path to datasets.yaml file

    Returns
    -------
    list
        List of dataset IDs
    """
    registry = load_dataset_registry(registry_path)
    return list(registry.keys())


def filter_datasets(
    registry_path: Path = Path("configs/datasets.yaml"),
    region: str = None,
    min_contacts: int = None,
    max_contacts: int = None,
) -> Dict[str, Any]:
    """Filter datasets by criteria.

    Parameters
    ----------
    registry_path : Path
        Path to datasets.yaml file
    region : str, optional
        Filter by brain region (e.g., 'CA1', 'CA3')
    min_contacts : int, optional
        Minimum number of contacts
    max_contacts : int, optional
        Maximum number of contacts

    Returns
    -------
    dict
        Filtered registry
    """
    registry = load_dataset_registry(registry_path)
    filtered = {}

    for dataset_id, info in registry.items():
        # Check region filter
        if region is not None:
            if info.get("region") != region:
                continue

        # Check contact count filters
        n_contacts = info.get("n_contacts", 0)
        if min_contacts is not None and n_contacts < min_contacts:
            continue
        if max_contacts is not None and n_contacts > max_contacts:
            continue

        filtered[dataset_id] = info

    return filtered
