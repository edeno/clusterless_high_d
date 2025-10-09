"""Clusterless high-dimensional decoding for hippocampal datasets."""

__version__ = "0.1.0"

from .decode import run_decode
from .registry import load_dataset_registry
from .spyglass_io import load_dataset

__all__ = ["run_decode", "load_dataset", "load_dataset_registry"]
