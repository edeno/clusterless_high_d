"""Command-line interface for clusterless high-dimensional decoding."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .decode import run_decode
from .preprocessing import populate_waveform_features
from .registry import get_dataset_info, load_dataset_registry


def preprocess_command(args: argparse.Namespace) -> int:
    """Run preprocessing to extract waveform features."""
    try:
        # Load dataset info from registry
        dataset_info = get_dataset_info(args.dataset_id, args.datasets)
        spyglass_query = dataset_info["spyglass_query"]
        preprocessing_params = dataset_info.get("preprocessing", {})

        # Extract parameters
        nwb_file_name = spyglass_query["nwb_file_name"]
        waveform_group_name = spyglass_query.get("waveform_features_group_name")
        position_group_name = spyglass_query.get("position_group_name")
        encoding_interval = preprocessing_params.get("encoding_interval")
        decoding_interval = preprocessing_params.get("decoding_interval")

        # Validate required parameters
        if not waveform_group_name:
            raise ValueError(
                f"waveform_features_group_name not specified in dataset config"
            )
        if not position_group_name:
            raise ValueError(f"position_group_name not specified in dataset config")
        if not encoding_interval:
            raise ValueError(f"encoding_interval not specified in preprocessing config")
        if not decoding_interval:
            raise ValueError(f"decoding_interval not specified in preprocessing config")

        print(f"Preprocessing dataset: {args.dataset_id}")
        print(f"  NWB file: {nwb_file_name}")
        print(f"  Waveform group: {waveform_group_name}")
        print(f"  Position group: {position_group_name}")
        print(f"  Encoding interval: {encoding_interval}")
        print(f"  Decoding interval: {decoding_interval}")

        # Run preprocessing
        populate_waveform_features(
            nwb_file_name=nwb_file_name,
            waveform_features_group_name=waveform_group_name,
            position_group_name=position_group_name,
            encoding_interval=encoding_interval,
            decoding_interval=decoding_interval,
            skip_duplicates=not args.overwrite,
            display_progress=True,
        )

        print(f"✓ Preprocessing completed for {args.dataset_id}")
        return 0
    except Exception as e:
        print(f"✗ Error during preprocessing: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def decode_command(args: argparse.Namespace) -> int:
    """Run decoding for a single dataset."""
    try:
        run_decode(
            dataset_id=args.dataset_id,
            config_path=Path(args.config),
            plots_config_path=Path(args.plots) if args.plots else None,
            outdir=Path(args.outdir),
        )
        print(f"✓ Decoding completed for {args.dataset_id}")
        return 0
    except Exception as e:
        print(f"✗ Error during decoding: {e}", file=sys.stderr)
        return 1


def batch_command(args: argparse.Namespace) -> int:
    """Run decoding in batch mode over all datasets in registry."""
    try:
        registry = load_dataset_registry(args.datasets or "configs/datasets.yaml")

        print(f"Running batch decode for {len(registry)} datasets...")

        failures = []
        for dataset_id in registry.keys():
            print(f"\n→ Processing {dataset_id}...")
            try:
                run_decode(
                    dataset_id=dataset_id,
                    config_path=Path(args.config),
                    plots_config_path=Path(args.plots) if args.plots else None,
                    outdir=Path(args.outdir),
                )
                print(f"  ✓ Completed {dataset_id}")
            except Exception as e:
                print(f"  ✗ Failed {dataset_id}: {e}", file=sys.stderr)
                failures.append((dataset_id, str(e)))

        print(f"\n{'='*60}")
        print(
            f"Batch complete: {len(registry) - len(failures)}/{len(registry)} succeeded"
        )

        if failures:
            print(f"\nFailed datasets:")
            for dataset_id, error in failures:
                print(f"  - {dataset_id}: {error}")
            return 1

        return 0
    except Exception as e:
        print(f"✗ Batch error: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clusterless high-dimensional decoding for hippocampal datasets"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Extract waveform features (marks) from raw ephys data"
    )
    preprocess_parser.add_argument(
        "--dataset-id", required=True, help="Dataset identifier"
    )
    preprocess_parser.add_argument(
        "--datasets",
        default="configs/datasets.yaml",
        help="Path to datasets registry with preprocessing params",
    )
    preprocess_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing features (default: skip duplicates)",
    )
    preprocess_parser.add_argument(
        "--verbose", action="store_true", help="Show detailed error messages"
    )

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode a single dataset")
    decode_parser.add_argument("--dataset-id", required=True, help="Dataset identifier")
    decode_parser.add_argument(
        "--config", default="configs/base.yaml", help="Path to base config YAML"
    )
    decode_parser.add_argument(
        "--plots", default="configs/plots.yaml", help="Path to plots config YAML"
    )
    decode_parser.add_argument("--outdir", default="results", help="Output directory")

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Batch decode all datasets in registry"
    )
    batch_parser.add_argument(
        "--config", default="configs/base.yaml", help="Path to base config YAML"
    )
    batch_parser.add_argument(
        "--plots", default="configs/plots.yaml", help="Path to plots config YAML"
    )
    batch_parser.add_argument(
        "--datasets", default="configs/datasets.yaml", help="Path to datasets registry"
    )
    batch_parser.add_argument("--outdir", default="results", help="Output directory")

    args = parser.parse_args(argv)

    if args.command == "preprocess":
        return preprocess_command(args)
    elif args.command == "decode":
        return decode_command(args)
    elif args.command == "batch":
        return batch_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
