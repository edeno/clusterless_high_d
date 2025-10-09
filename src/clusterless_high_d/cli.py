"""Command-line interface for clusterless high-dimensional decoding."""

import argparse
from pathlib import Path
import sys
from typing import Optional

from .decode import run_decode
from .registry import load_dataset_registry


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
        print(f"Batch complete: {len(registry) - len(failures)}/{len(registry)} succeeded")

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

    # Decode command
    decode_parser = subparsers.add_parser(
        "decode", help="Decode a single dataset"
    )
    decode_parser.add_argument(
        "--dataset-id", required=True, help="Dataset identifier"
    )
    decode_parser.add_argument(
        "--config", default="configs/base.yaml", help="Path to base config YAML"
    )
    decode_parser.add_argument(
        "--plots", default="configs/plots.yaml", help="Path to plots config YAML"
    )
    decode_parser.add_argument(
        "--outdir", default="results", help="Output directory"
    )

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
    batch_parser.add_argument(
        "--outdir", default="results", help="Output directory"
    )

    args = parser.parse_args(argv)

    if args.command == "decode":
        return decode_command(args)
    elif args.command == "batch":
        return batch_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
