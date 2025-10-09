#!/bin/bash
# Shell helper script for running decoding
#
# Usage:
#   bash scripts/run_decode.sh <dataset_id> [config_path] [plots_path]
#
# Examples:
#   bash scripts/run_decode.sh rat01_day1
#   bash scripts/run_decode.sh rat01_day1 configs/base.yaml configs/plots.yaml

set -e  # Exit on error

# Check if dataset_id is provided
if [ -z "$1" ]; then
    echo "Error: dataset_id required"
    echo "Usage: $0 <dataset_id> [config_path] [plots_path]"
    echo ""
    echo "Available datasets (from configs/datasets.yaml):"
    if [ -f configs/datasets.yaml ]; then
        grep "^[a-zA-Z]" configs/datasets.yaml | grep ":" | sed 's/://' | sed 's/^/  - /'
    else
        echo "  (configs/datasets.yaml not found)"
    fi
    exit 1
fi

# Set defaults
DATASET_ID="$1"
CONFIG_PATH="${2:-configs/base.yaml}"
PLOTS_PATH="${3:-configs/plots.yaml}"
OUTDIR="${4:-results}"

# Print configuration
echo "=================================================="
echo "Running clusterless decoding"
echo "=================================================="
echo "Dataset ID:    $DATASET_ID"
echo "Config:        $CONFIG_PATH"
echo "Plots config:  $PLOTS_PATH"
echo "Output dir:    $OUTDIR"
echo "=================================================="
echo ""

# Run the decode command
uv run python -m clusterless_high_d decode \
    --dataset-id "$DATASET_ID" \
    --config "$CONFIG_PATH" \
    --plots "$PLOTS_PATH" \
    --outdir "$OUTDIR"

echo ""
echo "=================================================="
echo "Decoding complete!"
echo "Results saved to: $OUTDIR/$DATASET_ID/"
echo "=================================================="
