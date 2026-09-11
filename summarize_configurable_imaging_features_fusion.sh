#!/bin/bash -l

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR:-validation_charts}

python -m mriBreastDuke.summarize_configurable_imaging_features_fusion \
    --input-dir "$VALIDATION_OUTPUT_DIR" \
    "$@"
