#!/usr/bin/env bash
exec sbatch "$(dirname -- "${BASH_SOURCE[0]}")/run_xgboost_fusion.slurm" "$@"
