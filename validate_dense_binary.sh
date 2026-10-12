#!/usr/bin/env bash
exec sbatch "$(dirname -- "${BASH_SOURCE[0]}")/validate_dense_binary.slurm" "$@"
