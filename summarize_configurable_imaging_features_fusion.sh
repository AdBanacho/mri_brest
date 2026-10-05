#!/usr/bin/env bash
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_experiment.slurm" --summarize "$@"
