#!/usr/bin/env bash
# Bash entry point for training, validation and summarization.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
configuration="$ROOT/run_configuration.sh"
runner_args=()
while (($#)); do
  case "$1" in
    --config)
      if (($# < 2)); then echo "Error: --config needs a Bash file" >&2; exit 2; fi
      configuration="$2"
      shift 2
      ;;
    *)
      runner_args+=("$1")
      shift
      ;;
  esac
done
configuration="$(realpath -- "$configuration")"
if [[ ! -f "$configuration" ]]; then
  echo "Error: Bash configuration not found: $configuration" >&2
  exit 2
fi
# Configuration is trusted Bash code. It defines CFG_* associative arrays.
source "$configuration"

write_section() {
  local section="$1" variable="CFG_${1//./_}" key sorted_keys sorted_keys
  local -n entries="$variable"
  printf '\n[%s]\n' "$section"
  # Sorting is for readable, reproducible generated configs.
  sorted_keys="$(printf '%s\\n' "${!entries[@]}" | LC_ALL=C sort)"
  sorted_keys="$(printf '%s\n' "${!entries[@]}" | LC_ALL=C sort)"
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    printf '%s = %s\n' "$key" "${entries[$key]}"
  done <<< "$sorted_keys"
}
write_config() {
  local section
  for section in data paths cleanup jobs jobs.train jobs.validate jobs.summarize \
      train train.xgb_extra train.mlp_extra train.mri_extra train.trainer_extra \
      train.grid validate validate.grid summarize; do
    write_section "$section"
  done
}

supports_runner() {
  "$1" -c 'import sys; assert sys.version_info >= (3, 10)' >/dev/null 2>&1
}
python_bin="${PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1 || ! supports_runner "$python_bin"; then
  if command -v python3.11 >/dev/null 2>&1 && supports_runner python3.11; then
    python_bin=python3.11
  elif type module >/dev/null 2>&1 && module load ML-bundle >&2 && supports_runner python3; then
    python_bin=python3
  else
    echo "Error: Python 3.10+ is required. Load an available Python module (check: module avail Python), or set PYTHON_BIN." >&2
    exit 2
  fi
fi

# Python 3.10 needs tomli; Python 3.11+ provides tomllib.
if ! "$python_bin" -c 'import importlib.util; assert importlib.util.find_spec("tomllib") or importlib.util.find_spec("tomli")' >/dev/null 2>&1; then
  "$python_bin" -m pip install tomli
fi

temporary_dir="${TMPDIR:-/tmp}"
[[ -d "$temporary_dir" && -w "$temporary_dir" ]] || temporary_dir="$ROOT"
temporary_config="$(mktemp "$temporary_dir/mri-experiment.XXXXXXXX.toml")"
trap 'rm -f -- "$temporary_config"' EXIT
write_config > "$temporary_config"
cd "$ROOT"
"$python_bin" "$ROOT/run_experiment.py" --config "$temporary_config" \
  --config-base "$(dirname -- "$configuration")" "${runner_args[@]}"
