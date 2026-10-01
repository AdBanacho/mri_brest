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
  local section="$1" variable="CFG_${1//./_}" key sorted_keys
  local -n entries="$variable"
  printf '\n[%s]\n' "$section"
  # Sorting is for readable, reproducible generated configs.
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
  "$1" -c 'import sys; assert sys.version_info >= (3, 10); import importlib.util; assert importlib.util.find_spec("tomllib") or importlib.util.find_spec("tomli")' >/dev/null 2>&1
}
# Planning needs only Python and a TOML parser. Never load the GPU ML bundle
# here: its CUDA dependencies may be unavailable on a login node.
python_bin=""
if [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 || ! supports_runner "$PYTHON_BIN"; then
    echo "Error: PYTHON_BIN=$PYTHON_BIN needs Python 3.10+ with tomllib/tomli." >&2
    exit 2
  fi
  python_bin="$PYTHON_BIN"
else
  for candidate in python3 python3.14 python3.13 python3.12 python3.11 python3.10 python; do
    if command -v "$candidate" >/dev/null 2>&1 && supports_runner "$candidate"; then
      python_bin="$candidate"
      break
    fi
  done
  if [[ -z "$python_bin" ]] && type module >/dev/null 2>&1; then
    # Probe in a subshell so a failed module load cannot pollute later attempts.
    module_candidates=()
    if [[ -n "${RUNNER_PYTHON_MODULE:-}" ]]; then
      module_candidates+=("$RUNNER_PYTHON_MODULE")
    else
      module_candidates+=(Python)
      available_python_modules="$(
        { module -t avail Python 2>&1 || true; } |
          sed -nE 's/^[[:space:]]*(Python\/[^[:space:]()]+).*/\1/p' |
          LC_ALL=C sort -Vr -u
      )"
      while IFS= read -r candidate; do
        [[ -n "$candidate" ]] && module_candidates+=("$candidate")
      done <<< "$available_python_modules"
    fi
    for candidate in "${module_candidates[@]}"; do
      if (module load "$candidate" >/dev/null 2>&1 && supports_runner python3); then
        if module load "$candidate" >&2 && supports_runner python3; then
          python_bin=python3
          break
        fi
      fi
    done
  fi
fi
if [[ -z "$python_bin" ]]; then
  echo "Error: No usable Python 3.10+ with tomllib/tomli was found." >&2
  echo "Run 'module spider Python' to find a lightweight Python module and its prerequisites, then load it and retry." >&2
  echo "Alternatively set RUNNER_PYTHON_MODULE to an available Python module, or PYTHON_BIN to a suitable interpreter." >&2
  echo "Python 3.10 also requires tomli (install with: python3 -m pip install --user tomli)." >&2
  exit 2
fi

temporary_dir="${TMPDIR:-/tmp}"
[[ -d "$temporary_dir" && -w "$temporary_dir" ]] || temporary_dir="$ROOT"
temporary_config="$(mktemp "$temporary_dir/mri-experiment.XXXXXXXX.toml")"
trap 'rm -f -- "$temporary_config"' EXIT
write_config > "$temporary_config"
cd "$ROOT"
"$python_bin" "$ROOT/run_experiment.py" --config "$temporary_config" \
  --config-base "$(dirname -- "$configuration")" "${runner_args[@]}"
