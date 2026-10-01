#!/usr/bin/env bash
# Regression checks without Lmod, Slurm, or ML dependencies.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "$root/launcher-test.XXXXXXXX")"
trap 'rm -rf -- "$test_dir"' EXIT
cp "$root/run_experiment.sh" "$root/run_configuration.sh" "$test_dir/"
printf '%s\n' 'print("planner reached")' > "$test_dir/run_experiment.py"
export TEST_REAL_PYTHON="$(command -v python3)"
export TEST_MODULE_LOG="$test_dir/modules.log"
export TEST_ACTIVE=0 TEST_MODE=default
python3() {
  if [[ "$TEST_ACTIVE" != 1 ]]; then return 1; fi
  "$TEST_REAL_PYTHON" "$@"
}
python() { return 1; }
python3.10() { return 1; }
python3.11() { return 1; }
python3.12() { return 1; }
python3.13() { return 1; }
python3.14() { return 1; }
module() {
  printf '%s\n' "$*" >> "$TEST_MODULE_LOG"
  if [[ "$1" == -t ]]; then
    printf '%s\n' 'Python/3.12-test (D)' >&2
    return 0
  fi
  if [[ "$1" == load && "$TEST_MODE" != failed &&
        ( "$2" == Python/3.12-test || ( "$2" == Python && "$TEST_MODE" == default ) ) ]]; then
    export TEST_ACTIVE=1
    return 0
  fi
  return 1
}
export -f python3 python python3.10 python3.11 python3.12 python3.13 python3.14 module
for TEST_MODE in default discovered; do
  export TEST_MODE
  output="$(bash "$test_dir/run_experiment.sh" --train --dry-run)"
  [[ "$output" == 'planner reached' ]]
done
export TEST_MODE=failed
if bash "$test_dir/run_experiment.sh" --train --dry-run > "$test_dir/out" 2> "$test_dir/err"; then
  echo "Expected unavailable Python to fail" >&2
  exit 1
fi
grep -q 'module spider Python' "$test_dir/err"
export TEST_MODE=discovered RUNNER_PYTHON_MODULE=Python/3.12-test
[[ "$(bash "$test_dir/run_experiment.sh" --train --dry-run)" == 'planner reached' ]]
unset RUNNER_PYTHON_MODULE
[[ "$(PYTHON_BIN="$TEST_REAL_PYTHON" bash "$test_dir/run_experiment.sh" --train --dry-run)" == 'planner reached' ]]
if PYTHON_BIN=/nonexistent/python bash "$test_dir/run_experiment.sh" --train --dry-run > "$test_dir/out" 2> "$test_dir/err"; then
  echo "Expected invalid explicit PYTHON_BIN to fail" >&2
  exit 1
fi
if grep -q 'ML-bundle' "$TEST_MODULE_LOG"; then
  echo "Launcher attempted to load the ML bundle" >&2
  exit 1
fi
echo "Launcher regression checks passed"
