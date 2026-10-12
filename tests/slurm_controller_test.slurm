#!/usr/bin/env bash
# Integration tests run locally with stub Slurm and model executables.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TMP=$(mktemp -d "$ROOT/test-slurm.XXXXXXXX")
trap 'rm -rf -- "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/config with spaces" "$TMP/work"
export STUB_SBATCH_ARGS="$TMP/sbatch.args" STUB_MODEL_ARGS="$TMP/model.args" STUB_MODULES="$TMP/modules"
cat > "$TMP/bin/sbatch" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$STUB_SBATCH_ARGS"
printf '12345\n'
STUB
cat > "$TMP/bin/model" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$STUB_MODEL_ARGS"
printf '%s\n' "$PWD" > "$STUB_MODEL_ARGS.cwd"
STUB
chmod +x "$TMP/bin/"*
export PATH="$TMP/bin:$PATH"
module() { printf '%s\n' "$*" >> "$STUB_MODULES"; }
export -f module
CONFIG="$TMP/config with spaces/experiment.conf"
cp "$ROOT/experiment.conf" "$CONFIG"
cat >> "$CONFIG" <<OVERRIDES
MAX_JOBS=2
MODULES=(stub-module)
PYTHON_BIN='$TMP/bin/model'
PATHS[slurm_logs]='$TMP/work/logs'
PATHS[checkpoint_root]='$TMP/work/checkpoints'
PATHS[logs_root]='$TMP/work/tensorboard'
PATHS[validation_root]='$TMP/work/validation'
PATHS[summary_root]='$TMP/work/validation/summary'
VALIDATE_GRID[fusion_alpha]='0.25|0.75'
OVERRIDES
assert_contains() { [[ "$1" == *"$2"* ]] || { printf 'Missing: %s\n' "$2" >&2; exit 1; }; }
arg_value() { awk -v key="$1" '$0 == key {getline; print; exit}' "$2"; }
expect_fail() {
  if "$@" > "$TMP/failure.out" 2>&1; then printf 'Unexpected success: %s\n' "$*" >&2; exit 1; fi
}
# A dry run should need neither Slurm nor a model interpreter and change no files.
output=$(bash "$ROOT/run_experiment.sh" --train --config "$CONFIG" --dry-run)
assert_contains "$output" 'train: 2 job(s)'
assert_contains "$output" '--array=0-1%4'
[[ ! -e "$TMP/work/logs" && ! -e "$STUB_MODEL_ARGS" && ! -e "$STUB_MODULES" ]]
output=$(bash "$ROOT/run_experiment.sh" --validate --config "$CONFIG" --dry-run)
assert_contains "$output" 'validate: 4 job(s)'
train_row=$(bash "$ROOT/run_experiment.slurm" --train --config "$CONFIG" --task-index 1 --dry-run)
validation_row=$(bash "$ROOT/run_experiment.slurm" --validate --config "$CONFIG" --task-index 3 --dry-run)
train_id=$(printf '%s' "$train_row" | sed -n 's/.*--experiment_id \([^ ]*\).*/\1/p')
validate_id=$(printf '%s' "$validation_row" | sed -n 's/.*--experiment_id \([^ ]*\).*/\1/p')
[[ -n "$train_id" && "$train_id" == "$validate_id" ]]
assert_contains "$validation_row" '--fusion_alpha 0.75'
[[ "$validation_row" != *'--epoch'* && "$validation_row" != *'--trainer_extra_json'* ]]

# Check the final full-grid row and multi-group argument expansion.
cp "$CONFIG" "$TMP/full.conf"
printf '\nMAX_JOBS=0\n' >> "$TMP/full.conf"
last_row=$(bash "$ROOT/run_experiment.slurm" --train --config "$TMP/full.conf" --task-index 23 --dry-run)
assert_contains "$last_row" '--mri_model resnet18'
assert_contains "$last_row" '--feature_groups clinical kinetic morphology heterogeneity'
assert_contains "$last_row" '--feature_model mlp'
assert_contains "$last_row" '--sensitivity_lambda 0.2'

# Resource, validation, and output changes must preserve checkpoint identities.
cp "$CONFIG" "$TMP/config with spaces/resources.conf"
cat >> "$TMP/config with spaces/resources.conf" <<OVERRIDES
JOB_TRAIN[memory]=64G
PATHS[validation_root]='$TMP/work/other-validation'
VALIDATE_GRID[fusion_alpha]='0.1|0.9'
OVERRIDES
same_row=$(bash "$ROOT/run_experiment.slurm" --validate --config "$TMP/config with spaces/resources.conf" --task-index 3 --dry-run)
assert_contains "$same_row" "--experiment_id $train_id"
printf '\nTRAIN[epoch]=31\n' >> "$TMP/config with spaces/resources.conf"
changed_row=$(bash "$ROOT/run_experiment.slurm" --validate --config "$TMP/config with spaces/resources.conf" --task-index 3 --dry-run)
[[ "$changed_row" != *"--experiment_id $train_id"* ]]

# Submission passes exact resource values and freezes both code and config.
bash "$ROOT/run_experiment.sh" --train --config "$CONFIG" > "$TMP/submit.out"
submission=$(cat "$STUB_SBATCH_ARGS")
assert_contains "$submission" '--array=0-1%4'
assert_contains "$submission" '--gres=gpu:1'
snapshot=$(arg_value --snapshot "$STUB_SBATCH_ARGS")
controller=$(awk '/\/run_experiment.slurm$/ {print; exit}' "$STUB_SBATCH_ARGS")
[[ -f "$snapshot" && -f "$controller" ]]
printf '\nTRAIN[epoch]=99\n' >> "$CONFIG"
# Simulate Slurm's spooled script path and execute from another working directory.
cp "$controller" "$TMP/work/slurm_script"
(cd "$TMP" && SLURM_JOB_ID=12345 SLURM_ARRAY_TASK_ID=1 bash "$TMP/work/slurm_script" --train --snapshot "$snapshot" --project-root "$ROOT") > "$TMP/worker.out"
[[ "$(arg_value --epoch "$STUB_MODEL_ARGS")" == 30 ]]
[[ "$(arg_value --experiment_id "$STUB_MODEL_ARGS")" == "$train_id" ]]
[[ "$(arg_value --imaging_patient_id_column "$STUB_MODEL_ARGS")" == 'Patient ID' ]]
[[ "$(arg_value --clinical_features_file "$STUB_MODEL_ARGS")" == "$TMP/config with spaces/mriBreastDuke/features/Clinical_and_Other_Features.xlsx" ]]
[[ "$(cat "$STUB_MODULES")" == 'load stub-module' && "$(cat "$STUB_MODEL_ARGS.cwd")" == "$ROOT" ]]
assert_contains "$(cat "$STUB_MODEL_ARGS")" '--use_annotation_boxes'
[[ "$(cat "$STUB_MODEL_ARGS")" != *'--include_sensitive'* ]]
# Tampering fails before sourcing the modified configuration.
chmod u+w "$snapshot"
printf '\necho tampered > "%s"\n' "$TMP/tampered" >> "$snapshot"
expect_fail env SLURM_JOB_ID=12345 bash "$controller" --train --snapshot "$snapshot" --project-root "$ROOT"
[[ ! -e "$TMP/tampered" ]]

# Summary gets one CPU-only task and hyphenated CLI flags. Dependencies survive.
bash "$ROOT/summarize_configurable_imaging_features_fusion.sh" --config "$CONFIG" --dependency afterok:12345 > "$TMP/summary.out"
submission=$(cat "$STUB_SBATCH_ARGS")
[[ "$submission" != *'--gres='* && "$submission" != *'--array='* ]]
assert_contains "$submission" '--dependency=afterok:12345'
snapshot=$(arg_value --snapshot "$STUB_SBATCH_ARGS")
controller=$(awk '/\/run_experiment.slurm$/ {print; exit}' "$STUB_SBATCH_ARGS")
SLURM_JOB_ID=12346 bash "$controller" --summarize --snapshot "$snapshot" --project-root "$ROOT" > "$TMP/summary-worker.out"
[[ "$(arg_value --expected-folds "$STUB_MODEL_ARGS")" == 5 ]]
[[ "$(arg_value --output-dir "$STUB_MODEL_ARGS")" == "$TMP/work/validation/summary" ]]

# Invalid grids/resources/booleans/indices fail before submission.
expect_fail bash "$ROOT/run_experiment.slurm" --train --validate --dry-run
expect_fail bash "$ROOT/run_experiment.slurm" --train --config "$CONFIG" --task-index 2 --dry-run
for setting in 'TRAIN_GRID[lr]="1e-4|"' 'TRAIN_GRID[lr]="1e-4|1e-4"' 'TRAIN[typo]=1' 'TRAIN[use_annotation_boxes]=yes' 'MAX_CONCURRENT=0' 'JOB_TRAIN[cpus]=0'; do
  cp "$CONFIG" "$TMP/invalid.conf"
  printf '\n%s\n' "$setting" >> "$TMP/invalid.conf"
  expect_fail bash "$ROOT/run_experiment.slurm" --train --config "$TMP/invalid.conf" --dry-run
 done
# Cleanup protects source/input/snapshot paths, including summary ancestor roots.
for setting in "PATHS[checkpoint_root]='$ROOT'" "PATHS[checkpoint_root]='$TMP/work/logs'" 'PATHS[checkpoint_root]=/'; do
  cp "$CONFIG" "$TMP/invalid.conf"
  printf '\nCLEAN_BEFORE_TRAIN=true\n%s\n' "$setting" >> "$TMP/invalid.conf"
  expect_fail bash "$ROOT/run_experiment.slurm" --train --config "$TMP/invalid.conf" --dry-run
 done
cp "$CONFIG" "$TMP/invalid.conf"
printf '\nCLEAN_BEFORE_VALIDATE=true\nPATHS[summary_root]="%s"\n' "$TMP/work" >> "$TMP/invalid.conf"
expect_fail bash "$ROOT/run_experiment.slurm" --validate --config "$TMP/invalid.conf" --dry-run
# Cleanup actually runs once in the controller and never again in its array tasks.
cp "$CONFIG" "$TMP/clean.conf"
printf '\nCLEAN_BEFORE_TRAIN=true\n' >> "$TMP/clean.conf"
mkdir -p "$TMP/work/checkpoints" "$TMP/work/tensorboard"
touch "$TMP/work/checkpoints/old" "$TMP/work/tensorboard/old"
bash "$ROOT/run_experiment.slurm" --train --config "$TMP/clean.conf" > "$TMP/clean.out"
[[ ! -e "$TMP/work/checkpoints" && ! -e "$TMP/work/tensorboard" ]]
snapshot=$(arg_value --snapshot "$STUB_SBATCH_ARGS")
controller=$(awk '/\/run_experiment.slurm$/ {print; exit}' "$STUB_SBATCH_ARGS")
mkdir -p "$TMP/work/checkpoints"
touch "$TMP/work/checkpoints/new"
SLURM_JOB_ID=12345 SLURM_ARRAY_TASK_ID=0 bash "$controller" --train --snapshot "$snapshot" --project-root "$ROOT" > "$TMP/clean-worker.out"
[[ -f "$TMP/work/checkpoints/new" ]]
# Every production .sh is now just a launcher, with workload logic in .slurm.
while IFS= read -r file; do
  [[ $(wc -l < "$file") -le 5 ]] || { printf 'Nontrivial launcher: %s\n' "$file" >&2; exit 1; }
  bash -n "$file"
done <<< "$(find "$ROOT" -name '*.sh' -not -path "$TMP/*")"
while IFS= read -r file; do bash -n "$file"; done <<< "$(find "$ROOT" -name '*.slurm' -not -path "$TMP/*")"
printf 'PASS: Slurm controller integration tests\n'
