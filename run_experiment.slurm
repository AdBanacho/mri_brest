#!/bin/bash -l
# This Slurm controller is also the batch task script. All planning, configuration
# loading, cleanup, submission, and command construction live in this file.
# Invoke with bash for submission; the controller calls sbatch on a frozen copy.
set -euo pipefail
export LC_ALL=C
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'HELP'
Usage: bash run_experiment.slurm --train|--validate|--summarize [options]
  --config FILE       Shell configuration (default: experiment.conf)
  --dry-run           Print submission and task 0 without modifying files
  --dependency SPEC   Slurm dependency, e.g. afterok:12345
  --help              Show this help
Aliases: --training, --validation
HELP
}
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SELF=$(realpath -- "${BASH_SOURCE[0]}")
config= stage= dependency= task_index= snapshot= dry_run=false
while (($#)); do
  case "$1" in
    --train|--training|--validate|--validation|--summarize)
      [[ -z "$stage" ]] || fail 'Select exactly one stage'
      case "$1" in --train*) stage=train;; --valid*) stage=validate;; *) stage=summarize;; esac
      shift;;
    --config|--dependency|--task-index|--snapshot|--project-root)
      (($# >= 2)) || fail "$1 needs a value"
      case "$1" in
        --config) config=$2;; --dependency) dependency=$2;;
        --task-index) task_index=$2;; --snapshot) snapshot=$2;;
        --project-root) PROJECT_ROOT=$(realpath -e -- "$2");;
      esac
      shift 2;;
    --dry-run) dry_run=true; shift;;
    --help|-h) usage; exit 0;;
    *) fail "Unknown argument: $1";;
  esac
done
[[ -n "$stage" ]] || { usage >&2; exit 2; }
[[ -z "$config" || -z "$snapshot" ]] || fail 'Use --config or --snapshot'
if [[ -n "$snapshot" ]]; then
  config=$(realpath -e -- "$snapshot")
  actual=$(sha256sum -- "$config"); actual=${actual%% *}
  [[ "${config##*/}" == "experiment-$actual.conf" ]] || fail 'Configuration snapshot was changed'
else
  config=$(realpath -e -- "${config:-$PROJECT_ROOT/experiment.conf}")
fi
config_dir=$(dirname -- "$config")
# Configurations are trusted shell code, not executable launchers.
source "$config"

TRAIN_KEYS='mri_model subtraction_mode feature_groups feature_model feature_selector imaging_patient_id_column allow_missing_imaging_features include_sensitive use_annotation_boxes epoch num_folds threshold_calibration_folds batch_size num_workers positive_boost sensitivity_lambda lr xgb_n_estimators xgb_max_depth xgb_learning_rate xgb_subsample xgb_colsample_bytree xgb_n_jobs lasso_cv_folds lasso_cs lasso_max_iter lasso_tolerance lasso_min_features lasso_n_jobs lasso_plot_top_n mlp_hidden_layers mlp_alpha mlp_learning_rate mlp_max_iter xgb_extra_json mlp_extra_json mri_extra_json trainer_extra_json seed target_size'
VALIDATE_KEYS='num_workers fusion_alpha'
SUMMARY_KEYS='expected_folds top_n include_incomplete'
PATH_KEYS='clinical_features_file metadata_file imaging_features_file annotation_boxes_file image_root prepared_root subtraction_root logs_root checkpoint_root validation_root summary_root slurm_logs'
BOOL_KEYS='allow_missing_imaging_features include_sensitive use_annotation_boxes include_incomplete'
VALIDATION_TRAIN_KEYS='mri_model subtraction_mode feature_groups feature_model feature_selector imaging_patient_id_column allow_missing_imaging_features include_sensitive use_annotation_boxes num_folds threshold_calibration_folds batch_size num_workers positive_boost sensitivity_lambda lr lasso_cv_folds lasso_cs lasso_max_iter lasso_tolerance lasso_min_features lasso_n_jobs mri_extra_json seed target_size'
contains() { [[ " $1 " == *" $2 "* ]]; }
check_keys() {
  local -n values=$1
  local key
  for key in "${!values[@]}"; do contains "$2" "$key" || fail "Unknown key in $1: $key"; done
}
for spec in 'TRAIN:TRAIN_KEYS' 'TRAIN_GRID:TRAIN_KEYS' 'VALIDATE:VALIDATE_KEYS' 'VALIDATE_GRID:VALIDATE_KEYS' 'SUMMARIZE:SUMMARY_KEYS' 'PATHS:PATH_KEYS'; do
  allowed=${spec#*:}
  check_keys "${spec%%:*}" "${!allowed}"
done
[[ "$MAX_JOBS" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'MAX_JOBS must be >= 0'
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || fail 'MAX_CONCURRENT must be >= 1'
for boolean in "$CLEAN_BEFORE_TRAIN" "$CLEAN_BEFORE_VALIDATE"; do
  [[ "$boolean" == true || "$boolean" == false ]] || fail 'Cleanup flags must be true or false'
done
for key in $PATH_KEYS; do
  [[ -n "${PATHS[$key]:-}" ]] || fail "Missing PATHS[$key]"
  value=${PATHS[$key]}
  [[ "$value" == /* ]] || value=$config_dir/$value
  PATHS[$key]=$(realpath -m -- "$value")
done
for resource_stage in TRAIN VALIDATE SUMMARIZE; do
  declare -n resources=JOB_$resource_stage
  check_keys "JOB_$resource_stage" 'job_name cpus gpus memory time account partition'
  [[ "${resources[cpus]}" =~ ^[1-9][0-9]*$ ]] || fail 'cpus must be positive'
  [[ "${resources[gpus]}" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'gpus must be nonnegative'
  for key in job_name memory time; do [[ -n "${resources[$key]}" ]] || fail "Missing resource $key"; done
  unset -n resources
done

# Sorted keys make task ordering and experiment identities deterministic.
# A grid is encoded with | separators; no eval or whitespace splitting of scalars.
grid_keys() {
  local -n grid=$1
  if ((${#grid[@]})); then printf '%s\n' "${!grid[@]}" | sort; fi
}
grid_count() {
  local -n grid=$1
  local key value total=1 choices choice
  local -A seen=()
  for key in "${!grid[@]}"; do
    value=${grid[$key]}
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || fail "Grid choices must be on one line: $key"
    [[ -n "$value" && "$value" != '|'* && "$value" != *'|' && "$value" != *'||'* ]] || fail "Empty grid choice: $1[$key]"
    IFS='|' read -r -a choices <<< "$value"
    seen=()
    for choice in "${choices[@]}"; do
      [[ -z "${seen[$choice]:-}" ]] || fail "Duplicate grid choice: $key=$choice"
      seen[$choice]=1
    done
    total=$((total * ${#choices[@]}))
    ((total > 0 && total <= 1000000)) || fail 'Grid exceeds 1,000,000 tasks'
  done
  printf '%s' "$total"
}
select_grid() {
  local -n grid=$1 result=$3
  local index=$2 key choices position
  local -a keys=()
  local sorted_keys
  sorted_keys=$(grid_keys "$1")
  [[ -z "$sorted_keys" ]] || mapfile -t keys <<< "$sorted_keys"
  for ((position=${#keys[@]}-1; position>=0; position--)); do
    key=${keys[$position]}
    IFS='|' read -r -a choices <<< "${grid[$key]}"
    result[$key]=${choices[$((index % ${#choices[@]}))]}
    index=$((index / ${#choices[@]}))
  done
}
train_count=$(grid_count TRAIN_GRID)
if ((MAX_JOBS > 0 && MAX_JOBS < train_count)); then train_count=$MAX_JOBS; fi
validate_choices=$(grid_count VALIDATE_GRID)
case "$stage" in train) count=$train_count;; validate) count=$((train_count * validate_choices));; summarize) count=1;; esac

# Serialize resolved shell values, not a generated TOML/JSON configuration.
write_snapshot() {
  declare -p PYTHON_BIN MODULES MAX_JOBS MAX_CONCURRENT CLEAN_BEFORE_TRAIN CLEAN_BEFORE_VALIDATE \
    PATHS JOB_TRAIN JOB_VALIDATE JOB_SUMMARIZE EXTRA_SBATCH_TRAIN EXTRA_SBATCH_VALIDATE \
    EXTRA_SBATCH_SUMMARIZE TRAIN TRAIN_GRID VALIDATE VALIDATE_GRID SUMMARIZE
}
build_command() {
  local index=$1 key identity train_index=$1 validation_index=0 value
  local -a words=()
  declare -A row=() validation=() options=()
  if [[ "$stage" == summarize ]]; then
    for key in "${!SUMMARIZE[@]}"; do options[$key]=${SUMMARIZE[$key]}; done
    options[input_dir]=${PATHS[validation_root]}; options[output_dir]=${PATHS[summary_root]}
    COMMAND=("$PYTHON_BIN" -m mriBreastDuke.summarize_configurable_imaging_features_fusion)
  else
    if [[ "$stage" == validate ]]; then
      train_index=$((index / validate_choices)); validation_index=$((index % validate_choices))
    fi
    for key in "${!TRAIN[@]}"; do row[$key]=${TRAIN[$key]}; done
    select_grid TRAIN_GRID "$train_index" row
    # Hash all training settings, input paths and cache paths. Exclude output and
    # scheduler settings, so validation-only changes never select new checkpoints.
    identity=$({
      while IFS= read -r key; do printf '%s\0%s\0' "$key" "${row[$key]}"; done <<< "$(printf '%s\n' "${!row[@]}" | sort)"
      for key in clinical_features_file metadata_file imaging_features_file annotation_boxes_file image_root prepared_root subtraction_root; do
        printf '%s\0%s\0' "$key" "${PATHS[$key]}"
      done
    } | sha256sum); identity=${identity%% *}
    for key in "${!row[@]}"; do
      if [[ "$stage" == train ]] || contains "$VALIDATION_TRAIN_KEYS" "$key"; then options[$key]=${row[$key]}; fi
    done
    options[experiment_id]=cfg${identity:0:12}
    for key in clinical_features_file metadata_file imaging_features_file annotation_boxes_file image_root prepared_root subtraction_root checkpoint_root; do
      options[$key]=${PATHS[$key]}
    done
    if [[ "$stage" == train ]]; then
      options[logs_root]=${PATHS[logs_root]}
      COMMAND=("$PYTHON_BIN" -m mriBreastDuke.configurable_imaging_features_fusion_workflow)
    else
      for key in "${!VALIDATE[@]}"; do validation[$key]=${VALIDATE[$key]}; done
      select_grid VALIDATE_GRID "$validation_index" validation
      for key in "${!validation[@]}"; do options[$key]=${validation[$key]}; done
      options[output_dir]=${PATHS[validation_root]}
      COMMAND=("$PYTHON_BIN" -m mriBreastDuke.validate_configurable_imaging_features_fusion)
    fi
  fi
  while IFS= read -r key; do
    value=${options[$key]}
    flag=--$key
    [[ "$stage" != summarize ]] || flag=${flag//_/-}
    if contains "$BOOL_KEYS" "$key"; then
      [[ "$value" == true || "$value" == false ]] || fail "$key must be true or false"
      [[ "$value" != true ]] || COMMAND+=("$flag")
    elif [[ "$key" == feature_groups ]]; then
      IFS=',' read -r -a words <<< "$value"
      COMMAND+=("$flag" "${words[@]}")
    elif [[ "$key" == target_size ]]; then
      read -r -a words <<< "$value"
      ((${#words[@]} == 3)) || fail 'target_size needs three positive integers'
      for word in "${words[@]}"; do [[ "$word" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid target_size'; done
      COMMAND+=("$flag" "${words[@]}")
    else
      COMMAND+=("$flag" "$value")
    fi
  done <<< "$(printf '%s\n' "${!options[@]}" | sort)"
}
print_command() { printf '%q ' "$@"; printf '\n'; }

if [[ -n "$snapshot" && -n "${SLURM_JOB_ID:-}" && -z "$task_index" ]]; then
  task_index=${SLURM_ARRAY_TASK_ID:-0}
fi
if [[ -n "$task_index" ]]; then
  [[ "$task_index" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'Invalid task index'
  ((task_index < count)) || fail "Task index outside 0-$((count - 1))"
  build_command "$task_index"
  print_command "${COMMAND[@]}"
  if [[ "$dry_run" != true ]]; then
    cd -- "$PROJECT_ROOT"
    for module_name in "${MODULES[@]}"; do module load "$module_name"; done
    exec "${COMMAND[@]}"
  fi
  exit 0
fi
[[ -z "$snapshot" ]] || fail 'A snapshot is only used by a Slurm task'

# Preflight every row before any filesystem changes or cleanup.
for ((index=0; index<count; index++)); do build_command "$index"; done
build_command 0
upper_stage=${stage^^}
declare -n job=JOB_$upper_stage extra_sbatch=EXTRA_SBATCH_$upper_stage
# Managed options must be edited in JOB_* / MAX_CONCURRENT instead of overriding
# controller internals through EXTRA_SBATCH_*.
for option in "${extra_sbatch[@]}"; do
  [[ "$option" == --*=* ]] || fail 'EXTRA_SBATCH options must use --name=value'
  case "${option%%=*}" in
    --qos|--constraint|--reservation|--nice|--mail-type|--mail-user) ;;
    *) fail "Unsupported EXTRA_SBATCH option: $option";;
  esac
done
SBATCH=(sbatch --parsable "--job-name=${job[job_name]}" --nodes=1 --ntasks=1 \
  "--cpus-per-task=${job[cpus]}" "--mem=${job[memory]}" "--time=${job[time]}" \
  "--output=${PATHS[slurm_logs]}/%x-%A_%a.out" "--error=${PATHS[slurm_logs]}/%x-%A_%a.err")
((job[gpus] == 0)) || SBATCH+=("--gres=gpu:${job[gpus]}")
for key in account partition; do [[ -z "${job[$key]:-}" ]] || SBATCH+=("--$key=${job[$key]}"); done
SBATCH+=("${extra_sbatch[@]}")
[[ "$stage" == summarize ]] || SBATCH+=("--array=0-$((count - 1))%$MAX_CONCURRENT")
[[ -z "$dependency" ]] || SBATCH+=("--dependency=$dependency")

# Cleanup happens once in the submitting controller, never in array tasks.
cleanup_targets=()
if [[ "$stage" == train && "$CLEAN_BEFORE_TRAIN" == true ]]; then
  cleanup_targets=("${PATHS[checkpoint_root]}" "${PATHS[logs_root]}")
elif [[ "$stage" == validate && "$CLEAN_BEFORE_VALIDATE" == true ]]; then
  cleanup_targets=("${PATHS[validation_root]}")
  [[ "${PATHS[summary_root]}" == "${PATHS[validation_root]}"/* ]] || cleanup_targets+=("${PATHS[summary_root]}")
fi
for target in "${cleanup_targets[@]}"; do
  [[ "$target" != / && "$target" != "$HOME" && "$PROJECT_ROOT" != "$target" && "$PROJECT_ROOT" != "$target"/* && "$HOME" != "$target"/* ]] || fail "Refusing broad cleanup: $target"
  # Require at least three path components and reject overlap with all inputs,
  # cache roots, Slurm logs/snapshots, and unrelated output roots.
  [[ "$target" == /*/*/* ]] || fail "Refusing broad cleanup: $target"
  for key in clinical_features_file metadata_file imaging_features_file annotation_boxes_file image_root prepared_root subtraction_root slurm_logs; do
    protected=${PATHS[$key]}
    [[ "$protected" != "$target" && "$protected" != "$target"/* && "$target" != "$protected"/* ]] || fail "Cleanup overlaps $key: $target"
  done
  for key in checkpoint_root logs_root validation_root summary_root; do
    protected=${PATHS[$key]}
    # A nested summary inside validation_root is the only permitted overlap.
    [[ "$stage" != validate || "$key" != summary_root || "$protected" != "$target"/* ]] || continue
    [[ "$protected" == "$target" ]] && continue
    [[ "$protected" != "$target"/* && "$target" != "$protected"/* ]] || fail "Cleanup overlaps $key: $target"
  done
  [[ ! -e "$target" || -d "$target" ]] || fail "Cleanup target is not a directory: $target"
done
printf '%s: %s job(s), max concurrent %s\n' "$stage" "$count" "$MAX_CONCURRENT"
if [[ "$dry_run" == true ]]; then
  for target in "${cleanup_targets[@]}"; do printf 'Would remove: %s\n' "$target"; done
  print_command "${SBATCH[@]}" '<frozen-controller.slurm>' --"$stage" --snapshot '<resolved-experiment.conf>' --project-root "$PROJECT_ROOT"
  printf 'Example task 0: '; print_command "${COMMAND[@]}"
  exit 0
fi
command -v sbatch >/dev/null || fail 'sbatch is unavailable (use --dry-run to inspect locally)'
mkdir -p -- "${PATHS[slurm_logs]}"
run_dir=$(mktemp -d "${PATHS[slurm_logs]}/run.XXXXXXXX")
write_snapshot > "$run_dir/resolved.conf"
digest=$(sha256sum -- "$run_dir/resolved.conf"); digest=${digest%% *}
snapshot_path=$run_dir/experiment-$digest.conf
mv -- "$run_dir/resolved.conf" "$snapshot_path"
cp -- "$SELF" "$run_dir/run_experiment.slurm"
chmod a-w -- "$snapshot_path" "$run_dir/run_experiment.slurm"
for target in "${cleanup_targets[@]}"; do printf 'Removing previous output: %s\n' "$target"; rm -rf -- "$target"; done
job_id=$("${SBATCH[@]}" "$run_dir/run_experiment.slurm" --"$stage" --snapshot "$snapshot_path" --project-root "$PROJECT_ROOT")
printf 'Submitted %s: Slurm ID %s\nConfiguration: %s\n' "$stage" "$job_id" "$snapshot_path"
