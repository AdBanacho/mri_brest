#!/bin/bash -l
#SBATCH --job-name=validate_imaging_features_fusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --account=plgvirtudrel2026-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --array=0-31%8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

ml ML-bundle

sleep $(( (SLURM_ARRAY_TASK_ID % 8) * 30 ))

cd /net/home/plgrid/plgabanacho/mri_brest

pip install -e .
pip install --no-cache-dir xgboost pytorch-lightning==2.5.6 torchmetrics==1.8.2 monai pandas 'scikit-learn>=1.7,<2' nibabel filelock openpyxl tensorboard matplotlib joblib

MRI_MODELS=(densenet121 resnet18)
SUBTRACTIONS=(none post_minus_pre)
FEATURE_GROUP_SETS=("clinical" "clinical,kinetic,morphology,heterogeneity")
FEATURE_MODELS=(xgboost mlp)
BATCH_SIZES=(4)
POS_BOOSTS=(1.0 2.0)
SENS_LAMBDAS=(0.3)
LRS=(1e-4)

FUSION_ALPHA=${FUSION_ALPHA:-0.5}
IMAGING_FEATURES_FILE=${IMAGING_FEATURES_FILE:-/net/home/plgrid/plgabanacho/mri_brest/mriBreastDuke/features/Imaging_Features.xlsx}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/net/scratch/hscra/plgrid/plgabanacho/check_points}
VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR:-validation_charts}

N_MRI=${#MRI_MODELS[@]}
N_SUB=${#SUBTRACTIONS[@]}
N_GROUPS=${#FEATURE_GROUP_SETS[@]}
N_FEATURE_MODEL=${#FEATURE_MODELS[@]}
N_BS=${#BATCH_SIZES[@]}
N_BOOST=${#POS_BOOSTS[@]}
N_SENS=${#SENS_LAMBDAS[@]}
N_LR=${#LRS[@]}
TOTAL_CONFIGS=$(( N_MRI * N_SUB * N_GROUPS * N_FEATURE_MODEL * N_BS * N_BOOST * N_SENS * N_LR ))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_CONFIGS )); then
    echo "Task $SLURM_ARRAY_TASK_ID is outside the grid 0-$((TOTAL_CONFIGS - 1))"
    exit 2
fi

IDX=$SLURM_ARRAY_TASK_ID
LR_IDX=$(( IDX % N_LR )); IDX=$(( IDX / N_LR ))
SENS_IDX=$(( IDX % N_SENS )); IDX=$(( IDX / N_SENS ))
BOOST_IDX=$(( IDX % N_BOOST )); IDX=$(( IDX / N_BOOST ))
BS_IDX=$(( IDX % N_BS )); IDX=$(( IDX / N_BS ))
FEATURE_MODEL_IDX=$(( IDX % N_FEATURE_MODEL )); IDX=$(( IDX / N_FEATURE_MODEL ))
GROUP_IDX=$(( IDX % N_GROUPS )); IDX=$(( IDX / N_GROUPS ))
SUB_IDX=$(( IDX % N_SUB )); IDX=$(( IDX / N_SUB ))
MRI_IDX=$(( IDX % N_MRI ))

MRI_MODEL=${MRI_MODELS[$MRI_IDX]}
SUBTRACTION=${SUBTRACTIONS[$SUB_IDX]}
FEATURE_GROUPS_CSV=${FEATURE_GROUP_SETS[$GROUP_IDX]}
FEATURE_MODEL=${FEATURE_MODELS[$FEATURE_MODEL_IDX]}
BATCH_SIZE=${BATCH_SIZES[$BS_IDX]}
POSITIVE_BOOST=${POS_BOOSTS[$BOOST_IDX]}
SENSITIVITY_LAMBDA=${SENS_LAMBDAS[$SENS_IDX]}
LEARNING_RATE=${LRS[$LR_IDX]}

IFS=',' read -r -a FEATURE_GROUP_ARGS <<< "$FEATURE_GROUPS_CSV"
IMAGING_ARGS=()
if [[ "$FEATURE_GROUPS_CSV" != "clinical" ]]; then
    if [[ ! -s "$IMAGING_FEATURES_FILE" ]]; then
        echo "Missing Imaging_Features workbook: $IMAGING_FEATURES_FILE"
        exit 3
    fi
    IMAGING_ARGS=(--imaging_features_file "$IMAGING_FEATURES_FILE")
fi

echo "========================================"
echo "Validate Imaging_Features.xlsx fusion experiment"
echo "Task: $SLURM_ARRAY_TASK_ID / $((TOTAL_CONFIGS - 1))"
echo "MRI model: $MRI_MODEL"
echo "Subtraction: $SUBTRACTION"
echo "Feature groups: $FEATURE_GROUPS_CSV"
echo "Feature model: $FEATURE_MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Positive boost: $POSITIVE_BOOST"
echo "Sensitivity lambda: $SENSITIVITY_LAMBDA"
echo "MRI learning rate: $LEARNING_RATE"
echo "Fusion alpha: $FUSION_ALPHA"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Validation output: $VALIDATION_OUTPUT_DIR"
echo "========================================"

python -m mriBreastDuke.validate_configurable_imaging_features_fusion \
    --mri_model "$MRI_MODEL" \
    --subtraction_mode "$SUBTRACTION" \
    --feature_groups "${FEATURE_GROUP_ARGS[@]}" \
    --feature_model "$FEATURE_MODEL" \
    --num_folds 5 \
    --batch_size "$BATCH_SIZE" \
    --num_workers 4 \
    --positive_boost "$POSITIVE_BOOST" \
    --sensitivity_lambda "$SENSITIVITY_LAMBDA" \
    --lr "$LEARNING_RATE" \
    --fusion_alpha "$FUSION_ALPHA" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --output_dir "$VALIDATION_OUTPUT_DIR" \
    "${IMAGING_ARGS[@]}"
