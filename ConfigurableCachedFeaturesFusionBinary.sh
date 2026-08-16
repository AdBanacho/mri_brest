#!/bin/bash -l
#SBATCH --job-name=config_cached_features
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

ml ML-bundle

sleep $(( (SLURM_ARRAY_TASK_ID % 8) * 30 ))

cd /net/home/plgrid/plgabanacho/mri_brest

pip install -e .
pip install --no-cache-dir xgboost pytorch_lightning lightning==2.5.6 torchmetrics==1.8.2 monai pandas 'scikit-learn>=1.7,<2' nibabel SimpleITK filelock openpyxl tensorboard matplotlib

# Starter grid: 2 MRI models x 2 inputs x 2 feature sets x 2 tabular
# models x 1 batch size x 2 boosts x 1 sensitivity penalty x 1 LR = 32.
MRI_MODELS=(densenet121 resnet18)
SUBTRACTIONS=(none post_minus_pre)
FEATURE_GROUP_SETS=("clinical" "clinical,morphology,heterogeneity")
FEATURE_MODELS=(xgboost mlp)
BATCH_SIZES=(4)
POS_BOOSTS=(1.0 2.0)
SENS_LAMBDAS=(0.3)
LRS=(1e-4)

RADIOMICS_CSV=${RADIOMICS_CSV:-/net/home/plgrid/plgabanacho/mri_brest/duke_mri_features.csv}
IMAGE_ROOT=${IMAGE_ROOT:-/net/storage/pr3/plgrid/plggsiwomipipan/aBanacho_duke_oncotype/tciaNifti}

# Required on the first non-clinical run if RADIOMICS_CSV does not exist.
# Use either LESION_MASKS_CSV or LESION_MASK_ROOT.
LESION_MASKS_CSV=${LESION_MASKS_CSV:-}
LESION_MASK_ROOT=${LESION_MASK_ROOT:-}
REGISTERED_MASK_ROOT=${REGISTERED_MASK_ROOT:-/net/home/plgrid/plgabanacho/mri_brest/registered_lesion_masks}
MASK_TRANSFORM_COLUMN=${MASK_TRANSFORM_COLUMN:-}

N_MRI=${#MRI_MODELS[@]}
N_SUB=${#SUBTRACTIONS[@]}
N_GROUPS=${#FEATURE_GROUP_SETS[@]}
N_FEATURE_MODEL=${#FEATURE_MODELS[@]}
N_BS=${#BATCH_SIZES[@]}
N_BOOST=${#POS_BOOSTS[@]}
N_SENS=${#SENS_LAMBDAS[@]}
N_LR=${#LRS[@]}
TOTAL_CONFIGS=$(( N_MRI * N_SUB * N_GROUPS * N_FEATURE_MODEL * N_BS * N_BOOST * N_SENS * N_LR ))

if (( SLURM_ARRAY_TASK_ID >= TOTAL_CONFIGS )); then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds grid size $TOTAL_CONFIGS"
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
BS=${BATCH_SIZES[$BS_IDX]}
BOOST=${POS_BOOSTS[$BOOST_IDX]}
SENS=${SENS_LAMBDAS[$SENS_IDX]}
LR=${LRS[$LR_IDX]}

IFS=',' read -r -a FEATURE_GROUP_ARGS <<< "$FEATURE_GROUPS_CSV"
RADIOMICS_ARGS=()
if [[ "$FEATURE_GROUPS_CSV" != "clinical" ]]; then
    RADIOMICS_ARGS=(
        --radiomics_csv "$RADIOMICS_CSV"
        --radiomics_key studyId
        --image_root "$IMAGE_ROOT"
        --registered_mask_root "$REGISTERED_MASK_ROOT"
    )
    if [[ -n "$LESION_MASKS_CSV" ]]; then
        RADIOMICS_ARGS+=(--lesion_masks_csv "$LESION_MASKS_CSV")
    elif [[ -n "$LESION_MASK_ROOT" ]]; then
        RADIOMICS_ARGS+=(--lesion_mask_root "$LESION_MASK_ROOT")
    elif [[ ! -s "$RADIOMICS_CSV" ]]; then
        echo "Feature cache is missing: $RADIOMICS_CSV"
        echo "Set LESION_MASKS_CSV or LESION_MASK_ROOT so it can be created."
        exit 3
    fi
    if [[ -n "$MASK_TRANSFORM_COLUMN" ]]; then
        RADIOMICS_ARGS+=(--mask_transform_column "$MASK_TRANSFORM_COLUMN")
    fi
fi

echo "========================================"
echo "Configurable cached-features fusion grid"
echo "Task: $SLURM_ARRAY_TASK_ID / $((TOTAL_CONFIGS - 1))"
echo "MRI model: $MRI_MODEL"
echo "Subtraction: $SUBTRACTION"
echo "Feature groups: $FEATURE_GROUPS_CSV"
echo "Feature model: $FEATURE_MODEL"
echo "Batch size: $BS"
echo "Positive boost: $BOOST"
echo "Sensitivity lambda: $SENS"
echo "MRI learning rate: $LR"
if [[ "$FEATURE_GROUPS_CSV" != "clinical" ]]; then
    echo "Radiomics cache: $RADIOMICS_CSV"
fi
echo "========================================"

python -m mriBreastDuke.configurable_cached_features_fusion_workflow \
    --mri_model "$MRI_MODEL" \
    --subtraction_mode "$SUBTRACTION" \
    --feature_groups "${FEATURE_GROUP_ARGS[@]}" \
    --feature_model "$FEATURE_MODEL" \
    --epoch 30 \
    --num_folds 5 \
    --lr "$LR" \
    --batch_size "$BS" \
    --num_workers 4 \
    --positive_boost "$BOOST" \
    --sensitivity_lambda "$SENS" \
    --fusion_alpha 0.5 \
    "${RADIOMICS_ARGS[@]}"

