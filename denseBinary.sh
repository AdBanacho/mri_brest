#!/bin/bash -l
#SBATCH --job-name=dense_grid_small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --account=plgvirtudrel2026-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --array=0-27%9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

ml ML-bundle

sleep $(( (SLURM_ARRAY_TASK_ID % 9) * 30 ))

cd /net/home/plgrid/plgabanacho/mri_brest

pip install -e .
pip install --no-cache-dir pytorch_lightning lightning==2.5.6 torchmetrics==1.8.2 requests monai pandas scikit-learn nibabel openpyxl tensorboard einops matplotlib

LRS=(1e-3 1e-4 1e-5)
SENS_LAMBDAS=(0.3 0.7 1.0)
POS_BOOSTS=(1.0 2.0 3.0)
BATCH_SIZES=(8, 16)

N_LR=${#LRS[@]}
N_SENS=${#SENS_LAMBDAS[@]}
N_BOOST=${#POS_BOOSTS[@]}
N_BS=${#BATCH_SIZES[@]}

IDX=$SLURM_ARRAY_TASK_ID

BS_IDX=$(( IDX % N_BS ))
IDX=$(( IDX / N_BS ))

BOOST_IDX=$(( IDX % N_BOOST ))
IDX=$(( IDX / N_BOOST ))

SENS_IDX=$(( IDX % N_SENS ))
IDX=$(( IDX / N_SENS ))

LR_IDX=$(( IDX % N_LR ))

LR=${LRS[$LR_IDX]}
SENS=${SENS_LAMBDAS[$SENS_IDX]}
BOOST=${POS_BOOSTS[$BOOST_IDX]}
BS=${BATCH_SIZES[$BS_IDX]}

echo "========================================"
echo "DenseNet small hyperparameter grid"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "LR: $LR"
echo "Sensitivity lambda: $SENS"
echo "Positive boost: $BOOST"
echo "Batch size: $BS"
echo "========================================"

python -m mriBreastDuke.trainingWorkflow \
    --model 1 \
    --epoch 30 \
    --is_binary_classification True \
    --lr "$LR" \
    --sensitivity_lambda "$SENS" \
    --positive_boost "$BOOST" \
    --batch_size "$BS" \
    --num_workers 2