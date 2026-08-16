#!/bin/bash -l
#SBATCH --job-name=validate_resnet18_sub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --account=plgvirtudrel2026-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

ml ML-bundle

cd /net/home/plgrid/plgabanacho/mri_brest
pip install -e .
pip install --no-cache-dir pytorch_lightning lightning==2.5.6 torchmetrics==1.8.2 requests monai pandas scikit-learn nibabel openpyxl tensorboard einops matplotlib

LRS=(1e-2 1e-3 1e-4 1e-5 1e-6)
MODES=(post_minus_pre consecutive)

MODE_INDEX=$((SLURM_ARRAY_TASK_ID % ${#MODES[@]}))
LR_INDEX=$((SLURM_ARRAY_TASK_ID / ${#MODES[@]}))
MODE=${MODES[$MODE_INDEX]}
LR=${LRS[$LR_INDEX]}

echo "========================================"
echo "Validating 3D ResNet18 subtraction experiment"
echo "Learning rate: $LR"
echo "Subtraction mode: $MODE"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "========================================"

python -m mriBreastDuke.validate_best_checkpoints \
    --model 2 \
    --lr "$LR" \
    --sensitivity_lambda 0.3 \
    --positive_boost 1.0 \
    --batch_size 8 \
    --num_workers 2 \
    --top_k_checkpoints 1 \
    --subtraction_mode "$MODE"
