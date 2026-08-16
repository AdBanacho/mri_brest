#!/bin/bash -l
#SBATCH --job-name=resnet18_lr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --account=plgvirtudrel2026-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

ml ML-bundle

cd /net/home/plgrid/plgabanacho/mri_brest
pip install -e .
pip install --no-cache-dir pytorch_lightning lightning==2.5.6 torchmetrics==1.8.2 requests monai pandas scikit-learn nibabel openpyxl tensorboard einops matplotlib

LRS=(1e-2 1e-3 1e-4 1e-5 1e-6)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Running 3D ResNet18 with learning rate: $LR"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "========================================"

python -m mriBreastDuke.trainingWorkflow \
    --model 2 \
    --epoch 30 \
    --lr "$LR"
