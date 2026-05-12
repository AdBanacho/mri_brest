#!/bin/bash -l
#SBATCH --job-name=validate_dense
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --account=plgvirtudrel2026-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ml ML-bundle

cd /net/home/plgrid/plgabanacho/mri_brest
pip install -e .
pip install --no-cache-dir pytorch_lightning lightning==2.5.6 torchmetrics==1.8.2 requests monai pandas scikit-learn nibabel openpyxl tensorboardX einops

python -m mriBreastDuke.validate_best_checkpoints --model 1 --is_binary_classification True

