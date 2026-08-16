#!/bin/bash -l
#SBATCH --job-name=mri_xgb_fusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --account=plgvirtudrel2026-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ml ML-bundle

cd /net/home/plgrid/plgabanacho/mri_brest
pip install -e .
pip install --no-cache-dir pytorch_lightning lightning==2.5.6 torchmetrics==1.8.2 requests monai pandas scikit-learn xgboost nibabel openpyxl tensorboard einops matplotlib

python -m mriBreastDuke.xgboost_fu \
    --backbone resnet18 \
    --epoch 50 \
    --lr 1e-4 \
    --fusion_alpha 0.5 \
    --xgb_n_estimators 300 \
    --xgb_max_depth 3 \
    --xgb_learning_rate 0.03 \
    --batch_size 4 \
    --num_workers 8
