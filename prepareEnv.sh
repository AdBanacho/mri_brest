#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=plgvirtudrel-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200

ml ML-bundle

cd /net/home/plgrid/plgabanacho/mri_brest
pip install -e .
pip install --no-cache-dir pytorch_lightning lightning==2.5.6 torchmetrics==1.8.2 requests monai pandas scikit-learn nibabel openpyxl tensorboardX

python -m mriBreastDuke.trainingWorkflow

