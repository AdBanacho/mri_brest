#!/bin/bash -l
#SBATCH --job-name=dense
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --account=plgvirtudrel-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

ml ML-bundle

cd /net/home/plgrid/plgabanacho/mri_brest
pip install -e .
pip install --no-cache-dir requests monai pandas nibabel openpyxl einops

python -m mriBreastDuke.dataDownloaders.downloadDicom

