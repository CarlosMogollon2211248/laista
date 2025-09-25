#!/bin/bash
#SBATCH --job-name=unet_training_c
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G  # 
#SBATCH --time=0-80:00:00
#SBATCH --output=job_out/unet_training_c%j.out
# Send email
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

# Set path to repository 
CODE_PATH="/work/ececis_research/kevinroa/TG/laista/"
vpkg_require anaconda/2024.02
source ~/.bashrc
conda init
conda activate /work/ececis_research/kevinroa/CONDA/cunet
cd ${CODE_PATH}

# Run the training script
python train.py