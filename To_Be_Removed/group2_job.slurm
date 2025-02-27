#! /bin/bash
# ====================================
#SBATCH --job-name=group2-garbage-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
# ====================================

# Script to initialize conda (base)
source ~/software/init-conda

# Activate your environment
conda activate pytorch

# Execute the python script
python $SLURM_SUBMIT_DIR/Assignment2.py

# Deactivate your environment
conda deactivate

# Deactivate base
conda deactivate
