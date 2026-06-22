#!/bin/bash
#SBATCH --job-name=mbert_ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/mbert-server-%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mbert

python src/experiments/downstream_tasks/main.py
