#!/bin/bash
#SBATCH --job-name=structml_experiments
#SBATCH --output=logs/experiments_%j.out
#SBATCH --error=logs/experiments_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=all

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate structml1

cd /home/zeev.kliot/structML_project/aspect1
mkdir -p logs

echo "Starting experiment run at $(date)"
/home/zeev.kliot/miniconda3/envs/structml1/bin/python run_experiments.py
echo "Done at $(date)"
