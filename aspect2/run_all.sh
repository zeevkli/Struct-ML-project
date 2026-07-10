#!/bin/bash
#SBATCH --job-name=aspect2
#SBATCH --output=/home/zeev.kliot/structML_project/aspect2/logs/aspect2_%j.out
#SBATCH --error=/home/zeev.kliot/structML_project/aspect2/logs/aspect2_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=all

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate structml1

cd /home/zeev.kliot/structML_project/aspect2

echo "Starting at $(date)"
echo "=== Preprocessing ==="
python preprocess.py

echo "=== Training ==="
python -u run_experiments.py
echo "Done at $(date)"
