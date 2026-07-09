#!/bin/bash
#SBATCH --job-name=structml_aspect4
#SBATCH --output=logs/experiments_%j.out
#SBATCH --error=logs/experiments_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=all

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate structml1

# Run from the directory this script lives in (portable across users).
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs

echo "Starting Aspect 4 run at $(date)"
python preprocess.py       # builds column-feature graphs for all datasets (skips if cached)
python run_experiments.py  # trains all depth/mitigation variants
echo "Done at $(date)"
