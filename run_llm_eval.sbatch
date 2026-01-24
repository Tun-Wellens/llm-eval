#!/usr/bin/bash --login
#SBATCH --job-name=llm-eval
#SBATCH --output=llm-eval-%j.out
#SBATCH --error=llm-eval-%j.err

# --- Resources ---
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=2-00:00:00

# --- Partition / QoS ---
#SBATCH --partition=gpu
#SBATCH --qos=normal

set -e

cd llm-eval
uv run python -u main.py "$1"

