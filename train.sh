#!/bin/bash

#SBATCH --job-name=data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -o slurm/%A.out

python3 main.py
exit 0