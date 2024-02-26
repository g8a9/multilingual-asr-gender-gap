#!/bin/bash

#SBATCH --account=attanasiog
#SBATCH --job-name=FLEURS_unique_users
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8000MB
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --output=./logs/slurm-%A.out
#SBAATCH --array=0-13

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

python ./src/fleurs/extract_unique_users.py