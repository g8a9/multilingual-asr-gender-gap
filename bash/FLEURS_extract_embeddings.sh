#!/bin/bash

#SBATCH --job-name=fleurs_count_users
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000MB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=attanasiog
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --array=14-18%4

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

LANGS=( "de" "en" "nl" "ru" "sr" "it" "fr" "es" "ca" "pt" "sw" "yo" "ja" "hu" "fi" "ro" "cs" "sk" "ar" )

LANG=${LANGS[${SLURM_ARRAY_TASK_ID}]}

echo "STARTING ${LANG}"
python ./src/fleurs/extract_embeddings.py --lang ${LANG}
echo "DONE ${LANG}"