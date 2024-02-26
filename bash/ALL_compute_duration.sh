#!/bin/bash

#SBATCH --job-name=compute_seconds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=350000MB
#SBATCH --time=1:00:00
#SBATCH --partition=compute
#SBATCH --account=attanasiog
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --array=0

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

# DATASET="google/fleurs"
DATASET="mozilla-foundation/common_voice_16_0"
# DATASET="facebook/voxpopuli"
LANGS=( "de" "en" "nl" "ru" "sr" "it" "fr" "es" "ca" "pt" "sw" "yo" "ja" "hu" )
# LANGS=( "de" "en" "nl" "it" "fr" "es" "hu" )
OUTPUT_DIR="results-interim-asr-performance-gap/dataset_statistics"

LANG=${LANGS[${SLURM_ARRAY_TASK_ID}]}

echo "$LANG"
python ./src/5_compute_duration.py \
    --dataset ${DATASET} \
    --lang ${LANG} \
    --output_dir ${OUTPUT_DIR} \
    --num_workers 7
