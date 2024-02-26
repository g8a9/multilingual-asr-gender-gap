#!/bin/bash

#SBATCH --job-name=compute_seconds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=300000MB
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --account=attanasiog
#SBAATCH --output=./logs/slurm-%A.out
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --array=0-18

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

# DATASET="google/fleurs"
# DATASET="facebook/voxpopuli"
DATASET="mozilla-foundation/common_voice_16_0"
LANGS=( "de" "en" "nl" "ru" "sr" "it" "fr" "es" "ca" "pt" "sw" "yo" "ja" "hu" "fi" "ro" "cs" "sk" "ar" )
# LANGS=( "de" "en" "nl" "it" "fr" "es" "hu" )
OUTPUT_DIR="results-interim-asr-performance-gap/dataset_statistics"


LANG=${LANGS[${SLURM_ARRAY_TASK_ID}]}

echo "$LANG"
python ./src/5_compute_duration.py \
    --dataset ${DATASET} \
    --lang ${LANG} \
    --output_dir ${OUTPUT_DIR} \
    --model "openai/whisper-large-v3" \
    --reference_col "sentence" \
    --num_workers 4
