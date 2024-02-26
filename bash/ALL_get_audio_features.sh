#!/bin/bash

#SBATCH --job-name=audio_feat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256000MB
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --account=attanasiog
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --array=0-18

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

# DATASET="facebook/voxpopuli"
# DATASET="google/fleurs"
DATASET="mozilla-foundation/common_voice_16_0"
LANGS=( "de" "en" "nl" "ru" "sr" "it" "fr" "es" "ca" "pt" "sw" "yo" "ja" "hu" "fi" "ro" "cs" "sk" "ar" )

LANG=${LANGS[${SLURM_ARRAY_TASK_ID}]}

python ./src/4_extract_acoustic_features.py \
    --lang ${LANG} \
    --dataset ${DATASET} \
    --output_dir "./results-interim-asr-performance-gap/audio_features/" \
    --reference_col "sentence" \
    --num_workers 2

# for FLEURS
# --reference_col "raw_transcription" \
# for VP
# --reference_col "raw_text"