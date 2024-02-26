#!/bin/bash

#SBATCH --job-name=vad
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32000MB
#SBATCH --partition=gpu
#SBAATCH --array=0-10%4
#SBATCH --array=0-18%1
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --account=attanasiog


export TOKENIZERS_PARALLELISM=true

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

RESULTS_DIR="./results-interim-asr-performance-gap/dataset_statistics"
# DATASET="google/fleurs"
# DATASET="facebook/voxpopuli"
DATASET="mozilla-foundation/common_voice_16_1"
LANGS=( "de" "en" "nl" "ru" "sr" "it" "fr" "es" "ca" "pt" "sw" "yo" "ja" "hu" "fi" "ro" "cs" "sk" "ar" )
# LANGS=( "fi" "ro" "cs" "sk" "ar" )
# LANGS=( "de" "en" "nl" "it" "fr" "es" "hu" "fi" "ro" "cs" "sk" )
LANG=${LANGS[${SLURM_ARRAY_TASK_ID}]}

python src/voice_activity_detection.py \
    --output_dir ${RESULTS_DIR} \
    --dataset_name ${DATASET} \
    --lang ${LANG} \
    --split "all" \
    --overwrite_output \
    --num_workers 7