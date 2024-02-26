#!/bin/bash

#SBATCH --account=attanasiog
#SBATCH --job-name=CV_sample_F-M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128000MB
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --array=0-18

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

DATASET="mozilla-foundation/common_voice_16_0"
MODELS=( "openai/whisper-large-v3" "facebook/seamless-m4t-v2-large" )
LANGS=( "de" "en" "nl" "ru" "sr" "it" "fr" "es" "ca" "pt" "sw" "yo" "ja" "hu" "fi" "ro" "cs" "sk" "ar" )
# LANGS=( "en" "it" "ca" "pt" "sw" "ja" )
TRANSCRIPTION_DIR="./results-interim-asr-performance-gap/bs4_new/transcriptions"
RESULTS_DIR="./results-interim-asr-performance-gap/bs4_new/metrics_n1000_s40"
LANG=${LANGS[${SLURM_ARRAY_TASK_ID}]}

for SPLIT in "devtest"; do
    for MODEL in ${MODELS[@]}; do

        do_sampling="True"
        echo "do sampling: $do_sampling"

        echo "STARTING ${LANG}"

        python ./src/2_sample_and_compute_metrics_binary.py \
            --lang ${LANG} \
            --transcription_dir ${TRANSCRIPTION_DIR} \
            --model ${MODEL} \
            --dataset ${DATASET} \
            --output_dir ${RESULTS_DIR} \
            --target_col "gender" --minority_group "female" --majority_group "male" \
            --split ${SPLIT} \
            --num_proc 11 \
            --apply_sampling_minority "False" \
            --apply_sampling_majority "False" \
            --overwrite_results \
            --do_sampling ${do_sampling}

        echo "DONE ${LANG}"
    done
done
