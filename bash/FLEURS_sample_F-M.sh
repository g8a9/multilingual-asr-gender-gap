#!/bin/bash

#SBATCH --job-name=FLEURS_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000MB
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --account=attanasiog
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --array=0-18

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

DATASET="google/fleurs"
MODELS=( "openai/whisper-large-v3" "facebook/seamless-m4t-v2-large" )
LANGS=( "de" "en" "nl" "ru" "sr" "it" "fr" "es" "ca" "pt" "sw" "yo" "ja" "hu" "fi" "ro" "cs" "sk" "ar" )
TRANSCRIPTION_DIR="./results-interim-asr-performance-gap/transcriptions"
TRANSCRIPTION_DIR="./results-interim-asr-performance-gap/bs4_new/transcriptions"
RESULTS_DIR="./results-interim-asr-performance-gap/metrics_n1000_s40"
RESULTS_DIR="./results-interim-asr-performance-gap/bs4_new/metrics_n1000_s40"
LANG=${LANGS[${SLURM_ARRAY_TASK_ID}]}

# for SPLIT in "train" "validation" "test" "all"; do
for SPLIT in "devtest"; do
    for MODEL in ${MODELS[@]}; do

        # if [ "$SPLIT" == "all" ]; then
        #     do_sampling="True"
        # else
        #     do_sampling="False"
        # fi
        do_sampling="True"
        echo $do_sampling

        echo "STARTING ${LANG}"
        python ./src/2_sample_and_compute_metrics_binary.py \
            --lang ${LANG} \
            --transcription_dir ${TRANSCRIPTION_DIR} \
            --model ${MODEL} \
            --dataset ${DATASET} \
            --output_dir ${RESULTS_DIR} \
            --target_col "gender" --minority_group "female" --majority_group "male" \
            --split ${SPLIT} \
            --num_proc 7 \
            --load_type "remote" \
            --reference_col "raw_transcription" \
            --n_iterations 1000 \
            --minority_accept_percentile 100 \
            --fleurs_speaker_info_dir "./results-interim-asr-performance-gap/dataset_statistics/fleurs" \
            --apply_sampling_minority "False" \
            --apply_sampling_majority "False" \
            --overwrite_results \
            --do_sampling ${do_sampling}

        echo "DONE ${LANG}"
    done
done
