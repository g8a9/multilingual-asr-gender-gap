#!/bin/bash

#SBATCH --account=attanasiog
#SBATCH --job-name=compr_CV_metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem=32000MB
#SBATCH --partition=compute
#SBATCH --array=0-18
#SBATCH --output=./logs/slurm-%A_%a.out

export TOKENIZERS_PARALLELISM=true

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

DATASET="mozilla-foundation/common_voice_16_0"
MODELS=( "openai/whisper-large-v3" ) #"facebook/seamless-m4t-v2-large" )
LANGS=( "de" "en" "ca" "es" "fr" "nl" "ru" "sr" "it" "pt" "sw" "yo" "ja" "hu" "fi" "ro" "cs" "sk" "ar" )

OUTPUT_DIR="./results-interim-asr-performance-gap/gender_probing_female_male"
EMB_DIR="/data/milanlp/attanasiog/fair_asr/compression_female_male"
mkdir -p ${OUTPUT_DIR}

LANG=${LANGS[${SLURM_ARRAY_TASK_ID}]}

for MODEL in ${MODELS[@]}; do

    python src/3_compression_probe_gender.py \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --embedding_dir ${EMB_DIR} \
        --lang ${LANG} \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 11 \
        --probe_type "mdl" \
        --random_shift_labels "False"

done
        # --load_type "local" \
        # --overwrite_output
        # --subsample_frac 0.3 \
