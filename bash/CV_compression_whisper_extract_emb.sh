#!/bin/bash

#SBATCH --job-name=compression_CV
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=300000MB
#SBATCH --partition=gpu
#SBATCH --array=14-18
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --account=attanasiog

export TOKENIZERS_PARALLELISM=true

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

DATASET="mozilla-foundation/common_voice_16_0"
MODELS=( "openai/whisper-large-v3" ) #"facebook/seamless-m4t-v2-large" )
LANGS=( "de" "en" "ca" "es" "fr" "nl" "ru" "sr" "it" "pt" "sw" "yo" "ja" "hu" "fi" "ro" "cs" "sk" "ar" )

OUTPUT_DIR="/data/milanlp/attanasiog/fair_asr/compression_female_male/"
mkdir -p ${OUTPUT_DIR}

LANG=${LANGS[${SLURM_ARRAY_TASK_ID}]}
MINORITY_GROUP="other"
MINORITY_GROUP="female"
MAJORITY_GROUP="male"

for MODEL in ${MODELS[@]}; do

    python src/3_compression_extract_embeddings.py \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --lang ${LANG} \
        --output_dir ${OUTPUT_DIR} \
        --load_type "remote" \
        --num_workers 7 \
        --subsample_n 5000 \
        --batch_size 1 \
        --overwrite_output \
        --minority_group ${MINORITY_GROUP} \
        --majority_group ${MAJORITY_GROUP}
done
