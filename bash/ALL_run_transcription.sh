#!/bin/bash

#SBATCH --job-name=bs4_ASR
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=gpu
#SBATCH --array=0-149
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --account=attanasiog

export TOKENIZERS_PARALLELISM=false

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

OUTPUT_DIR="./results-interim-asr-performance-gap/transcriptions"
CONFIG_FILE="./configs/all_transcription_config.json"
mkdir -p ${OUTPUT_DIR}

python src/transcribe_dataset.py \
    --config_file ${CONFIG_FILE} \
    --config_id ${SLURM_ARRAY_TASK_ID} \
    --output_dir ${OUTPUT_DIR} \
    --num_workers 7 \
    --batch_size 2 \
    --enable_chunk_decoding \
    --chunk_size 5000
