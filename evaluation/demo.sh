#!/bin/bash

MODEL_PATH="tencent/WeDLM-8B-Instruct"

OUTPUT_DIR="output/model"

bash evaluation/evaluation_base.sh \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --datasets "mmlu,arc_c,arc_e,hellaswag" \
    --wedlm_entropy_threshold None \
    --wedlm_window_size 4 \
    --wedlm_pos_penalty_factor 1 \
    --per_worker_batch_size 16 \
    --num_gpus 8