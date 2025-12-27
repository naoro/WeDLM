#!/bin/bash

# --- Default Configuration ---
NUM_GPUS=8
GPU_MEMORY_UTILIZATION=0.9
MAX_NUM_SEQS=512
PER_WORKER_BATCH_SIZE=16
WeDLM_WINDOW_SIZE=16
MAX_MODEL_LEN=4096

# WeDLM sampling parameters (now in SamplingParams)
# Use "None" to disable parallel decoding and enable one-by-one generation mode
WeDLM_ENTROPY_THRESHOLD="0.4"
WeDLM_POS_PENALTY_FACTOR="0.02"

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Required:
  --model_path PATH            Full path to the HuggingFace model directory
  --output_dir PATH            Output directory for evaluation results

Optional:
  --datasets LIST              Dataset list (default: mbpp,humaneval,gsm8k)
  --per_worker_batch_size N    Batch size per worker (default: 16)
  --wedlm_window_size N         WeDLM sliding window size (default: 16)
  --wedlm_entropy_threshold F   Entropy threshold for parallel decoding (default: 0.4)
                               Use 'None' to disable parallel decoding (one-by-one mode)
  --wedlm_pos_penalty_factor F  Position penalty factor (default: 0.02)
  --num_gpus N                 Number of GPUs to use (default: 8)
  --gpu_memory_utilization F   GPU memory utilization 0.0-1.0 (default: 0.9)
  --max_num_seqs N             Max concurrent sequences (default: 512)
  --max_model_len N            Max context length (default: 4096)
  --enforce_eager              Disable CUDA graphs
  --help                       Show this help message
EOF
}

# ================= Argument Parsing =================

DATASETS="mbpp,humaneval,gsm8k"
ENFORCE_EAGER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        
        --per_worker_batch_size) PER_WORKER_BATCH_SIZE="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max_num_seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
        
        --wedlm_window_size) WeDLM_WINDOW_SIZE="$2"; shift 2 ;;
        --wedlm_entropy_threshold) WeDLM_ENTROPY_THRESHOLD="$2"; shift 2 ;;
        --wedlm_pos_penalty_factor) WeDLM_POS_PENALTY_FACTOR="$2"; shift 2 ;;
        
        --enforce_eager) ENFORCE_EAGER="--enforce-eager"; shift ;;
        
        --help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# ================= Validation =================

if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo -e "\033[31m[ERROR] Missing --model_path or --output_dir.\033[0m"
    show_help
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "\033[31m[ERROR] Model path not found: $MODEL_PATH\033[0m"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
IFS=',' read -ra BENCHMARKS <<< "$DATASETS"

# ================= Environment Setup =================

pip install evalplus

# ================= Main Evaluation Logic =================

for dataset in "${BENCHMARKS[@]}"; do
    echo -e "\033[32m[EVAL] Processing dataset: $dataset\033[0m"
    
    # Display entropy threshold info
    if [ "$WeDLM_ENTROPY_THRESHOLD" = "None" ] || [ "$WeDLM_ENTROPY_THRESHOLD" = "none" ]; then
        echo "[INFO] WeDLM entropy threshold: None (one-by-one generation mode)"
    else
        echo "[INFO] WeDLM entropy threshold: $WeDLM_ENTROPY_THRESHOLD"
    fi
    
    CMD="python -m evaluation.wedlm_eval \
        --model-path $MODEL_PATH \
        --dataset-name $dataset \
        --output-dir $OUTPUT_DIR \
        --num-gpus $NUM_GPUS \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --max-num-seqs $MAX_NUM_SEQS \
        --max-model-len $MAX_MODEL_LEN \
        --per-worker-batch-size $PER_WORKER_BATCH_SIZE \
        --wedlm-window-size $WeDLM_WINDOW_SIZE \
        --wedlm-entropy-threshold $WeDLM_ENTROPY_THRESHOLD \
        --wedlm-pos-penalty-factor $WeDLM_POS_PENALTY_FACTOR"

    # Add optional flags
    if [ -n "$ENFORCE_EAGER" ]; then
        CMD="$CMD $ENFORCE_EAGER"
    fi

    echo "[CMD] $CMD"
    eval $CMD

    if [ $? -eq 0 ]; then
        echo -e "\033[32m[SUCCESS] Finished $dataset\033[0m"
    else
        echo -e "\033[31m[FAIL] Failed $dataset\033[0m"
    fi
    sleep 2
done

echo ">>> All evaluations finished."