#! /bin/bash

pkill -9 python
pkill -9 -f "vllm.entrypoints.openai.api_server"
pkill -9 -f "VLLM::EngineCore"
sleep 1
pkill -9 python
pkill -9 -f "vllm.entrypoints.openai.api_server"
pkill -9 -f "VLLM::EngineCore"

set -ex pipefail

export LD_LIBRARY_PATH="/root/miniconda3/envs/perl/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"
# export VLLM_LOGGING_LEVEL="DEBUG"

PROJECT_DIR="/root/project/perl"
DATASET="aime2024@32,aime2025@32,amc2023@32,math500@4,minerva@4,hmmt2025@32"
# DATASET="hmmt2025@4"

# Follow JustRL --- link: https://github.com/thunlp/JustRL/blob/main/evals/gen_vllm.py#L28-L30
TEMPERATURE="0.7"
TOP_P="0.9"
MAX_NEW_TOKENS="31744"

function eval_model_with_adapter() {
  /root/miniconda3/envs/perl/bin/python "${PROJECT_DIR}/scripts/eval/eval.py" \
    --result-dir "$1" \
    --model "$2" \
    --adapter "$3" \
    --dataset "${DATASET}" \
    --serve-port 8000 \
    --dp-size 8 \
    --tp-size 1 \
    --gpu-memory-utilization 0.95 \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-num-request 512 \
    --dtype "bfloat16" 2>&1 | tee "eval.log";
}

# eval_model_with_adapter \
#   "${PROJECT_DIR}/results/qwen3-4b-thinking-2507" \
#   "/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Thinking-2507" \
#   "" \

eval_model_with_adapter \
  "${PROJECT_DIR}/results/deepseek-r1-1.5b" \
  "/mnt/shared-storage-user/p1-shared/Qwen/DeepSeek-R1-Distill-Qwen-1.5B" \
  "" \