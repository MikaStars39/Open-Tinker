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
MODEL_PATH="/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B-Thinking-2507"
ADAPTER_PATH=""
RESULT_DIR="${PROJECT_DIR}/results/qwen3-4b-thinking-2507"
DATASET="HuggingFaceH4/aime_2024"

/root/miniconda3/envs/perl/bin/python "${PROJECT_DIR}/scripts/eval/eval.py" \
  --result-dir "${RESULT_DIR}" \
  --model "${MODEL_PATH}" \
  --adapter "${ADAPTER_PATH}" \
  --dataset "${DATASET}" \
  --rollout-n 64 \
  --serve-port 8000 \
  --dp-size 8 \
  --tp-size 1 \
  --gpu-memory-utilization 0.95 \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-new-tokens 131072 \
  --max-num-request-per-dp 16 \
  --dtype "bfloat16" 2>&1 | tee "eval.log"