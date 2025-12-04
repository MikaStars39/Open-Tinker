#! /bin/bash

set -exo pipefail

export LD_LIBRARY_PATH="/root/miniconda3/envs/perl/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"
# export VLLM_LOGGING_LEVEL="DEBUG"

PROJECT_DIR="/root/project/perl"
BASE_MODEL_PATH="/mnt/shared-storage-user/p1-shared/Qwen/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET="aime2024@32,aime2025@32,amc2023@32,math500@4,minerva@4,hmmt2025@32"
# DATASET="hmmt2025@4"

# Follow JustRL --- link: https://github.com/thunlp/JustRL/blob/main/evals/gen_vllm.py#L28-L30
TEMPERATURE="0.7"
TOP_P="0.9"
MAX_NEW_TOKENS="31744"

function eval_model_with_adapter() {
  pkill -9 python;
  pkill -9 -f "vllm.entrypoints.openai.api_server";
  pkill -9 -f "VLLM::EngineCore";
  sleep 1;
  pkill -9 python;
  pkill -9 -f "vllm.entrypoints.openai.api_server";
  pkill -9 -f "VLLM::EngineCore";
  /root/miniconda3/envs/perl/bin/python "${PROJECT_DIR}/scripts/eval/eval.py" \
    --result-dir "$1" \
    --model "$2" \
    --adapter "$3" \
    --dataset "${DATASET}" \
    --serve-port 8000 \
    --dp-size 8 \
    --tp-size 1 \
    --gpu-memory-utilization 0.95 \
    --seed "42" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-num-request 1024 \
    --dtype "bfloat16" 2>&1 | tee "eval.log";
}

# eval_model_with_adapter \
#   "${PROJECT_DIR}/results/qwen3-4b-thinking-2507" \
#   "${BASE_MODEL_PATH}" \
#   ""

function download_and_eval_model_with_adapter() {
  mkdir -p "${PROJECT_DIR}/ckpts/$1/$2";

  hf download MikaStars39/PeRL \
      --repo-type model \
      --local-dir "${PROJECT_DIR}/ckpts/" \
      --include "$1/$2/*";

  ls -a "${PROJECT_DIR}/ckpts/$1/$2";

  eval_model_with_adapter \
      "${PROJECT_DIR}/results/$1___$2" \
      "${BASE_MODEL_PATH}" \
      "${PROJECT_DIR}/ckpts/$1/$2";
}

set +e
download_and_eval_model_with_adapter "dapo_dora_qwen2_5_1_5b_20251126_115730"      "checkpoint-1024"
download_and_eval_model_with_adapter "dapo_ia3_qwen2_5_1_5b_20251128_120647"       "checkpoint-1024"
download_and_eval_model_with_adapter "dapo_layernorm_qwen2_5_1_5b_20251127_195534" "checkpoint-1024"
download_and_eval_model_with_adapter "dapo_lora_qwen_1_5b"                         "checkpoint-1024"
download_and_eval_model_with_adapter "dapo_milora_qwen2_5_1_5b_20251126_224006"    "checkpoint-1024"
download_and_eval_model_with_adapter "dapo_miss_qwen2_5_1_5b_20251124_220354"      "checkpoint-1024"
download_and_eval_model_with_adapter "dapo_pissa_qwen2_5_1_5b_20251126_192154"     "checkpoint-1024"
download_and_eval_model_with_adapter "dapo_vera_qwen2_5_1_5b_20251126_190555"      "checkpoint-1024"
