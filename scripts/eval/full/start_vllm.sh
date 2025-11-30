#!/bin/bash
# 启动vLLM服务用于评测
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --served-model-name "local-model" \
    --gpu-memory-utilization 0.9 \
    --port 8001

CUDA_VISIBLE_DEVICES=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --port 8002