#!/bin/bash
# 启动vLLM服务用于评测

CUDA_VISIBLE_DEVICES=0,1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --port 8001 \
    --tensor-parallel-size 2

CUDA_VISIBLE_DEVICES=2,3 vllm serve outputs/grpo_full_qwen2_5_3b_20251121_111716/checkpoint-1024 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --port 8002 \
    --tensor-parallel-size 2

