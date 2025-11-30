python debug.py

vllm serve outputs/grpo_full_qwen2_5_3b_20251121_111716/checkpoint-1024 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --port 8000