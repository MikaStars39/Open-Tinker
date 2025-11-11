CUDA_VISIBLE_DEVICES=0,1 ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --config_file config/accelerate/ds_zero2.yaml \
    run.py \
    --config_path config/open_r1_example.toml \
    --output_dir outputs/grpo-lora-qwen3-8b