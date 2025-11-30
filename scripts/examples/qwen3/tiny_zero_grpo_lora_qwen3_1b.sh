OUTPUT_DIR=outputs/grpo_lora_qwen3_4b_$(date +%Y%m%d_%H%M%S)
# OUTPUT_DIR=outputs/debug
LOG_FILE=${OUTPUT_DIR}/output.log

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --config_file config/accelerate/ds_zero2_4gpu.yaml \
    run.py \
    --config_path config/qwen3/count_down_grpo_lora_qwen2_5_3b.toml \
    --output_dir ${OUTPUT_DIR} &> ${LOG_FILE}