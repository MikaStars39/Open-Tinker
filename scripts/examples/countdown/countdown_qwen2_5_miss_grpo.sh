unset WANDB_DISABLED
OUTPUT_DIR=outputs/grpo_miss_qwen2_5_3b_$(date +%Y%m%d_%H%M%S)
# OUTPUT_DIR=outputs/debug
LOG_FILE=${OUTPUT_DIR}/output.log

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --main_process_port 29502 \
    --config_file scripts/accelerate/ds_zero2_2gpu.yaml \
    run.py train \
   --config.common.seed 42 \
    --config.common.debug false \
    --config.model.model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --config.model.dtype "bfloat16" \
    --config.peft.type "miss" \
    --config.peft.use_peft true \
    --config.peft.task_type "CAUSAL_LM" \
    --config.peft.r 32 \
    --config.peft.lora_alpha 128 \
    --config.peft.lora_dropout 0.05 \
    --config.peft.total_step 1000 \
    --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj"]' \
    --config.training.learning_rate 5e-6 \
    --config.training.beta 0.001 \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.run_name "${OUTPUT_DIR}" \
    --config.training.remove_unused_columns false \
    --config.training.gradient_accumulation_steps 16 \
    --config.training.num_train_epochs 1 \
    --config.training.max_completion_length 4096 \
    --config.training.num_generations 8 \
    --config.training.warmup_ratio 0.01 \
    --config.training.max_prompt_length 256 \
    --config.training.logging_steps 1 \
    --config.training.save_strategy "steps" \
    --config.training.save_steps 128 \
    --config.training.max_steps 450 \
    --config.training.use_vllm true \
    --config.training.lr_scheduler_type "linear" \
    --config.training.vllm_mode "colocate" \
    --config.training.vllm_gpu_memory_utilization 0.4 \
    --config.training.use_liger_kernel true \
    --config.training.loss_type "grpo" \
    --config.training.report_to '["wandb"]' \
    --config.logging.trackio_space_id "Open-Tinker/Open-Tinker" \
    --config.logging.trackio_project "grpo-lora-qwen2-5-3b" \
    --config.logging.wandb_project "grpo-lora-qwen2-5-3b" \
    --config.dataset.dataset_name_or_path "Jiayi-Pan/Countdown-Tasks-3to4" \
    --config.dataset.example_numbers 1000000000 \
    &> ${LOG_FILE}
