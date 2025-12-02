unset WANDB_DISABLED
OUTPUT_DIR=outputs/dapo_lora_plus_$(date +%Y%m%d_%H%M%S)
# OUTPUT_DIR=outputs/debug
LOG_FILE=${OUTPUT_DIR}/output.log

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --main_process_port 29500 \
    --config_file scripts/accelerate/ds_zero2_4gpu.yaml \
    run.py train \
    --config.common.seed 42 \
    --config.common.debug false \
    --config.model.model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --config.model.dtype "bfloat16" \
    --config.peft.use_peft true \
    --config.peft.type "lora_plus" \
    --config.peft.task_type "CAUSAL_LM" \
    --config.peft.r 32 \
    --config.peft.lora_alpha 64 \
    --config.peft.lora_dropout 0.05 \
    --config.peft.total_step 1000 \
    --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"]' \
    --config.training.learning_rate 1e-5 \
    --config.training.beta 0.0 \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.run_name "${OUTPUT_DIR}" \
    --config.training.remove_unused_columns false \
    --config.training.gradient_accumulation_steps 8 \
    --config.training.num_train_epochs 1 \
    --config.training.max_completion_length 16384 \
    --config.training.num_generations 8 \
    --config.training.warmup_ratio 0.0 \
    --config.training.max_prompt_length 512 \
    --config.training.logging_steps 1 \
    --config.training.per_device_train_batch_size 4 \
    --config.training.save_strategy "steps" \
    --config.training.save_steps 64 \
    --config.training.max_steps 1024 \
    --config.training.use_vllm true \
    --config.training.top_entropy_quantile 1.0 \
    --config.training.epsilon_high 0.28 \
    --config.training.lr_scheduler_type "constant" \
    --config.training.lr_scheduler_kwargs.min_lr_rate 0.1 \
    --config.training.vllm_mode "colocate" \
    --config.training.vllm_gpu_memory_utilization 0.4 \
    --config.training.use_liger_kernel false \
    --config.training.loss_type "dapo" \
    --config.training.report_to '["wandb"]' \
    --config.logging.trackio_space_id "Open-Tinker/Open-Tinker" \
    --config.logging.trackio_project "grpo-full-qwen3-4b" \
    --config.logging.wandb_project "grpo-full-qwen3-4b" \
    --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed" \
    --config.dataset.example_numbers 1000000000 \
    &> ${LOG_FILE}
