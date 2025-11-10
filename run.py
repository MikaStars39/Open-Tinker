
import torch
import os
from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from fire import Fire

from open_tinker.utils.utils import (
    parse_toml_to_args, # toml to args
)
from open_tinker.utils.logging import init_logger, logger
from open_tinker.data import load_dataset

def fuzzy_jobs(
    config_path: str,
    output_dir: str
):
    args = parse_toml_to_args(config_path)
    init_logger()
    args.training.output_dir = output_dir
    if not os.path.exists(output_dir): # check if output_dir exists
        os.makedirs(output_dir)
    else:
        logger.info(f"Output directory {output_dir} already exists, using it")
    set_seed(args.common.seed)
    
    return args

def train(
    config_path: str,
    output_dir: str
):
    # 0. parse args and prepare logger
    args = fuzzy_jobs(config_path, output_dir)

    # 1. load dataset
    logger.info(f"Loading dataset from {args.dataset.dataset_name_or_path}")
    dataset = load_dataset(
        args.dataset.dataset_name_or_path,
        example_numbers=args.dataset.example_numbers
    )
    train_dataset = dataset["train_dataset"]
    test_dataset = dataset["test_dataset"]
    reward_functions = dataset["reward_functions"]

    # 2. load and configure model
    logger.info(f"Loading model from {args.model.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model.model_name_or_path, 
        torch_dtype= torch.bfloat16 if args.model.dtype == "bfloat16" else torch.float16, 
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    logger.info(f"Model loaded successfully")

    # 3. configure lora
    if args.peft.use_peft:
        logger.info(f"Detected PEFT configuration, configuring lora")
        lora_config = LoraConfig(
            task_type=args.peft.task_type,
            use_dora=args.peft.use_dora,
            r=args.peft.r,
            lora_alpha=args.peft.lora_alpha,
            lora_dropout=args.peft.lora_dropout,
            target_modules=args.peft.target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Lora configured successfully")
    
    model.print_trainable_parameters()

    # 4.Load and configure tokenizer for left padding
    logger.info(f"Loading tokenizer from {args.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    tokenizer.padding_side = "left"  # Configure for decoder-only architecture: use left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id


    # 5.Training configuration
    training_args = GRPOConfig(
        **vars(args.training),
    )

    # 6.Train
    logger.info(f"Training model with GRPO")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()
    logger.info(f"Training completed successfully")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    Fire(train)