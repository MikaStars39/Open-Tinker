from dataclasses import dataclass, field
from typing import List

@dataclass
class CommonConfig:
    """Common configuration settings"""
    seed: int = 42
    debug: bool = False  # set to true to use debug mode w/ no logging to trackio or wandb


@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_name_or_path: str = None # keep none for cli reminder "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "bfloat16"


@dataclass
class PeftConfig:
    """PEFT configuration settings"""
    type: str = None # keep none for cli reminder "lora"
    use_peft: bool = True
    task_type: str = "CAUSAL_LM"
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    total_step: int = 1000  # for adalora
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
    )


@dataclass
class TrainingConfig:
    """Training configuration settings"""
    learning_rate: float = 1e-5
    output_dir: str = ""
    run_name: str = ""
    remove_unused_columns: bool = False
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 1
    max_completion_length: int = 1024
    num_generations: int = 8
    max_prompt_length: int = 128
    logging_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 128
    max_steps: int = 1024
    use_vllm: bool = True  # (1) full-ft w/ vllm (2) others do not use vllm and rollout w/ model.generate instead.
    # Mika Update: all vllm
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.4
    use_liger_kernel: bool = True
    lr_scheduler_type: str = "cosine"
    loss_type: str = "dr_grpo"  # default is dapo which does not support by liger kernel lol
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    beta: float = 0.0
    warmup_ratio: float = 0.0


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    trackio_space_id: str = "Open-Tinker/Open-Tinker"
    trackio_project: str = "grpo-lora-qwen2-5-3b"
    wandb_project: str = "grpo-lora-qwen2-5-3b"


@dataclass
class DatasetConfig:
    """Dataset configuration settings"""
    dataset_name_or_path: str = None # keep none for cli reminder "Jiayi-Pan/Countdown-Tasks-3to4"
    example_numbers: int = 1000000000  # use all examples


@dataclass
class TrainConfig:
    """Main training configuration"""
    common: CommonConfig = field(default_factory=CommonConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
