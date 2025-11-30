import torch
import numpy as np
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_lora_layer(weights, rank, mode="min"):
    """
    Initialize LoRA layer using SVD decomposition
    """
    U, S, V = torch.linalg.svd(weights, full_matrices=False)
    lora_alpha = rank
    
    if mode == "min":
        U_select = U[:, -rank:]
        S_select = S[-rank:]
        V_select = V[-rank:, :]
    elif mode == "mid":
        mid_start = (len(S) - rank) // 2
        mid_end = mid_start + rank
        U_select = U[:, mid_start:mid_end]
        S_select = S[mid_start:mid_end]
        V_select = V[mid_start:mid_end, :]
    elif mode == "max":
        U_select = U[:, :rank]
        S_select = S[:rank]
        V_select = V[:rank, :]
    elif mode == "random":
        indices = np.random.choice(len(S), rank, replace=False)
        indices = np.sort(indices)
        U_select = U[:, indices]
        S_select = S[indices]
        V_select = V[indices, :]
    else:
        raise ValueError("Unknown mode!")

    scaling = lora_alpha / rank
    S_select /= scaling
    S_sqrt = torch.sqrt(S_select)
    B = U_select @ torch.diag(S_sqrt)
    A = torch.diag(S_sqrt) @ V_select
    delta = scaling * B @ A

    return A, B, delta


def add_svd_initialized_lora(model, 
                           rank=64, 
                           mode="min", 
                           hyper_param_type="LLM-Adapters",
                           ):
    """
    Add SVD-initialized LoRA adapters to a HuggingFace model
    
    Args:
        model: HuggingFace pretrained model
        rank (int): LoRA rank size, default 64
        mode (str): SVD mode, options "min", "max", "mid", "random", default "min"
        hyper_param_type (str): hyperparameter type, options "LLM-Adapters", "QLoRA", default "LLM-Adapters"
        device (str): device, if None auto-detect
        
    Returns:
        model: model with SVD-initialized LoRA
    """
    
    # Set config based on hyperparameter type
    if hyper_param_type == "LLM-Adapters":
        lora_alpha = rank
        lora_dropout = 0.05
        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    elif hyper_param_type == "QLoRA":
        lora_alpha = rank
        lora_dropout = 0.1
        target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
    else:
        raise ValueError(f"Unknown hyper_param_type: {hyper_param_type}")
    
    # Create LoRA config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=rank,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    
    # Convert model to PEFT model
    model = get_peft_model(model, peft_config)
    
    logger.info(f"Starting SVD initialization of LoRA layers...")
    start_time = time.time()
    
    # SVD initialization - directly iterate over LoRA modules
    with torch.no_grad():
        for name, module in model.named_modules():
            # Check if this is a LoRA-wrapped module
            if hasattr(module, 'base_layer') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Check if this module is in our target modules
                module_type = type(module.base_layer).__name__
                if module_type == 'Linear' and any(proj in name for proj in target_modules):
                    try:
                        # Get the base layer weights
                        base_weight = module.base_layer.weight.data
                        
                        # Perform SVD initialization
                        lora_A_init, lora_B_init, delta = initialize_lora_layer(
                            base_weight.float(), rank, mode=mode
                        )
                        
                        # Update weights
                        module.base_layer.weight.data -= delta.to(base_weight.device).to(base_weight.dtype)
                        module.lora_A['default'].weight.data = lora_A_init.to(base_weight.device).to(base_weight.dtype)
                        module.lora_B['default'].weight.data = lora_B_init.to(base_weight.device).to(base_weight.dtype)
                        
                        logger.info(f"Processed: {name} (mode: {mode}, rank: {rank})")
                    except Exception as e:
                        logger.warning(f"Error processing {name}: {e}")
    
    end_time = time.time()
    logger.info(f"SVD initialization completed, total time: {end_time - start_time:.2f} seconds")
    
    return model


# Usage example
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    
    # Load original model
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # replace with your model path
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    
    # Add SVD-initialized LoRA
    model_with_lora = add_svd_initialized_lora(
        model=model,
        rank=64,
        mode="min",
        hyper_param_type="LLM-Adapters",
        device="cuda:0"
    )
    
    # Now model_with_lora has SVD-initialized LoRA adapters
    print("Model successfully added SVD-initialized LoRA adapters")