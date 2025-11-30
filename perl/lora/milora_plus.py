import torch
import numpy as np
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_lora_layer_plus(weights, rank, mode="min"):
    """
    MiLoRA++ Initialization:
    1. Extract Directions (V) from SVD to align with specific subspaces.
    2. Discard singular values (S) to avoid vanishing gradients or PiSSA instability.
    3. Initialize A with orthonormal directions (Unit Scale).
    4. Initialize B with Zeros (Stable Start).
    5. Return delta=0 (Do NOT modify base model).
    """
    # 1. Decompose
    # Note: torch.linalg.svd returns U, S, Vh (where Vh is V transpose)
    U, S, Vh = torch.linalg.svd(weights.float(), full_matrices=False)
    
    # 2. Select Directions (Subspace)
    if mode == "min":
        # Off-Principal Directions: The last 'rank' rows of Vh
        # Corresponds to smallest singular values (RLVR Preferred)
        V_select = Vh[-rank:, :] 
    elif mode == "mid":
        mid_start = (len(S) - rank) // 2
        mid_end = mid_start + rank
        V_select = Vh[mid_start:mid_end, :]
    elif mode == "max":
        # Principal Directions: The first 'rank' rows (PiSSA style direction, but safer init)
        V_select = Vh[:rank, :]
    elif mode == "random":
        indices = np.random.choice(len(S), rank, replace=False)
        indices = np.sort(indices)
        V_select = Vh[indices, :]
    else:
        raise ValueError("Unknown mode!")

    # 3. Initialize A (Direction Only)
    # We use the raw orthonormal vectors. 
    # This provides a strong, clean signal flow into the subspace 
    # without the "magnitude trap" of multiplying by tiny S.
    lora_A_init = V_select 

    # 4. Initialize B (Zero)
    # Ensures identity mapping at initialization (W_new = W_base + 0)
    lora_B_init = torch.zeros((weights.shape[0], rank), device=weights.device)
    
    # 5. Delta (Zero)
    # We do NOT subtract anything from the base model in MiLoRA++
    delta = torch.zeros_like(weights)

    return lora_A_init, lora_B_init, delta


def add_svd_initialized_lora(model, 
                           rank=64, 
                           mode="min", 
                           hyper_param_type="LLM-Adapters",
                           device=None):
    """
    Add MiLoRA++ (SVD-Direction-Initialized) LoRA adapters
    """
    
    # Set config based on hyperparameter type
    if hyper_param_type == "LLM-Adapters":
        lora_alpha = rank # Alpha=Rank is standard for "Unit Scale" A matrices
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
    
    logger.info(f"Starting MiLoRA++ (Direction-Only) initialization...")
    logger.info(f"Mode: {mode} | Rank: {rank} | Target: Off-Principal Subspace")
    start_time = time.time()
    
    # SVD initialization
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'base_layer') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                module_type = type(module.base_layer).__name__
                if module_type == 'Linear' and any(proj in name for proj in target_modules):
                    try:
                        base_weight = module.base_layer.weight.data
                        
                        # --- USE NEW INITIALIZATION ---
                        lora_A_init, lora_B_init, _ = initialize_lora_layer_plus(
                            base_weight.float(), rank, mode=mode
                        )
                        # ------------------------------
                        
                        # Note: We do NOT subtract delta from base_layer in MiLoRA++
                        # module.base_layer.weight.data -= delta (SKIPPED)
                        
                        # Assign specialized weights
                        module.lora_A['default'].weight.data = lora_A_init.to(base_weight.device).to(base_weight.dtype)
                        module.lora_B['default'].weight.data = lora_B_init.to(base_weight.device).to(base_weight.dtype)
                        
                        logger.info(f"Processed: {name}")
                    except Exception as e:
                        logger.warning(f"Error processing {name}: {e}")
    
    end_time = time.time()
    logger.info(f"MiLoRA++ initialization completed in {end_time - start_time:.2f} seconds")
    
    return model

# Usage Example
if __name__ == "__main__":
    # Load model
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    # Apply MiLoRA++
    # mode="min" ensures we target the Off-Principal directions (RLVR native)
    model = add_svd_initialized_lora(
        model=model,
        rank=64,
        mode="min", 
        hyper_param_type="LLM-Adapters"
    )
    
    print("Model ready for RLVR training.")
