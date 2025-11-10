
def apply_adalora(model, args):
    from peft import LoraConfig, AdaLoraModel, AdaLoraConfig
    config = AdaLoraConfig(
        peft_type="ADALORA", 
        task_type=args.peft.task_type,
        init_r=args.peft.init_r,
        lora_alpha=args.peft.lora_alpha, 
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    )
    return get_peft_model(model, config)

def apply_lora(model, args):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        peft_type="LORA",
        task_type=args.peft.task_type,
        use_dora=args.peft.use_dora,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    )
    return get_peft_model(model, config)

def apply_vera(model, args):
    from peft import VeraConfig, get_peft_model
    config = VeraConfig(r=args.peft.r)
    return get_peft_model(model, config)

def apply_pissa(model, args):
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        # init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
        init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        lora_dropout=args.peft.lora_dropout, # Since the component of the PiSSA adapter are the principal singular values and vectors, dropout should be set to 0 to avoid random discarding.
        target_modules=args.peft.target_modules,
        task_type=args.peft.task_type,
    )
    return get_peft_model(model, lora_config)