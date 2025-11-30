# PERL: Parameter-Efficient Reinforcement Learning  
> A minimal, modular, and lightning-fast framework for fine-tuning language models with PEFT + RL.

---

## 🧩 Supported Parameter-Efficient Methods

| Method        | Status | Notes |
|---------------|--------|-------|
| LoRA          | ✅     | Fully tested |
| DoRA          | ✅     | Weight-decomposed LoRA |
| MiSS          | ✅     | Mixture of Sub-Spaces |
| VeRA          | ✅     | Vector-based Random Adaptation |
| PiSSA         | ✅     | Principal Singular values & Singular vectors Adaptation |
| AdaLoRA       | ❌     | Rank allocation unstable under RL |
| X-LoRA        | 🔄     | Cross-layer routing |
| QLoRA         | 🔄     | Kronecker-product adaptation |
| MiLoRA        | 🔄     | Kronecker-product adaptation |


> Full list & references: [Awesome-LoRA](https://github.com/Yuheng2000/Awesome-LoRA)

---

## ⚙️ Environment Setup

```
pip install -r requirements.txt
pip install vllm --no-build-isolation # vllm for trl rollout
```

### Flash Attention

```
uv pip install flash-attn --no-cache-dir --no-build-isolation
python -c "import flash_attn" # verify
```

### Liger-Kernel for faster training

```
pip install liger-kernel --no-build-isolation
```

## Benchmark

Math: aime24, aime25, math500, GPQA diamond, amc23

Code: 


|  Task  |Version|  Metric  |Value |   |Stderr|
|--------|-------|----------|-----:|---|-----:|
|aime24:0|       |pass@k:k=1|0.3667|±  |0.0895|
|        |       |avg@n:n=1 |0.3667|±  |0.0895|
|all     |       |pass@k:k=1|0.3667|±  |0.0895|
|        |       |avg@n:n=1 |0.3667|±  |0.0895|

|  Task  |Version|  Metric  |Value |   |Stderr|
|--------|-------|----------|-----:|---|-----:|
|aime24:0|       |pass@k:k=1|0.4667|±  |0.0926|
|        |       |avg@n:n=1 |0.4667|±  |0.0926|
|all     |       |pass@k:k=1|0.4667|±  |0.0926|
|        |       |avg@n:n=1 |0.4667|±  |0.0926|
