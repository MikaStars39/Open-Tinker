## Env Settings

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