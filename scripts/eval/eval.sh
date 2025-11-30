#!/bin/bash

# 1. 定义模型列表 (可以是 HuggingFace ID，也可以是本地绝对路径)
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # 可以在这里加更多模型
)

# 2. 定义评测任务 (你提供的任务字符串)
TASKS="aime24_gpassk,math_500,aime2025@k=2,gpqa:diamond@k=2"

# 3. 定义配置
PORT=8001
GPU_ID=0
CONFIG_PATH="scripts/eval/eval_template.yaml" # 指向上面创建的 yaml

# --- 清理函数 (Ctrl+C 中断时触发) ---
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo ">>> 正在强制关闭 vLLM (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    exit
}
trap cleanup SIGINT SIGTERM

# --- 主循环 ---
for MODEL in "${MODELS[@]}"; do
    echo "=================================================="
    echo "正在评测模型: $MODEL"
    echo "=================================================="

    # 1. 启动 vLLM
    # 注意添加了 --served-model-name local-model
    # 这样无论实际加载什么模型，API 只要请求 "local-model" 都能通
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --served-model-name "local-model" \
        --max-model-len 32768 \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.9 \
        --port $PORT \
        --trust-remote-code &

    SERVER_PID=$!
    echo ">>> vLLM 已后台启动，PID: $SERVER_PID"

    # 2. 健康检查循环 (等待服务就绪)
    echo ">>> 等待服务就绪 (Port: $PORT)..."
    MAX_RETRIES=60 # 等待 5分钟 (60 * 5s)
    COUNT=0
    SERVER_READY=false

    while [ $COUNT -lt $MAX_RETRIES ]; do
        # 检查 vLLM 的 health 接口 (OpenAI 兼容接口通常在 /health 或 /v1/models)
        if curl -s http://localhost:$PORT/health > /dev/null; then
            SERVER_READY=true
            break
        fi
        
        # 检查进程是否还活着
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo ">>> 错误：vLLM 进程意外退出！"
            break
        fi

        sleep 5
        ((COUNT++))
        echo -n "."
    done
    echo ""

    if [ "$SERVER_READY" = true ]; then
        echo ">>> 服务已就绪，开始运行 LightEval..."
        
        # 3. 运行 LightEval
        # 使用你提供的命令，yaml 已经在上面准备好了
        lighteval endpoint litellm \
            "$CONFIG_PATH" \
            "$TASKS" \
            --output_dir "./results/${MODEL##*/}" # 自动根据模型名生成结果目录
        
        echo ">>> 模型 $MODEL 评测完成。"
    else
        echo ">>> 服务启动超时或失败，跳过此模型。"
    fi

    # 4. 停止 vLLM
    echo ">>> 关闭 vLLM 服务..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    
    # 5. 等待显存释放 (非常重要)
    echo ">>> 等待显存释放..."
    sleep 10
done

echo "所有评测任务结束。"
