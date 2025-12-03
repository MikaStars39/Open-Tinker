#!/usr/bin/env python3
# 本脚本依据用户需求：实现评测流程（参数解析、模型合并、启动vLLM、生成、打分、缓存/恢复、日志、阶段化提示、最终统计）。
# 实现方案：使用argparse解析常规与--vllm-*透传参数，必要时在CPU上合并LoRA并保存；后台启动支持数据并行的vLLM服务器，
# 轮询后端生成多次rollout并缓存到文件，随后调用score_response汇总为result.jsonl，最后新增一个统计阶段输出avg@k/pass@k，
# 同时记录日志并将stdout/stderr写入latest_run.log；通过阶段化日志标明第几阶段的开始/结束（含emoji）。

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import math

from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_torch_dtype


class StreamToLogger:
    """Redirect stdout/stderr到logger，确保输出被文件与控制台同时记录。"""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, buffer: str) -> None:
        self._buffer += buffer
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.logger.log(self.level, line)

    def flush(self) -> None:
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""


def setup_logging(result_dir: Path) -> logging.Logger:
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "latest_run.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    stdout_logger.propagate = True
    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    stderr_logger.propagate = True
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

    return logging.getLogger("eval_all")


class StageContext:
    """阶段化日志上下文，标记开始/结束和失败场景。"""

    def __init__(
        self,
        logger: logging.Logger,
        stage_id: int,
        name: str,
        emoji_start: str = "🚀",
        emoji_end: str = "🏁",
        emoji_fail: str = "💥",
    ) -> None:
        self.logger = logger
        self.stage_id = stage_id
        self.name = name
        self.emoji_start = emoji_start
        self.emoji_end = emoji_end
        self.emoji_fail = emoji_fail

    def __enter__(self) -> "StageContext":
        self.logger.info("%s 第%d阶段开始：%s", self.emoji_start, self.stage_id, self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if exc_type is None:
            self.logger.info("%s 第%d阶段结束：%s", self.emoji_end, self.stage_id, self.name)
        else:
            self.logger.error("%s 第%d阶段失败：%s，错误：%s", self.emoji_fail, self.stage_id, self.name, exc)


def parse_args() -> Tuple[argparse.Namespace, List[str], List[str]]:
    parser = argparse.ArgumentParser(description="评测入口脚本，支持模型合并、vLLM启动与多数据集评测。")
    parser.add_argument("--result-dir", required=True, help="中间过程与结果输出目录。")
    parser.add_argument("--model", required=True, help="基础模型名称或路径。")
    parser.add_argument("--adapter", default="", help="LoRA/PEFT adapter路径，留空表示不合并。")
    parser.add_argument("--dataset", default="HuggingFaceH4/aime_2024", help="要评测的数据集，英文逗号分隔。")
    parser.add_argument("--rollout-n", type=int, default=1, help="每个sample生成多少次rollout。")
    parser.add_argument("--serve-port", type=int, default=8000, help="第一个vLLM后端端口号。")
    parser.add_argument("--dp-size", type=int, default=1, help="数据并行后端数量（启动多个vLLM）。")
    parser.add_argument("--tp-size", type=int, default=1, help="传给vLLM的张量并行大小。")
    parser.add_argument("--temperature", type=float, default=1.0, help="生成温度。")
    parser.add_argument("--top-p", type=float, default=1.0, help="生成top-p。")
    parser.add_argument("--max-new-tokens", type=int, default=131072, help="生成长度。")
    parser.add_argument("--dtype", default="auto", help="模型dtype，用于合并环节。")
    parser.add_argument("--trust-remote-code", action="store_true", help="是否信任远程代码。")
    parser.add_argument("--served-model-name", default="eval-model", help="vLLM对外暴露的模型名。")
    parser.add_argument("--api-key", default="dummy", help="OpenAI兼容接口的API Key。")
    parser.add_argument("--request-timeout", type=float, default=600.0, help="请求单次超时时间。")
    parser.add_argument("--max-samples", type=int, default=None, help="调试用，限制评测样本数量。")

    args, unknown = parser.parse_known_args()
    vllm_args, leftover = extract_vllm_args(unknown)
    return args, vllm_args, leftover


def extract_vllm_args(unknown: List[str]) -> Tuple[List[str], List[str]]:
    vllm_args: List[str] = []
    leftover: List[str] = []
    idx = 0
    while idx < len(unknown):
        token = unknown[idx]
        if token.startswith("--vllm-"):
            stripped = "--" + token[len("--vllm-"):]
            if "=" in token:
                _, value = token.split("=", 1)
                vllm_args.extend([stripped, value])
            elif idx + 1 < len(unknown) and not unknown[idx + 1].startswith("-"):
                vllm_args.extend([stripped, unknown[idx + 1]])
                idx += 1
            else:
                vllm_args.append(stripped)
        else:
            leftover.append(token)
        idx += 1
    return vllm_args, leftover


def prepare_prompt(sample: Dict[str, Any]) -> str:
    """根据sample构建模型输入prompt，可按需修改增强。"""
    if isinstance(sample, dict):
        if "prompt" in sample:
            return str(sample["prompt"])
        if "instruction" in sample and "input" in sample:
            return f"{sample['instruction']}\n{sample['input']}"
        if "instruction" in sample:
            return str(sample["instruction"])
        if "question" in sample:
            return str(sample["question"])
        if "text" in sample:
            return str(sample["text"])
    return str(sample)


def score_response(prompt: str, response: str, sample: Dict[str, Any]) -> float:
    """简单占位评分：若sample包含answer/label且出现在response则记1，否则0。"""
    answer = None
    if isinstance(sample, dict):
        answer = sample.get("answer") or sample.get("label")
    if answer is None:
        return 0.0
    return float(str(answer) in response)


def merge_model_if_needed(args: argparse.Namespace, result_dir: Path, logger: logging.Logger) -> Path:
    if not args.adapter:
        logger.info("未提供adapter，直接使用基础模型：%s", args.model)
        return Path(args.model)

    output_dir = result_dir / "model"
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info("检测到已存在的合并模型目录，直接复用：%s", output_dir)
        return output_dir

    torch_dtype = get_torch_dtype(args.dtype)
    logger.info("加载基础模型：%s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("加载分词器：%s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("加载LoRA/PEFT adapter：%s", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    logger.info("执行merge_and_unload，将LoRA权重写入基础模型。")
    model = model.merge_and_unload()

    logger.info("保存合并模型至：%s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def build_vllm_command(model_path: Path, port: int, args: argparse.Namespace, vllm_args: List[str]) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        args.served_model_name,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(args.tp_size),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(vllm_args)
    return cmd


def pipe_to_logger(stream: Iterable[str], logger: logging.Logger, level: int, prefix: str) -> None:
    for line in stream:
        logger.log(level, "%s%s", prefix, line.rstrip("\n"))


def start_vllm_processes(
    model_path: Path, args: argparse.Namespace, vllm_args: List[str], logger: logging.Logger
) -> Tuple[List[subprocess.Popen], List[int]]:
    ports: List[int] = []
    processes: List[subprocess.Popen] = []
    env = os.environ.copy()

    for rank in range(max(1, args.dp_size)):
        port = args.serve_port + rank
        cmd = build_vllm_command(model_path, port, args, vllm_args)
        logger.info("启动vLLM后端[%d/%d]，端口%d，命令：%s", rank + 1, args.dp_size, port, " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        processes.append(proc)
        ports.append(port)
        if proc.stdout:
            threading.Thread(
                target=pipe_to_logger,
                args=(proc.stdout, logger, logging.INFO, f"[vllm:{port}] "),
                daemon=True,
            ).start()
        if proc.stderr:
            threading.Thread(
                target=pipe_to_logger,
                args=(proc.stderr, logger, logging.ERROR, f"[vllm:{port}] "),
                daemon=True,
            ).start()
    return processes, ports


def stop_vllm_processes(processes: List[subprocess.Popen], logger: logging.Logger) -> None:
    for proc in processes:
        if proc.poll() is None:
            try:
                logger.info("尝试终止vLLM进程(pid=%d)。", proc.pid)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as exc:  # noqa: BLE001
                logger.warning("终止进程(pid=%d)时发生异常：%s", proc.pid, exc)
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass


def wait_for_vllm_ready(port: int, process: subprocess.Popen, timeout: float, logger: logging.Logger) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            logger.error("vLLM进程(pid=%d)提前退出。", process.pid)
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("端口%d的vLLM已就绪。", port)
                    return True
        except Exception:
            time.sleep(2)
    logger.error("等待端口%d的vLLM超时。", port)
    return False


def load_dataset_by_name(name: str, split: str):
    if ":" in name:
        path, subset = name.split(":", 1)
        return load_dataset(path, subset, split=split)
    return load_dataset(name, split=split)


def generate_with_vllm(prompt: str, port: int, args: argparse.Namespace) -> str:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "n": 1,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
            body = response.read().decode("utf-8")
            content = json.loads(body)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"vLLM返回HTTP错误: {exc}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM连接失败: {exc}") from exc

    try:
        return content["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"解析vLLM响应失败: {content}") from exc


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def evaluate_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    ports: List[int],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    dataset_dir = Path(args.result_dir) / dataset_name
    outputs_dir = dataset_dir / "outputs"
    result_file = dataset_dir / "result.jsonl"

    if result_file.exists():
        logger.warning("检测到已存在的结果文件，跳过重新评测数据集 %s : %s", dataset_name, result_file)
        try:
            with result_file.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception as exc:  # noqa: BLE001
            logger.error("读取已有结果失败，将重新评测。错误：%s", exc)

    # 用户需求：删除 --dataset-split 参数，默认使用 test，不存在则通过 logger.warning 报错并回退到 train。
    split = "test"
    try:
        logger.info("加载数据集 %s split=%s", dataset_name, split)
        ds = load_dataset_by_name(dataset_name, split)
    except ValueError as exc:
        logger.warning(
            "数据集 %s 不存在 split=%s，将回退到 split=train。原始错误：%s",
            dataset_name,
            split,
            exc,
        )
        split = "train"
        logger.info("加载数据集 %s split=%s", dataset_name, split)
        ds = load_dataset_by_name(dataset_name, split)
    records: List[Dict[str, Any]] = []
    ports_cycle = len(ports)
    rollout_counter = 0

    for idx, sample in enumerate(ds):
        if args.max_samples is not None and idx >= args.max_samples:
            logger.info("命中max_samples=%d，提前结束。", args.max_samples)
            break
        prompt = prepare_prompt(sample)
        problem_dir = outputs_dir / f"{idx:06d}"
        for rollout_id in range(args.rollout_n):
            output_path = problem_dir / f"rollout_{rollout_id:03d}.txt"
            if output_path.exists() and output_path.stat().st_size > 0:
                response = output_path.read_text(encoding="utf-8")
                logger.info("复用缓存结果：%s", output_path)
            else:
                port = ports[rollout_counter % ports_cycle]
                rollout_counter += 1
                logger.info("向端口%d请求生成，problem=%06d rollout=%03d", port, idx, rollout_id)
                response = generate_with_vllm(prompt, port, args)
                save_text(output_path, response)
            score = score_response(prompt, response, sample)
            records.append(
                {
                    "problem_id": idx,
                    "rollout_id": rollout_id,
                    "prompt": prompt,
                    "response": response,
                    "score": score,
                }
            )

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("数据集 %s 评测完成，结果写入 %s", dataset_name, result_file)
    return records


def compute_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_correct == 0:
        return 0.0
    if num_samples <= k:
        return 1.0 if num_correct > 0 else 0.0
    return 1 - (math.comb(num_samples - num_correct, k) / math.comb(num_samples, k))


def compute_metrics(records: List[Dict[str, Any]], rollout_n: int) -> Dict[str, Dict[int, float]]:
    by_problem: Dict[int, List[float]] = {}
    for rec in records:
        by_problem.setdefault(int(rec["problem_id"]), []).append(float(rec["score"]))

    avg_at_k: Dict[int, float] = {}
    pass_at_k: Dict[int, float] = {}

    for k in range(1, rollout_n + 1):
        avg_scores = []
        pass_scores = []
        for scores in by_problem.values():
            sorted_scores = sorted(scores, reverse=True)
            topk = sorted_scores[:k]
            if topk:
                avg_scores.append(sum(topk) / len(topk))
            c = sum(1 for s in scores if s > 0)
            pass_scores.append(compute_pass_at_k(len(scores), c, k))

        avg_at_k[k] = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
        pass_at_k[k] = sum(pass_scores) / len(pass_scores) if pass_scores else 0.0

    return {"avg_at_k": avg_at_k, "pass_at_k": pass_at_k}


def main() -> None:
    args, vllm_args, leftover = parse_args()
    logger = setup_logging(Path(args.result_dir))
    if leftover:
        logger.warning("检测到无法识别的参数（将被忽略）：%s", leftover)

    with StageContext(logger, 1, "准备模型/合并LoRA"):
        model_path = merge_model_if_needed(args, Path(args.result_dir), logger)

    with StageContext(logger, 2, "启动vLLM后端"):
        processes, ports = start_vllm_processes(model_path, args, vllm_args, logger)
        atexit.register(stop_vllm_processes, processes, logger)

        def handle_signal(signum, frame):  # noqa: ANN001
            logger.warning("收到信号%d，准备清理后退出。", signum)
            stop_vllm_processes(processes, logger)
            sys.exit(1)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        for proc, port in zip(processes, ports):
            if not wait_for_vllm_ready(port, proc, timeout=300, logger=logger):
                stop_vllm_processes(processes, logger)
                sys.exit(1)

    all_records: Dict[str, List[Dict[str, Any]]] = {}
    datasets_to_run = [item.strip() for item in args.dataset.split(",") if item.strip()]
    with StageContext(logger, 3, "数据集评测与缓存/生成"):
        for name in datasets_to_run:
            logger.info("🧪 开始评测数据集：%s", name)
            records = evaluate_dataset(name, args, ports, logger)
            all_records[name] = records
            logger.info("✅ 完成评测数据集：%s", name)

    with StageContext(logger, 4, "统计阶段：计算avg@k与pass@k"):
        overall_records: List[Dict[str, Any]] = []
        for name, records in all_records.items():
            overall_records.extend(records)
            metrics = compute_metrics(records, args.rollout_n)
            logger.info("📊 数据集%s avg@k: %s", name, metrics["avg_at_k"])
            logger.info("📈 数据集%s pass@k: %s", name, metrics["pass_at_k"])

        overall_metrics = compute_metrics(overall_records, args.rollout_n) if overall_records else None
        if overall_metrics:
            logger.info("🌐 全部数据集合并 avg@k: %s", overall_metrics["avg_at_k"])
            logger.info("🌟 全部数据集合并 pass@k: %s", overall_metrics["pass_at_k"])
        else:
            logger.warning("未获取到任何记录，跳过全局统计。")

    stop_vllm_processes(processes, logger)
    logger.info("全部评测流程完成。")


if __name__ == "__main__":
    main()
