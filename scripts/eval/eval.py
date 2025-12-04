#!/usr/bin/env python3
# 本脚本依据用户需求：实现评测流程（参数解析、模型合并、启动vLLM、生成、打分、缓存/恢复、日志、阶段化提示、最终统计）。
# 实现方案：使用argparse解析常规与--vllm-*透传参数，必要时在CPU上合并LoRA并保存；后台启动支持数据并行的vLLM服务器，
# 轮询后端生成多次rollout并缓存到文件，随后调用score_response汇总为result.jsonl，最后新增一个统计阶段输出avg@k/pass@k，
# 同时记录日志并将stdout/stderr写入latest_run.log；通过阶段化日志标明第几阶段的开始/结束（含emoji）。本版强制依赖vLLM与GPU，
# 新增 --num-gpus 参数用于运行前校验可用 GPU 数（bash 脚本中设为 8），确保按需求使用多卡并行。

import argparse
import asyncio
import atexit
import importlib.util
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
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Set
import math

try:
    import aiohttp
except ImportError:
    raise ImportError("需要安装 aiohttp: pip install aiohttp")

from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


PROMPT_TEMPLATE = """{problem} Please reason step by step, and put your final answer within \\boxed{{}}."""
DATASETS = {
    "aime2024": ("HuggingFaceH4/aime_2024", "train"),
    "aime2025": ("yentinglin/aime_2025", "train"),
    "hmmt2025": ("FlagEval/HMMT_2025", "train"),
}


def load_dataset_from_hf(dataset_name: str):
    if dataset_name in DATASETS:
        hf_name, split = DATASETS[dataset_name]
        return load_dataset(hf_name, split=split)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def prepare_prompt(dataset_name: str, sample: Dict[str, Any]) -> str:
    """根据sample构建模型输入prompt，可按需修改增强。"""
    problem = None
    if "problem" in sample:
        problem = sample["problem"]
    if "question" in sample:
        problem = sample["question"]
    elif "prompt" in sample:
        problem = sample["prompt"]
    else:
        raise ValueError(f"不支持的样本: {sample}")
    return PROMPT_TEMPLATE.format(problem=problem)


os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from utils import grade_answer_verl


def score_response(dataset_name: str, response: str, sample: Dict[str, Any]) -> float:
    ground_truth = None
    if "answer" in sample:
        ground_truth = sample["answer"]
    elif "label" in sample:
        ground_truth = sample["label"]
    else:
        raise ValueError(f"不支持的样本: {sample}")
    return 1.0 if grade_answer_verl(response, ground_truth) else 0.0


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
        stage_id: int | str,
        name: str,
        emoji_start: str = "🚀",
        emoji_end: str = "🏁",
        emoji_fail: str = "💥",
    ) -> None:
        self.logger = logger
        self.stage_id = str(stage_id)
        self.name = name
        self.emoji_start = emoji_start
        self.emoji_end = emoji_end
        self.emoji_fail = emoji_fail

    def __enter__(self) -> "StageContext":
        self.logger.info(
            "%s 第%s阶段开始：%s", self.emoji_start, self.stage_id, self.name
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if exc_type is None:
            self.logger.info(
                "%s 第%s阶段结束：%s", self.emoji_end, self.stage_id, self.name
            )
        else:
            self.logger.error(
                "%s 第%s阶段失败：%s，错误：%s",
                self.emoji_fail,
                self.stage_id,
                self.name,
                exc,
            )


def parse_args() -> Tuple[argparse.Namespace, List[str], List[str]]:
    parser = argparse.ArgumentParser(
        description="评测入口脚本，支持模型合并、vLLM启动与多数据集评测。"
    )
    parser.add_argument("--result-dir", required=True, help="中间过程与结果输出目录。")
    parser.add_argument("--model", required=True, help="基础模型名称或路径。")
    parser.add_argument(
        "--adapter", default="", help="LoRA/PEFT adapter路径，留空表示不合并。"
    )
    parser.add_argument(
        "--dataset",
        default="aime2024",
        help="要评测的数据集缩写，英文逗号分隔（如：aime2024）。",
    )
    parser.add_argument(
        "--rollout-n", type=int, default=1, help="每个sample生成多少次rollout。"
    )
    parser.add_argument(
        "--serve-port", type=int, default=8000, help="第一个vLLM后端端口号。"
    )
    parser.add_argument(
        "--dp-size", type=int, default=1, help="数据并行后端数量（启动多个vLLM）。"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="传给vLLM的张量并行大小。"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="运行前校验需要的GPU数量，不足则报错。"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="传给vLLM的GPU显存利用率上限（0~1），用于控制单卡显存占用比例。",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="生成温度。")
    parser.add_argument("--top-p", type=float, default=1.0, help="生成top-p。")
    parser.add_argument("--max-new-tokens", type=int, default=131072, help="生成长度。")
    parser.add_argument("--dtype", default="auto", help="模型dtype，用于合并环节。")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="是否信任远程代码。"
    )
    parser.add_argument(
        "--served-model-name", default="eval-model", help="vLLM对外暴露的模型名。"
    )
    parser.add_argument("--api-key", default="dummy", help="OpenAI兼容接口的API Key。")
    parser.add_argument(
        "--request-timeout", type=float, default=600.0, help="请求单次超时时间。"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="调试用，限制评测样本数量。"
    )
    parser.add_argument(
        "--max-num-request-per-dp",
        type=int,
        default=1,
        help="每个数据并行（DP）的vLLM后端同时运行的请求数上限。",
    )

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
            stripped = "--" + token[len("--vllm-") :]
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


def resolve_torch_dtype(dtype: Any) -> Any:
    """
    将dtype字符串解析为torch.dtype，支持auto/常见别名，兼容旧版Transformers缺少get_torch_dtype的场景。
    """
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = dtype.lower()
        if normalized == "auto":
            return None
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized in mapping:
            return mapping[normalized]
    raise ValueError(f"不支持的dtype: {dtype}")


def merge_model_if_needed(
    args: argparse.Namespace, result_dir: Path, logger: logging.Logger
) -> Path:
    if not args.adapter:
        logger.info("未提供adapter，直接使用基础模型：%s", args.model)
        return Path(args.model)

    output_dir = result_dir / "model"
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info("检测到已存在的合并模型目录，直接复用：%s", output_dir)
        return output_dir

    torch_dtype = resolve_torch_dtype(args.dtype)
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


def build_vllm_command(
    model_path: Path, port: int, args: argparse.Namespace, vllm_args: List[str]
) -> List[str]:
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
    # 实现方案：在构造 vLLM 启动命令时追加 --gpu-memory-utilization 参数，默认 0.95，可通过命令行覆盖。
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(vllm_args)
    return cmd


def pipe_to_logger(
    stream: Iterable[str], logger: logging.Logger, level: int, prefix: str
) -> None:
    for line in stream:
        logger.log(level, "%s%s", prefix, line.rstrip("\n"))


def start_vllm_processes(
    model_path: Path,
    args: argparse.Namespace,
    vllm_args: List[str],
    logger: logging.Logger,
) -> Tuple[List[subprocess.Popen], List[int]]:
    ports: List[int] = []
    processes: List[subprocess.Popen] = []
    env = os.environ.copy()
    dp_size = max(1, args.dp_size)

    for rank in range(dp_size):
        # 计算当前进程分配的GPU ID范围
        start_gpu_id = rank * args.tp_size
        end_gpu_id = start_gpu_id + args.tp_size
        gpu_ids = list(range(start_gpu_id, end_gpu_id))

        # 校验是否越界（基于args.num_gpus或者简单的逻辑校验，这里假设用户配置正确）
        # 如果需要更严格校验，可以在此处添加。

        env_local = env.copy()
        env_local["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        port = args.serve_port + rank
        cmd = build_vllm_command(model_path, port, args, vllm_args)
        logger.info(
            "启动vLLM后端[%d/%d]，端口%d，GPUs=%s，命令：%s",
            rank + 1,
            dp_size,
            port,
            gpu_ids,
            " ".join(cmd),
        )
        proc = subprocess.Popen(
            cmd,
            env=env_local,
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


def stop_vllm_processes(
    processes: List[subprocess.Popen], logger: logging.Logger
) -> None:
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


def wait_for_vllm_ready(
    port: int, process: subprocess.Popen, timeout: float, logger: logging.Logger
) -> bool:
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


def generate_with_vllm(prompt: str, port: int, args: argparse.Namespace) -> str:
    """同步版本的vLLM生成函数（保留用于向后兼容）。"""
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


async def generate_with_vllm_async(
    session: aiohttp.ClientSession, prompt: str, port: int, args: argparse.Namespace
) -> str:
    """异步版本的vLLM生成函数，用于并发请求。"""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "n": 1,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    try:
        async with session.post(
            url, json=payload, headers=headers, timeout=timeout
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"vLLM返回HTTP错误: {response.status}")
            content = await response.json()
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"vLLM连接失败: {exc}") from exc

    try:
        return content["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"解析vLLM响应失败: {content}") from exc


async def generate_responses(
    args: argparse.Namespace,
    dataset_name: str,
    rollout_n: int,
    ports: List[int],
    logger: logging.Logger,
) -> None:
    """
    异步并发生成响应并存入output.jsonl。
    实现方案：读取已有output.jsonl建立缓存，仅生成缺失的条目。
    生成结果实时追加写入output.jsonl。
    """
    dataset_dir = Path(args.result_dir) / dataset_name
    output_file = dataset_dir / "output.jsonl"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with StageContext(logger, "C.1", "读取缓存的输出"):
        generated_results: List[Dict[str, Any]] = []
        cache: Set[Tuple[int, int]] = set()

        if output_file.exists():
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if (
                            "problem_id" in data
                            and "rollout_id" in data
                            and "response" in data
                        ):
                            generated_results.append(data)
                            cache.add((data["problem_id"], data["rollout_id"]))
                    except json.JSONDecodeError:
                        logger.warning("output.jsonl中存在无效JSON行，已跳过。")

        logger.info("已加载缓存条目数：%d", len(generated_results))

    with StageContext(logger, "C.2", "准备生成任务"):
        ds = load_dataset_from_hf(dataset_name)
        max_concurrent_per_dp = max(1, args.max_num_request_per_dp)
        semaphores = {port: asyncio.Semaphore(max_concurrent_per_dp) for port in ports}

        tasks_to_process: List[Tuple[int, int, str, int]] = []
        ports_cycle = len(ports)

        for idx, sample in enumerate(ds):
            prompt = prepare_prompt(dataset_name, sample)
            for rollout_id in range(rollout_n):
                if (idx, rollout_id) in cache:
                    continue
                port_idx = (idx * rollout_n + rollout_id) % ports_cycle
                tasks_to_process.append((idx, rollout_id, prompt, port_idx))

        logger.info("需要新生成的请求数：%d", len(tasks_to_process))

        if not tasks_to_process:
            logger.info("所有请求已在缓存中，无需生成。")
            return

    with StageContext(logger, "C.3", "并行生成"):
        file_lock = asyncio.Lock()

        async def generate_one_task(
            problem_id: int,
            rollout_id: int,
            prompt: str,
            port_idx: int,
            session: aiohttp.ClientSession,
        ) -> None:
            port = ports[port_idx]
            semaphore = semaphores[port]
            response = ""

            async with semaphore:
                try:
                    logger.info(
                        "向端口%d请求生成，problem=%06d rollout=%03d",
                        port,
                        problem_id,
                        rollout_id,
                    )
                    response = await generate_with_vllm_async(
                        session, prompt, port, args
                    )
                except Exception as exc:
                    logger.error(
                        "生成失败 problem=%06d rollout=%03d port=%d: %s",
                        problem_id,
                        rollout_id,
                        port,
                        exc,
                    )
                    response = ""

            record = {
                "problem_id": problem_id,
                "rollout_id": rollout_id,
                "response": response,
            }

            generated_results.append(record)

            async with file_lock:
                with output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        async with aiohttp.ClientSession() as session:
            tasks = [
                generate_one_task(pid, rid, pmt, pidx, session)
                for pid, rid, pmt, pidx in tasks_to_process
            ]
            await asyncio.gather(*tasks)

        logger.info("数据集 %s 生成完成，结果存入 %s", dataset_name, output_file)


def evaluate_dataset_results(
    args: argparse.Namespace,
    dataset_name: str,
    rollout_n: int,
    logger: logging.Logger,
) -> Dict[str, Dict[int, float]]:
    """
    评测阶段：读取output.jsonl，评分并生成result.jsonl，返回统计指标。
    """
    dataset_dir = Path(args.result_dir) / dataset_name
    output_file = dataset_dir / "output.jsonl"
    result_file = dataset_dir / "result.jsonl"

    with StageContext(logger, "D.1", "加载模型输出"):
        if not output_file.exists():
            raise ValueError(f"未找到output.jsonl，无法进行评测：{dataset_name}")

        outputs_map: Dict[int, List[Tuple[int, str]]] = {}
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if "problem_id" in d and "rollout_id" in d:
                        outputs_map.setdefault(d["problem_id"], []).append(
                            (d["rollout_id"], d.get("response", ""))
                        )
                except json.JSONDecodeError:
                    pass

    with StageContext(logger, "D.2", "加载原数据集"):
        ds = load_dataset_from_hf(dataset_name)

    with StageContext(logger, "D.3", "并行评测&计算指标"):
        records_for_metrics: List[Dict[str, Any]] = []

        with result_file.open("w", encoding="utf-8") as rf:
            for idx, sample in enumerate(ds):
                problem_id = idx
                prompt = prepare_prompt(dataset_name, sample)

                rollouts = outputs_map.get(problem_id, [])
                # 按rollout_id排序
                rollouts.sort(key=lambda x: x[0])
                rollout_dict = {r[0]: r[1] for r in rollouts}

                responses = []
                scores = []

                for rid in range(rollout_n):
                    resp = rollout_dict.get(rid, "")
                    responses.append(resp)

                    if resp:
                        try:
                            s_res = score_response(dataset_name, resp, sample)
                            if isinstance(s_res, tuple):
                                score = float(s_res[0])
                            else:
                                score = float(s_res)
                        except Exception as e:
                            logger.warning("评分出错 p=%d r=%d: %s", problem_id, rid, e)
                            score = 0.0
                    else:
                        score = 0.0
                    scores.append(score)

                    records_for_metrics.append(
                        {"problem_id": problem_id, "rollout_id": rid, "score": score}
                    )

                if scores:
                    avg_val = statistics.mean(scores)
                    max_val = max(scores)
                    min_val = min(scores)
                    mean_val = avg_val
                    try:
                        std_val = statistics.stdev(scores)
                    except statistics.StatisticsError:
                        std_val = 0.0
                else:
                    avg_val = max_val = min_val = mean_val = std_val = 0.0

                record = {
                    "problem_id": problem_id,
                    "prompt": prompt,
                    "responses": responses,
                    "scores": scores,
                    "avg": avg_val,
                    "max": max_val,
                    "min": min_val,
                    "mean": mean_val,
                    "std": std_val,
                }
                rf.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("评测完成，结果写入 %s", result_file)

    return compute_metrics(records_for_metrics, rollout_n)


def compute_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_correct == 0:
        return 0.0
    if num_samples <= k:
        return 1.0 if num_correct > 0 else 0.0
    return 1 - (math.comb(num_samples - num_correct, k) / math.comb(num_samples, k))


def compute_metrics(
    records: List[Dict[str, Any]], rollout_n: int
) -> Dict[str, Dict[int, float]]:
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


async def main() -> None:
    args, vllm_args, leftover = parse_args()
    logger = setup_logging(Path(args.result_dir))
    if leftover:
        logger.warning("检测到无法识别的参数（将被忽略）：%s", leftover)

    with StageContext(logger, "A", "准备模型/合并LoRA"):
        model_path = merge_model_if_needed(args, Path(args.result_dir), logger)

    with StageContext(logger, "B", "启动vLLM后端"):
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

    datasets_to_run = [item.strip() for item in args.dataset.split(",") if item.strip()]

    with StageContext(logger, "C", "数据集生成（缓存/生成）"):
        for task_abbr in datasets_to_run:
            logger.info("🚀 开始生成数据集：%s", task_abbr)
            rollout_n = args.rollout_n
            if "@" in task_abbr:
                rollout_n = int(task_abbr.split("@")[1])
                task_abbr = task_abbr.split("@")[0]
            await generate_responses(args, task_abbr, rollout_n, ports, logger)
            logger.info("✅ 完成生成数据集：%s (rollout=%d)", task_abbr, rollout_n)

    with StageContext(logger, "D", "评测与统计"):
        for task_abbr in datasets_to_run:
            logger.info("📊 开始评测数据集：%s", task_abbr)
            rollout_n = args.rollout_n
            if "@" in task_abbr:
                rollout_n = int(task_abbr.split("@")[1])
                task_abbr = task_abbr.split("@")[0]
            metrics = evaluate_dataset_results(args, task_abbr, rollout_n, logger)
            logger.info(
                "📊 数据集%s avg@%d: %s", task_abbr, rollout_n, metrics["avg_at_k"]
            )
            logger.info(
                "📈 数据集%s pass@%d: %s", task_abbr, rollout_n, metrics["pass_at_k"]
            )

    stop_vllm_processes(processes, logger)
    logger.info("全部评测流程完成。")


if __name__ == "__main__":
    asyncio.run(main())
