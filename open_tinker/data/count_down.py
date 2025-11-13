
import os
import random
import re

from datasets import load_dataset

# from https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb
# simple count down dataset

def format_reward_func(completions, **kwargs):
    """
    格式奖励函数，检查模型输出格式是否匹配: <think>...</think><answer>...</answer>

    参数:
        completions (list[str]): 生成的输出
    返回:
        list[float]: 奖励分数
    """
    # 初始化奖励列表
    rewards = []
    # 遍历生成的输出
    for completion in completions:
        try:
            # 在生成的输出前添加<think>标签，便于后续正则表达式匹配
            completion = "<think>" + completion

            if random.random() < 0.1:  # 1% 的概率将生成输出写入文件
                # 创建生成输出目录（如果不存在）
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)  # 写入生成的输出

            # 定义正则表达式模式，用于匹配 <think> 和 <answer> 标签
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)  # 使用正则表达式进行匹配

            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)  # 如果格式不正确，奖励为 0
            else:
                rewards.append(1.0)  # 如果格式正确，奖励为 1
        except Exception:
            rewards.append(0.0)  # 如果发生异常，奖励为 0

    return rewards

def equation_reward_func(completions, target, nums, **kwargs):
    """
    方程奖励函数，检查计算结果是否正确，数字是否符合使用要求（每个数字只用一次，只使用所提供的数字）

    参数:
        completions (list[str]): 生成的输出
        target (list[str]): 预期的答案
        nums (list[str]): 可用的数字

    返回:
        list[float]: 奖励分数
    """
    # 初始化奖励列表
    rewards = []
    # 遍历生成的输出、预期的答案和可用的数字
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # 在生成的输出前添加 <think> 标签，便于后续正则表达式匹配
            completion = "<think>" + completion
            # 定义正则表达式模式，用于匹配 <answer> 标签
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)  # 如果没有匹配到 <answer> 标签，奖励为 0
                continue
            equation = match.group(1).strip()  # 提取 <answer> 标签中的内容
            # 提取方程中的所有数字
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # 检查所有数字是否被使用且只使用一次
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue

            # 定义允许的字符模式，只允许数字、运算符、括号和空白字符
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)  # 如果方程包含不允许的字符，奖励为 0
                continue

            # 计算方程的结果
            result = eval(equation, {"__builtins__": None}, {})
            # 检查方程是否正确且与预期答案匹配（误差小于 1e-5）
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)  # 如果正确，奖励为 1

                # 10% 的概率将成功的样本写入文件
                if random.random() < 0.10:
                    # 创建生成输出目录（如果不存在）
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "completion_samples", "success_completion_samples.txt"
                    )
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(completion)  # 写入生成的输出
            else:
                rewards.append(0.0)  # 如果不正确，奖励为 0
        except Exception:
            rewards.append(0.0)  # 如果评估失败，奖励为 0

    return rewards

def thought_len_reward_func(completions, **kwargs):
    """
    思考长度奖励函数，检查 <think> 标签的长度是否大于 1000

    参数:
        completions (list[str]): 生成的输出
    返回:
        list[float]: 奖励分数
    """
    # 初始化奖励列表
    rewards = []
    # 遍历生成的输出
    for completion in completions:
        try:
            # 在生成的输出前添加 <think> 标签，便于后续正则表达式匹配
            completion = "<think>" + completion
            # 定义正则表达式模式，用于匹配 <think> 标签
            match = re.search(r"<think>(.*?)</think>", completion)
            # 如果匹配到 <think> 标签
            if match:
                thought_process = match.group(1).strip()  # 提取 <think> 标签中的内容
                thought_length = len(thought_process)  # 计算思考过程的长度
                if thought_length > 1000:
                    rewards.append(1.0)  # 如果思考过程长度大于 1000，奖励为 1
                else:
                    rewards.append(0.0)  # 否则奖励为 0
            else:
                rewards.append(0.0)  # 如果没有匹配到 <think> 标签，奖励为 0
                continue
        except Exception:
            rewards.append(0.0)  # 如果发生异常，奖励为 0

    return rewards


def load_count_down_dataset(
    dataset_name_or_path: str, 
    example_numbers: int = None,
    tokenizer = None
):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # Load dataset from Hugging Face Hub
    dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset = load_dataset(dataset_id, split="train")
    # select a random subset of 50k samples
    dataset = dataset.shuffle(seed=42)

    if example_numbers is not None and len(dataset) > example_numbers:
        dataset = dataset.select(range(example_numbers))

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
        { 
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }]
        return {
                "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": target
            }

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "reward_functions": [format_reward_func, equation_reward_func, thought_len_reward_func]
    }

if __name__ == "__main__":
    correct_sample_1 = """We need to find an equation using the numbers 19, 36, 55, and 7
    exactly once, with basic arithmetic operations, that equals 65. One possible
    combination is 55 + 36 - 19 + 7... </think>
    <answer> 55 + 36 - 7 - 19 </answer>"""

    correct_sample_2 = """ ... </think>
    <answer> 55 + 36 - 7 - 19 </answer>"""

    wrong_format = """User: Using the numbers [19, 36, 55, 7], create an equation that equals 65."""

    wrong_format_2 = """To find the equation that equals 79 using the numbers 95, 78, 6, 88, I'll start by adding 88 and 95:                      
    95 + 88 = 183                                                                                                              
    Now, let's subtract 104 from 183 to get 79:
    183 - 104 = 79
    <think> 183 - 104 = 79 </think><think> 183 - 104 = 79 </think><answer> 183 - 104 = 79 </answer>"""

    wrong_result = """ ... </think>
    <answer> 55 + 36 - 7 - 18 </answer>"""

    test_rewards = format_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result], target=["65", "65", "65", "65", "65"], nums=[[19, 36, 55, 7]] * 5)
    assert test_rewards == [1.0, 1.0, 0.0, 0.0, 1.0], "Reward function is not working"
    test_rewards = equation_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result], target=["65", "65", "65", "65", "65"], nums=[[19, 36, 55, 7]] * 5)
    assert test_rewards == [1.0, 1.0, 0.0, 0.0, 0.0], "Reward function is not working"