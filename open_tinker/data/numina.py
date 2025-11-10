from math_verify import LatexExtractionConfig, parse, verify
from datasets import load_dataset

from .system_prompts import SYSTEM_PROMPT

# Load and prepare dataset
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


# Reward functions
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    return [1.0 if re.match(pattern, content) else 0.0 for content in completion_contents]

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

def load_numina_dataset(dataset_name_or_path: str, example_numbers: int = None):

    train_dataset, test_dataset = load_dataset(dataset_name_or_path, split=['train[:5%]', 'test[:5%]'])
        
    if example_numbers is not None and len(dataset) > example_numbers:
        train_dataset = train_dataset.select(range(example_numbers))
        test_dataset = test_dataset.select(range(example_numbers))

    train_dataset = train_dataset.map(make_conversation).remove_columns(['messages', 'problem'])
    test_dataset = test_dataset.map(make_conversation)

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "reward_functions": [format_reward, accuracy_reward],
    }