from typing import Optional
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from datasets import load_dataset

from .system_prompts import SYSTEM_PROMPT

# reward function from https://huggingface.co/datasets/burtenshaw/lora-without-regrets/blob/main/grpo.py
def strip_reasoning_accuracy_reward(
    completions: list[list[dict[str, str]]], solution: list[str], **kwargs
) -> list[Optional[float]]:
    """Reward function that strips reasoning tags and checks mathematical accuracy.
    This function:
    1. Extracts the content from completions
    2. Removes <think></think> tags (for reasoning that shouldn't be evaluated)
    3. Parses both the gold solution and the predicted answer
    4. Uses math_verify to check if they are mathematically equivalent
    Args:
        completions: List of model completions, each containing a list of messages
        solution: List of ground truth solutions
        **kwargs: Additional arguments (ignored but required for trainer compatibility)
    Returns:
        List of rewards where:
        - 1.0 if the answer is correct
        - 0.0 if the answer is incorrect
        - None if the solution is not parseable (skips this example)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # Strip reasoning tags from completion
        while "<think>" in content and "</think>" in content:
            start = content.find("<think>")
            end = content.find("</think>", start)
            if start != -1 and end != -1:
                content = content[:start] + content[end + len("</think>") :]
            else:
                break

        # Parse gold solution
        gold_parsed = parse(
            f"${sol}$",
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0, try_extract_without_anchor=True
                )
            ],
        )

        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        boxed_match_priority=0,
                        normalization_config=NormalizationConfig(
                            basic_latex=True,
                            units=True,
                            malformed_operators=False,
                            nits=False,
                            boxed=True,
                        ),
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(
                    f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}"
                )
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None

        rewards.append(reward)

    return rewards

def load_openr1_dataset(dataset_name_or_path: str, example_numbers: int = None):
    dataset = load_dataset(
        dataset_name_or_path, split="train"
    )
    
    def make_conversation(example):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": example["problem"]}
        ]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    if example_numbers is not None and len(dataset) > example_numbers:
        dataset = dataset.select(range(example_numbers))

    return {
        "train_dataset": dataset,
        "test_dataset": None,
        "reward_functions": [strip_reasoning_accuracy_reward],
    }