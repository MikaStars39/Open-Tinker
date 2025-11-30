import json
from typing import List, Dict, Any

def compute_pass_at_k(results_path: str) -> float:
    """
    Compute pass@k metric from a JSON file of evaluation results.
    
    A problem is considered passed if at least one of the k responses
    for that prompt hash is correct (exact match with target).
    
    Args:
        results_path: Path to JSON file containing evaluation results
        k: Number of responses to consider for each problem
        
    Returns:
        pass@k score as a float between 0 and 1
    """
    # Read JSONL file (one JSON object per line)
    data = []
    with open(results_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    # Group results by prompt hash
    prompt_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        prompt_hash = item.get('prompt_hash')
        if prompt_hash not in prompt_groups:
            prompt_groups[prompt_hash] = []
        prompt_groups[prompt_hash].append(item)
    
    total_problems = len(prompt_groups)
    passed_problems = 0
    
    for prompt_hash, group in prompt_groups.items():
        # Take up to k responses for this prompt
        responses = group
        
        # Check if any response is correct
        for resp in responses:
            if resp.get('exact_match', 0) == 1:
                passed_problems += 1
                break
    
    print(f"Total problems: {total_problems}")
    print(f"Passed problems: {passed_problems}")
    
    return passed_problems / total_problems if total_problems > 0 else 0.0

if __name__ == "__main__":
    results_path = "outputs/results_deepseek_aime24/checkpoint-64/outputs__grpo_full_qwen2_5_3b_20251121_111716__checkpoint-64/samples_aime24_2025-11-26T03-21-46.705601.jsonl"
    pass_at_k = compute_pass_at_k(results_path)
    print(f"Pass@k: {pass_at_k}")