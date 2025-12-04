import argparse
import pandas as pd
import sys
import os
import numpy as np

# Ensure we can import utils from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import utils


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate parquet file with utils.grade_answer_verl"
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="Path to the input parquet file"
    )
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return

    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Check for required columns
    required_columns = ["data_source", "reward_model", "responses"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns in input file: {missing_columns}")
        return

    # Aggregate results
    results = {}  # data_source -> {total_score: 0.0, count: 0}

    print(f"Processing {len(df)} rows from {input_file}...")

    for index, row in df.iterrows():
        data_source = row["data_source"]
        reward_model = row["reward_model"]
        responses = row["responses"]

        # Extract ground_truth from reward_model
        ground_truth = None
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth")
        elif hasattr(
            reward_model, "ground_truth"
        ):  # Handle object-like access if necessary
            ground_truth = reward_model.ground_truth

        # If ground_truth is still None or not a string, we might need to inspect data structure more closely
        # But assuming dict for now as per typical pandas parquet struct reading

        if ground_truth is None:
            # Skip if no ground truth
            continue

        # Ensure ground_truth is string
        ground_truth = str(ground_truth)

        # Handle responses list
        if not isinstance(responses, (list, np.ndarray)):
            # If it's a single value, wrap it
            responses = [responses]
        elif isinstance(responses, np.ndarray):
            responses = responses.tolist()

        for given_answer in responses:
            if given_answer is None:
                continue

            given_answer_str = str(given_answer)

            # Grade the answer
            # grade_answer_verl(given_answer: str, ground_truth: str) -> bool
            try:
                is_correct = utils.grade_answer_verl(given_answer_str, ground_truth)
            except Exception as e:
                print(f"Error grading row {index}: {e}")
                is_correct = False

            score = 1.0 if is_correct else 0.0

            if data_source not in results:
                results[data_source] = {"total_score": 0.0, "count": 0}

            results[data_source]["total_score"] += score
            results[data_source]["count"] += 1

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results by Data Source")
    print("=" * 50)
    print(f"{'Data Source':<30} | {'Average Score':<15} | {'Count':<10}")
    print("-" * 61)

    for data_source in sorted(results.keys()):
        stats = results[data_source]
        if stats["count"] > 0:
            avg_score = stats["total_score"] / stats["count"]
        else:
            avg_score = 0.0

        print(f"{data_source:<30} | {avg_score:<15.4f} | {stats['count']:<10}")
    print("=" * 50)


if __name__ == "__main__":
    main()
