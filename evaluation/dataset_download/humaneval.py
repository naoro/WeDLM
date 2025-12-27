"""
HumanEval Dataset Download and Preprocessing Script

This script downloads the OpenAI HumanEval dataset from HuggingFace
and saves it in JSONL format for evaluation purposes.

Usage:
    python -m evaluation.dataset_download.humaneval

Source:
    https://huggingface.co/datasets/openai/openai_humaneval
"""

import json
import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from this file's location to find project root
    current_file = Path(__file__).resolve()
    # evaluation/dataset_download/humaneval.py -> project root
    return current_file.parent.parent.parent


def download_humaneval_dataset():
    """
    Download HumanEval dataset from HuggingFace.
    
    Returns:
        Dataset object containing HumanEval data
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' library: pip install datasets"
        )
    
    print("Downloading HumanEval dataset from HuggingFace...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    print(f"Downloaded {len(dataset)} examples")
    
    return dataset


def process_and_save(dataset, output_path: Path):
    """
    Process the dataset and save to JSONL format.
    
    Args:
        dataset: HuggingFace dataset object
        output_path: Path to save the JSONL file
    
    Each line in the output file contains:
        - task_id: Unique identifier (e.g., "HumanEval/0")
        - prompt: Function signature and docstring
        - entry_point: Name of the function to be implemented
        - canonical_solution: Reference solution
        - test: Test cases for verification
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing and saving to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, example in enumerate(dataset):
            # Extract required fields
            record = {
                "task_id": example["task_id"],
                "prompt": example["prompt"],
                "entry_point": example["entry_point"],
                "canonical_solution": example["canonical_solution"],
                "test": example["test"],
            }
            
            # Write as JSON line (ensure ASCII to avoid any non-English characters)
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + "\n")
    
    print(f"Successfully saved {idx + 1} examples to {output_path}")


def verify_output(output_path: Path, num_samples: int = 3):
    """
    Verify the output file by reading and displaying sample entries.
    
    Args:
        output_path: Path to the JSONL file
        num_samples: Number of samples to display
    """
    print(f"\nVerifying output file...")
    print("-" * 60)
    
    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            record = json.loads(line.strip())
            print(f"\nSample {i + 1}:")
            print(f"  task_id: {record['task_id']}")
            print(f"  entry_point: {record['entry_point']}")
            print(f"  prompt length: {len(record['prompt'])} chars")
            print(f"  solution length: {len(record['canonical_solution'])} chars")
            print(f"  test length: {len(record['test'])} chars")
    
    # Count total lines
    with open(output_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    print("-" * 60)
    print(f"Total records in file: {total_lines}")


def main():
    """Main entry point for the script."""
    # Define output path
    project_root = get_project_root()
    output_path = project_root / "data" / "humaneval.jsonl"
    
    print("=" * 60)
    print("HumanEval Dataset Preprocessing")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Output path: {output_path}")
    print()
    
    # Download dataset
    dataset = download_humaneval_dataset()
    
    # Process and save
    process_and_save(dataset, output_path)
    
    # Verify output
    verify_output(output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()