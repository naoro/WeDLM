"""
MBPP Dataset Download and Preprocessing Script

This script downloads the MBPP (Mostly Basic Python Problems) dataset 
from HuggingFace and saves it in JSONL format for evaluation purposes.

Usage:
    python -m evaluation.dataset_download.mbpp

Source:
    https://huggingface.co/datasets/google-research-datasets/mbpp
"""

import json
import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from this file's location to find project root
    current_file = Path(__file__).resolve()
    # evaluation/dataset_download/mbpp.py -> project root
    return current_file.parent.parent.parent


def download_mbpp_dataset():
    """
    Download MBPP dataset from HuggingFace.
    
    Returns:
        Dataset object containing MBPP data (all splits combined)
    """
    try:
        from datasets import load_dataset, concatenate_datasets
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' library: pip install datasets"
        )
    
    print("Downloading MBPP dataset from HuggingFace...")
    # Use the official google-research-datasets/mbpp dataset
    # "full" config contains all 974 examples with the complete fields
    dataset = load_dataset("mbpp", "full")
    
    # Combine all splits (train, validation, test, prompt)
    all_splits = []
    for split_name in dataset.keys():
        print(f"  - {split_name}: {len(dataset[split_name])} examples")
        all_splits.append(dataset[split_name])
    
    combined_dataset = concatenate_datasets(all_splits)
    print(f"Total downloaded: {len(combined_dataset)} examples")
    
    return combined_dataset


def process_and_save(dataset, output_path: Path):
    """
    Process the dataset and save to JSONL format.
    
    Args:
        dataset: HuggingFace dataset object
        output_path: Path to save the JSONL file
    
    Each line in the output file contains:
        - text: Problem description
        - code: Reference solution code
        - task_id: Unique identifier
        - test_setup_code: Setup code for tests (if any)
        - test_list: List of test assertions
        - challenge_test_list: Additional challenge tests (if any)
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing and saving to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, example in enumerate(dataset):
            # Extract and map fields to desired format
            record = {
                "text": example["text"],
                "code": example["code"],
                "task_id": example["task_id"],
                "test_setup_code": example.get("test_setup_code", ""),
                "test_list": example["test_list"],
                "challenge_test_list": example.get("challenge_test_list", []),
            }
            
            # Write as JSON line (ensure ASCII to avoid encoding issues)
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
            print(f"  text: {record['text'][:80]}...")
            print(f"  code length: {len(record['code'])} chars")
            print(f"  test_list count: {len(record['test_list'])}")
            print(f"  challenge_test_list count: {len(record['challenge_test_list'])}")
    
    # Count total lines
    with open(output_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    print("-" * 60)
    print(f"Total records in file: {total_lines}")


def main():
    """Main entry point for the script."""
    # Define output path
    project_root = get_project_root()
    output_path = project_root / "data" / "mbpp.jsonl"
    
    print("=" * 60)
    print("MBPP Dataset Preprocessing")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Output path: {output_path}")
    print()
    
    # Download dataset
    dataset = download_mbpp_dataset()
    
    # Process and save
    process_and_save(dataset, output_path)
    
    # Verify output
    verify_output(output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()