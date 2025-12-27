"""
MATH-500 Dataset Download and Preprocessing Script

This script downloads the MATH-500 dataset from HuggingFace and converts it
to a standardized JSON format for evaluation purposes.

Source: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

Usage:
    python -m evaluation.dataset_download.math
"""

import json
import os
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "Please install the 'datasets' library: pip install datasets"
    )


def download_math500_dataset() -> Any:
    """
    Download the MATH-500 dataset from HuggingFace.
    
    Returns:
        The loaded dataset object.
    """
    print("Downloading MATH-500 dataset from HuggingFace...")
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    print("Download completed.")
    return dataset


def process_dataset(dataset: Any) -> list[dict]:
    """
    Process the dataset into the required JSON format.
    
    Args:
        dataset: The HuggingFace dataset object.
        
    Returns:
        A list of dictionaries in the standardized format.
    """
    processed_data = []
    
    # The MATH-500 dataset typically has a 'test' split
    # Check available splits and use the appropriate one
    if "test" in dataset:
        data_split = dataset["test"]
    elif "train" in dataset:
        data_split = dataset["train"]
    else:
        # If no standard split, try to get the first available split
        available_splits = list(dataset.keys())
        if available_splits:
            data_split = dataset[available_splits[0]]
        else:
            raise ValueError("No data splits found in the dataset")
    
    print(f"Processing {len(data_split)} samples...")
    
    for item in data_split:
        processed_item = {
            "problem": item.get("problem", ""),
            "solution": item.get("solution", ""),
            "answer": item.get("answer", ""),
            "subject": item.get("subject", ""),
            "level": item.get("level", 0),
            "unique_id": item.get("unique_id", ""),
        }
        processed_data.append(processed_item)
    
    print(f"Processed {len(processed_data)} samples.")
    return processed_data


def save_to_json(data: list[dict], output_path: str) -> None:
    """
    Save the processed data to a JSON file.
    
    Args:
        data: The processed data to save.
        output_path: The path to save the JSON file.
    """
    # Create the output directory if it does not exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Data saved to {output_path}")


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        The path to the project root directory.
    """
    # Navigate up from the current file location to find the project root
    current_file = Path(__file__).resolve()
    # Go up two levels: dataset_download -> evaluation -> project_root
    project_root = current_file.parent.parent.parent
    return project_root


def main() -> None:
    """
    Main function to download and process the MATH-500 dataset.
    """
    print("=" * 60)
    print("MATH-500 Dataset Download and Preprocessing")
    print("=" * 60)
    
    # Define output path relative to project root
    project_root = get_project_root()
    output_path = project_root / "data" / "math.json"
    
    print(f"Project root: {project_root}")
    print(f"Output path: {output_path}")
    print("-" * 60)
    
    # Download the dataset
    dataset = download_math500_dataset()
    
    # Print dataset info
    print("-" * 60)
    print("Dataset info:")
    print(dataset)
    print("-" * 60)
    
    # Process the dataset
    processed_data = process_dataset(dataset)
    
    # Save to JSON file
    save_to_json(processed_data, str(output_path))
    
    # Print summary
    print("-" * 60)
    print("Summary:")
    print(f"  Total samples: {len(processed_data)}")
    if processed_data:
        subjects = set(item["subject"] for item in processed_data)
        levels = set(item["level"] for item in processed_data)
        print(f"  Unique subjects: {len(subjects)}")
        print(f"  Levels: {sorted(levels)}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()