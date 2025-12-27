"""
ARC (AI2 Reasoning Challenge) Dataset Downloader and Preprocessor

This script downloads the ARC-Challenge and ARC-Easy datasets from HuggingFace
and converts them into a standardized JSON format for evaluation.

Usage:
    python -m evaluation.dataset_download.arc

Output:
    - data/arc-c.json: ARC-Challenge dataset
    - data/arc-e.json: ARC-Easy dataset
"""

import json
import os
from pathlib import Path

from datasets import load_dataset


def process_arc_example(example: dict) -> dict:
    """
    Process a single ARC example into the target format.
    
    Args:
        example: Raw example from the HuggingFace dataset
        
    Returns:
        Processed example in the target format
    """
    processed = {
        "question": example["question"],
    }
    
    # Extract choices - ARC dataset has choices in a nested structure
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    
    # Map each label to its corresponding text
    for label, text in zip(labels, texts):
        processed[label] = text
    
    # Add the answer key
    processed["answer"] = example["answerKey"]
    
    return processed


def download_and_process_arc(subset: str, output_path: str) -> None:
    """
    Download and process an ARC subset.
    
    Args:
        subset: Either 'ARC-Challenge' or 'ARC-Easy'
        output_path: Path to save the processed JSON file
    """
    print(f"Downloading {subset} dataset from HuggingFace...")
    
    # Load the dataset
    dataset = load_dataset("allenai/ai2_arc", subset)
    
    # Process all splits and combine them
    all_examples = []
    
    for split_name in dataset.keys():
        print(f"  Processing {split_name} split ({len(dataset[split_name])} examples)...")
        for example in dataset[split_name]:
            processed = process_arc_example(example)
            all_examples.append(processed)
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    print(f"Saving {len(all_examples)} examples to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved {subset} dataset to {output_path}")


def main():
    """Main function to download and process both ARC datasets."""
    # Define output paths
    data_dir = Path("data")
    
    # Process ARC-Challenge
    download_and_process_arc(
        subset="ARC-Challenge",
        output_path=str(data_dir / "arc-c.json")
    )
    
    print()
    
    # Process ARC-Easy
    download_and_process_arc(
        subset="ARC-Easy",
        output_path=str(data_dir / "arc-e.json")
    )
    
    print()
    print("=" * 50)
    print("All datasets downloaded and processed successfully!")
    print(f"  - ARC-Challenge: {data_dir / 'arc-c.json'}")
    print(f"  - ARC-Easy: {data_dir / 'arc-e.json'}")


if __name__ == "__main__":
    main()