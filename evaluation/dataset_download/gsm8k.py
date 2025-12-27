"""
GSM8K Dataset Download and Preprocessing Script

This script downloads the GSM8K dataset from Hugging Face and processes it
into a JSONL format suitable for evaluation.

Usage:
    python -m evaluation.dataset_download.gsm8k

Output:
    data/gsm8k.jsonl
"""

import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found.")
    print("Please install it using: pip install datasets")
    exit(1)


def download_gsm8k():
    """
    Download GSM8K dataset from Hugging Face.
    
    Returns:
        Dataset object containing the GSM8K data.
    """
    print("Downloading GSM8K dataset from Hugging Face...")
    dataset = load_dataset("openai/gsm8k", "main")
    print("Download completed.")
    return dataset


def process_example(example: dict) -> dict:
    """
    Process a single example from the dataset into the target format.
    
    Args:
        example: A dictionary containing 'question' and 'answer' fields.
    
    Returns:
        A dictionary with 'question' and 'answer' keys in the target format.
    """
    return {
        "question": example["question"],
        "answer": example["answer"]
    }


def save_to_jsonl(data: list, output_path: str):
    """
    Save processed data to a JSONL file.
    
    Args:
        data: List of dictionaries to save.
        output_path: Path to the output file.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            # Use ensure_ascii=False to keep unicode characters readable
            # but the original data should not contain Chinese characters
            json_line = json.dumps(item, ensure_ascii=True)
            f.write(json_line + "\n")
    
    print(f"Saved {len(data)} examples to {output_path}")


def main():
    """
    Main function to download, process, and save the GSM8K dataset.
    """
    # Configuration
    output_path = "data/gsm8k.jsonl"
    
    # Download dataset
    dataset = download_gsm8k()
    
    # Process train and test splits
    processed_data = []
    
    # Process test split (commonly used for evaluation)
    print("Processing test split...")
    for example in dataset["test"]:
        processed_example = process_example(example)
        processed_data.append(processed_example)
    
    print(f"Processed {len(processed_data)} examples from test split.")
    
    # Optionally process train split as well
    # Uncomment the following lines if you want to include training data
    # print("Processing train split...")
    # for example in dataset["train"]:
    #     processed_example = process_example(example)
    #     processed_data.append(processed_example)
    # print(f"Total processed: {len(processed_data)} examples.")
    
    # Save to JSONL
    save_to_jsonl(processed_data, output_path)
    
    # Print sample output
    print("\nSample output (first example):")
    print(json.dumps(processed_data[0], ensure_ascii=True, indent=2))
    
    print(f"\nDone! Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()