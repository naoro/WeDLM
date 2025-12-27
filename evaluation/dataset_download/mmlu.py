"""
MMLU Dataset Download and Preprocessing Script

This script downloads the MMLU (Massive Multitask Language Understanding) dataset
from Hugging Face and converts it to a unified JSON format.

Usage:
    python -m evaluation.dataset_download.mmlu

Output:
    data/mmlu.json - A JSON file containing all MMLU samples in the specified format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

try:
    from datasets import load_dataset, get_dataset_config_names
except ImportError:
    raise ImportError(
        "Please install the 'datasets' library: pip install datasets"
    )


# MMLU dataset identifier on Hugging Face
DATASET_NAME = "cais/mmlu"

# Output path relative to project root
OUTPUT_DIR = "data"
OUTPUT_FILE = "mmlu.json"

# Answer mapping from index to letter
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def get_all_subjects() -> List[str]:
    """
    Retrieve all available subject names (configs) from the MMLU dataset.
    
    Returns:
        List of subject names available in the dataset.
    """
    print("Fetching available subjects from MMLU dataset...")
    subjects = get_dataset_config_names(DATASET_NAME)
    # Filter out 'all' config if present, as we want individual subjects
    subjects = [s for s in subjects if s != "all"]
    print(f"Found {len(subjects)} subjects.")
    return subjects


def process_sample(sample: Dict[str, Any], subject: str) -> Dict[str, str]:
    """
    Convert a single dataset sample to the target JSON format.
    
    Args:
        sample: A single sample from the MMLU dataset.
        subject: The subject name for this sample.
    
    Returns:
        A dictionary in the target format with keys:
        subject, question, A, B, C, D, answer
    """
    choices = sample["choices"]
    answer_idx = sample["answer"]
    
    return {
        "subject": subject,
        "question": sample["question"],
        "A": choices[0] if len(choices) > 0 else "",
        "B": choices[1] if len(choices) > 1 else "",
        "C": choices[2] if len(choices) > 2 else "",
        "D": choices[3] if len(choices) > 3 else "",
        "answer": ANSWER_MAP.get(answer_idx, "A")
    }


def download_and_process_subject(subject: str, split: str = "test") -> List[Dict[str, str]]:
    """
    Download and process a single subject from the MMLU dataset.
    
    Args:
        subject: The subject name to download.
        split: The dataset split to use (default: "test").
    
    Returns:
        A list of processed samples for the given subject.
    """
    print(f"  Processing subject: {subject}...")
    
    try:
        dataset = load_dataset(DATASET_NAME, subject, split=split)
        samples = [process_sample(sample, subject) for sample in dataset]
        print(f"    -> {len(samples)} samples processed.")
        return samples
    except Exception as e:
        print(f"    -> Error processing {subject}: {e}")
        return []


def download_all_subjects(splits: List[str] = None) -> List[Dict[str, str]]:
    """
    Download and process all subjects from all specified splits.
    
    Args:
        splits: List of splits to download (default: ["test", "validation", "dev"]).
    
    Returns:
        A list of all processed samples from all subjects and splits.
    """
    if splits is None:
        splits = ["test", "validation", "dev"]
    
    subjects = get_all_subjects()
    all_samples = []
    
    for split in splits:
        print(f"\nProcessing split: {split}")
        print("-" * 40)
        
        for subject in subjects:
            try:
                samples = download_and_process_subject(subject, split=split)
                all_samples.extend(samples)
            except Exception as e:
                print(f"  Skipping {subject}/{split}: {e}")
    
    return all_samples


def save_to_json(samples: List[Dict[str, str]], output_path: str) -> None:
    """
    Save the processed samples to a JSON file.
    
    Args:
        samples: List of processed samples to save.
        output_path: Path to the output JSON file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(samples)} samples to {output_path}")


def main():
    """
    Main entry point for the MMLU dataset download and preprocessing script.
    """
    print("=" * 60)
    print("MMLU Dataset Download and Preprocessing")
    print("=" * 60)
    
    # Determine output path
    # When running with python -m, the working directory is typically the project root
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # Download and process all subjects from the test split
    # You can modify this to include other splits like "validation" or "dev"
    print("\nDownloading MMLU dataset from Hugging Face...")
    all_samples = download_all_subjects(splits=["test"])
    
    # Save to JSON
    save_to_json(all_samples, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Output file: {output_path}")
    print("=" * 60)
    
    # Print a sample for verification
    if all_samples:
        print("\nSample entry:")
        print(json.dumps(all_samples[0], indent=2))


if __name__ == "__main__":
    main()