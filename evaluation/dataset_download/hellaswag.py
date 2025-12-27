"""
HellaSwag Dataset Preprocessing Script

This script downloads the HellaSwag dataset from Hugging Face and converts it
to a standardized JSON format for evaluation purposes.

Usage:
    python -m evaluation.dataset_download.hellaswag

Output:
    data/hellaswag.json
"""

import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "Please install the 'datasets' library: pip install datasets"
    )


def convert_label_to_letter(label: int) -> str:
    """
    Convert numeric label (0-3) to letter (A-D).
    
    Args:
        label: Integer label from 0 to 3
        
    Returns:
        Corresponding letter A, B, C, or D
    """
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    return label_map.get(label, "A")


def format_question(activity_label: str, ctx: str) -> str:
    """
    Format the question by combining activity label and context.
    
    Args:
        activity_label: The activity label/topic
        ctx: The context text
        
    Returns:
        Formatted question string
    """
    # Combine activity label and context
    if activity_label:
        return f"{activity_label}: {ctx}"
    return ctx


def process_hellaswag_dataset(split: str = "test") -> list:
    """
    Download and process the HellaSwag dataset.
    
    Args:
        split: Dataset split to use (default: "test")
        
    Returns:
        List of processed examples in the target format
    """
    print(f"Loading HellaSwag dataset ({split} split)...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("Rowan/hellaswag", split=split)
    
    print(f"Processing {len(dataset)} examples...")
    
    processed_data = []
    
    for example in dataset:
        # Extract fields from the dataset
        activity_label = example.get("activity_label", "")
        ctx = example.get("ctx", "")
        endings = example.get("endings", [])
        label = example.get("label", 0)
        
        # Handle label - it might be string or int
        if isinstance(label, str):
            label = int(label) if label.isdigit() else 0
        
        # Ensure we have exactly 4 endings
        while len(endings) < 4:
            endings.append("")
        
        # Format the question
        question = format_question(activity_label, ctx)
        
        # Create the processed example
        processed_example = {
            "question": question,
            "A": endings[0],
            "B": endings[1],
            "C": endings[2],
            "D": endings[3],
            "answer": convert_label_to_letter(label)
        }
        
        processed_data.append(processed_example)
    
    return processed_data


def save_to_json(data: list, output_path: str) -> None:
    """
    Save processed data to a JSON file.
    
    Args:
        data: List of processed examples
        output_path: Path to save the JSON file
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data)} examples to {output_path}")


def main():
    """
    Main function to run the preprocessing pipeline.
    """
    # Configuration
    output_path = "data/hellaswag.json"
    split = "test"  # Only use test set
    
    # Process dataset
    processed_data = process_hellaswag_dataset(split=split)
    
    # Save to JSON
    save_to_json(processed_data, output_path)
    
    # Print sample
    print("\nSample output:")
    print(json.dumps(processed_data[0], indent=2))
    
    print(f"\nDone! Total examples: {len(processed_data)}")


if __name__ == "__main__":
    main()