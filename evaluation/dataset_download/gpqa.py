"""
GPQA Diamond Dataset Download and Preprocessing Script

This script downloads the GPQA Diamond dataset from Hugging Face and converts it
to a standardized JSON format for evaluation purposes.

Usage:
    python -m evaluation.dataset_download.gpqa

Requirements:
    - datasets (pip install datasets)
    - huggingface_hub (pip install huggingface_hub)

Note:
    This dataset requires authentication with a Hugging Face token.
    You can get your token from: https://huggingface.co/settings/tokens
"""

import json
import os
import sys
from pathlib import Path


def get_huggingface_token() -> str:
    """
    Get Hugging Face token from user input or environment variable.
    
    Returns:
        str: The Hugging Face API token
    """
    # First check if token is in environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if token:
        print("Using Hugging Face token from environment variable.")
        return token
    
    # Otherwise, prompt user for input
    print("=" * 60)
    print("GPQA Dataset Download")
    print("=" * 60)
    print("\nThis dataset requires authentication with Hugging Face.")
    print("You can get your token from: https://huggingface.co/settings/tokens")
    print("Make sure you have accepted the dataset's terms of use at:")
    print("https://huggingface.co/datasets/Idavidrein/gpqa")
    print()
    
    token = input("Please enter your Hugging Face token: ").strip()
    
    if not token:
        print("Error: Token cannot be empty.")
        sys.exit(1)
    
    return token


def download_gpqa_diamond(token: str) -> list:
    """
    Download the GPQA Diamond dataset from Hugging Face.
    
    Args:
        token: Hugging Face API token
        
    Returns:
        list: The downloaded dataset as a list of examples
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import login
    except ImportError as e:
        print("Error: Required packages not installed.")
        print("Please install them with: pip install datasets huggingface_hub")
        sys.exit(1)
    
    print("\nAuthenticating with Hugging Face...")
    login(token=token)
    
    print("Downloading GPQA Diamond dataset...")
    print("(This may take a moment)")
    
    try:
        dataset = load_dataset(
            "Idavidrein/gpqa",
            "gpqa_diamond",
            trust_remote_code=True
        )
        return dataset["train"]
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nPossible reasons:")
        print("1. Invalid or expired token")
        print("2. You haven't accepted the dataset's terms of use")
        print("3. Network connection issues")
        sys.exit(1)


def extract_answer_letter(row: dict) -> str:
    """
    Extract the correct answer letter (A, B, C, or D) from the dataset row.
    
    Args:
        row: A single row from the dataset
        
    Returns:
        str: The correct answer letter
    """
    correct_answer = row.get("Correct Answer", "")
    
    # Check which choice matches the correct answer
    choices = {
        "A": row.get("Incorrect Answer 1", ""),
        "B": row.get("Incorrect Answer 2", ""),
        "C": row.get("Incorrect Answer 3", ""),
        "D": row.get("Correct Answer", "")
    }
    
    # The dataset structure may vary, so we need to handle different cases
    # Usually the correct answer is directly provided
    for letter, choice in choices.items():
        if choice == correct_answer:
            return letter
    
    # If we can't determine, return empty string
    return ""


def process_dataset(dataset) -> list:
    """
    Process the raw dataset into the target JSON format.
    
    Args:
        dataset: The raw Hugging Face dataset
        
    Returns:
        list: Processed data in the target format
    """
    processed_data = []
    
    print(f"\nProcessing {len(dataset)} examples...")
    
    for idx, row in enumerate(dataset):
        # Extract question and choices
        question = row.get("Question", "")
        
        # The GPQA dataset has specific column names for choices
        # Correct Answer is one of the choices, and there are 3 incorrect answers
        correct_answer_text = row.get("Correct Answer", "")
        incorrect_1 = row.get("Incorrect Answer 1", "")
        incorrect_2 = row.get("Incorrect Answer 2", "")
        incorrect_3 = row.get("Incorrect Answer 3", "")
        
        # Get the original order or shuffle indicator if available
        # For consistency, we'll use a fixed mapping where D is always correct
        # But we should check if the dataset provides the original ordering
        
        # Standard format: A, B, C are incorrect, D is correct (unless specified otherwise)
        choice_a = incorrect_1
        choice_b = incorrect_2
        choice_c = incorrect_3
        choice_d = correct_answer_text
        answer_letter = "D"
        
        # Get explanation if available
        explanation = row.get("Explanation", "")
        
        # Get domain/subject
        domain = row.get("Subdomain", "") or row.get("Domain", "") or row.get("High-level domain", "")
        
        # Create the processed entry
        entry = {
            "id": idx,
            "question": question,
            "A": choice_a,
            "B": choice_b,
            "C": choice_c,
            "D": choice_d,
            "answer": answer_letter,
            "full_answer_text": correct_answer_text,
            "explanation": explanation,
            "domain": domain
        }
        
        processed_data.append(entry)
    
    return processed_data


def save_to_json(data: list, output_path: str) -> None:
    """
    Save the processed data to a JSON file.
    
    Args:
        data: The processed dataset
        output_path: Path to save the JSON file
    """
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"Total examples: {len(data)}")


def main():
    """
    Main function to orchestrate the download and processing pipeline.
    """
    # Define output path (relative to current working directory)
    output_dir = Path("data")
    output_path = output_dir / "gpqa_diamond.json"
    
    # Get authentication token
    token = get_huggingface_token()
    
    # Download dataset
    dataset = download_gpqa_diamond(token)
    
    # Process dataset
    processed_data = process_dataset(dataset)
    
    # Save to JSON
    save_to_json(processed_data, str(output_path))
    
    # Print sample entry for verification
    if processed_data:
        print("\n" + "=" * 60)
        print("Sample entry (first item):")
        print("=" * 60)
        sample = processed_data[0]
        print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")
    
    print("\nDone!")


if __name__ == "__main__":
    main()