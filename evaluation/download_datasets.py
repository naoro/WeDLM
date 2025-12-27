#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WeDLM Dataset Downloader

A unified script to download all evaluation datasets for WeDLM.
Supports downloading all datasets at once or selecting specific ones.

Usage:
    # Download all datasets
    python evaluation/download_datasets.py --all
    
    # Download specific datasets
    python evaluation/download_datasets.py --datasets gsm8k math humaneval
    
    # List available datasets
    python evaluation/download_datasets.py --list
    
    # Check dataset status
    python evaluation/download_datasets.py --status
"""

import argparse
import importlib
import os
import sys
from pathlib import Path


def get_project_root():
    """Get the project root directory (parent of evaluation/)."""
    return Path(__file__).parent.parent.absolute()


def setup_path():
    """Add project root to Python path for imports."""
    root = get_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# Setup path before imports
setup_path()


# Available datasets and their corresponding download modules
AVAILABLE_DATASETS = {
    "arc": {
        "module": "evaluation.dataset_download.arc",
        "files": ["data/arc-c.json", "data/arc-e.json"],
        "description": "ARC-Challenge and ARC-Easy (AI2 Reasoning Challenge)"
    },
    "gsm8k": {
        "module": "evaluation.dataset_download.gsm8k",
        "files": ["data/gsm8k.jsonl"],
        "description": "GSM8K (Grade School Math 8K)"
    },
    "math": {
        "module": "evaluation.dataset_download.math",
        "files": ["data/math.json"],
        "description": "MATH (Mathematics Aptitude Test of Heuristics)"
    },
    "humaneval": {
        "module": "evaluation.dataset_download.humaneval",
        "files": ["data/humaneval.jsonl"],
        "description": "HumanEval (Code Generation Benchmark)"
    },
    "mbpp": {
        "module": "evaluation.dataset_download.mbpp",
        "files": ["data/mbpp.jsonl"],
        "description": "MBPP (Mostly Basic Python Problems)"
    },
    "mmlu": {
        "module": "evaluation.dataset_download.mmlu",
        "files": ["data/mmlu.json"],
        "description": "MMLU (Massive Multitask Language Understanding)"
    },
    "hellaswag": {
        "module": "evaluation.dataset_download.hellaswag",
        "files": ["data/hellaswag.json"],
        "description": "HellaSwag (Commonsense NLI)"
    },
    "gpqa": {
        "module": "evaluation.dataset_download.gpqa",
        "files": ["data/gpqa_diamond.json"],
        "description": "GPQA Diamond (Graduate-Level Google-Proof Q&A)"
    },
}

# Dataset groups for convenience
DATASET_GROUPS = {
    "reasoning": ["gsm8k", "math", "gpqa"],
    "code": ["humaneval", "mbpp"],
    "knowledge": ["mmlu", "arc", "hellaswag"],
    "all": list(AVAILABLE_DATASETS.keys()),
}


def check_dataset_status(dataset_name: str) -> dict:
    """Check if a dataset's files exist."""
    root = get_project_root()
    info = AVAILABLE_DATASETS[dataset_name]
    status = {
        "name": dataset_name,
        "description": info["description"],
        "files": {},
        "complete": True
    }
    
    for file_path in info["files"]:
        full_path = root / file_path
        exists = full_path.exists()
        status["files"][file_path] = exists
        if not exists:
            status["complete"] = False
    
    return status


def print_status():
    """Print the status of all datasets."""
    print("\n" + "=" * 60)
    print("📊 Dataset Status")
    print("=" * 60)
    
    complete_count = 0
    for name in AVAILABLE_DATASETS:
        status = check_dataset_status(name)
        icon = "✅" if status["complete"] else "❌"
        print(f"\n{icon} {name}: {status['description']}")
        for file_path, exists in status["files"].items():
            file_icon = "  ✓" if exists else "  ✗"
            print(f"   {file_icon} {file_path}")
        if status["complete"]:
            complete_count += 1
    
    print("\n" + "-" * 60)
    print(f"Summary: {complete_count}/{len(AVAILABLE_DATASETS)} datasets ready")
    print("=" * 60 + "\n")


def print_available_datasets():
    """Print list of available datasets."""
    print("\n" + "=" * 60)
    print("📋 Available Datasets")
    print("=" * 60)
    
    for name, info in AVAILABLE_DATASETS.items():
        print(f"\n  • {name}")
        print(f"    {info['description']}")
    
    print("\n" + "-" * 60)
    print("Dataset Groups:")
    print("  • reasoning: gsm8k, math, gpqa")
    print("  • code: humaneval, mbpp")
    print("  • knowledge: mmlu, arc, hellaswag")
    print("  • all: all datasets")
    print("=" * 60 + "\n")


def download_dataset(dataset_name: str, force: bool = False) -> bool:
    """Download a single dataset."""
    if dataset_name not in AVAILABLE_DATASETS:
        print(f"❌ Unknown dataset: {dataset_name}")
        return False
    
    info = AVAILABLE_DATASETS[dataset_name]
    
    # Check if already downloaded
    if not force:
        status = check_dataset_status(dataset_name)
        if status["complete"]:
            print(f"⏭️  {dataset_name}: Already downloaded, skipping (use --force to re-download)")
            return True
    
    print(f"\n📥 Downloading {dataset_name}...")
    print(f"   {info['description']}")
    
    try:
        # Import and run the download module
        module = importlib.import_module(info["module"])
        
        # Most download scripts have a main() function or run on import
        if hasattr(module, "main"):
            module.main()
        elif hasattr(module, "download"):
            module.download()
        
        # Verify download
        status = check_dataset_status(dataset_name)
        if status["complete"]:
            print(f"✅ {dataset_name}: Download complete!")
            return True
        else:
            print(f"⚠️  {dataset_name}: Download may be incomplete")
            for file_path, exists in status["files"].items():
                if not exists:
                    print(f"   Missing: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ {dataset_name}: Download failed - {str(e)}")
        return False


def download_datasets(datasets: list, force: bool = False):
    """Download multiple datasets."""
    # Expand dataset groups
    expanded = []
    for d in datasets:
        if d in DATASET_GROUPS:
            expanded.extend(DATASET_GROUPS[d])
        else:
            expanded.append(d)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_datasets = []
    for d in expanded:
        if d not in seen:
            seen.add(d)
            unique_datasets.append(d)
    
    print("\n" + "=" * 60)
    print(f"🚀 WeDLM Dataset Downloader")
    print(f"   Datasets to download: {', '.join(unique_datasets)}")
    print("=" * 60)
    
    results = {"success": [], "failed": [], "skipped": []}
    
    for dataset in unique_datasets:
        if dataset not in AVAILABLE_DATASETS:
            print(f"\n⚠️  Unknown dataset: {dataset}, skipping...")
            results["failed"].append(dataset)
            continue
        
        success = download_dataset(dataset, force)
        if success:
            status = check_dataset_status(dataset)
            if status["complete"]:
                results["success"].append(dataset)
            else:
                results["skipped"].append(dataset)
        else:
            results["failed"].append(dataset)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Download Summary")
    print("=" * 60)
    if results["success"]:
        print(f"✅ Success: {', '.join(results['success'])}")
    if results["skipped"]:
        print(f"⏭️  Skipped: {', '.join(results['skipped'])}")
    if results["failed"]:
        print(f"❌ Failed: {', '.join(results['failed'])}")
    print("=" * 60 + "\n")
    
    return len(results["failed"]) == 0


def main():
    parser = argparse.ArgumentParser(
        description="WeDLM Dataset Downloader - Download evaluation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/download_datasets.py --all                    Download all datasets
  python evaluation/download_datasets.py --datasets gsm8k math    Download specific datasets
  python evaluation/download_datasets.py --datasets reasoning     Download all reasoning datasets
  python evaluation/download_datasets.py --list                   List available datasets
  python evaluation/download_datasets.py --status                 Check dataset status
  python evaluation/download_datasets.py --all --force            Re-download all datasets
        """
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        metavar="DATASET",
        help="Specific datasets to download (e.g., gsm8k math humaneval)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available datasets"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Check download status of all datasets"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if files exist"
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.list:
        print_available_datasets()
        return 0
    
    if args.status:
        print_status()
        return 0
    
    if args.all:
        success = download_datasets(["all"], force=args.force)
        return 0 if success else 1
    
    if args.datasets:
        success = download_datasets(args.datasets, force=args.force)
        return 0 if success else 1
    
    # No arguments provided, show help
    parser.print_help()
    print("\n💡 Tip: Use --all to download all datasets, or --list to see available options.")
    return 0


if __name__ == "__main__":
    sys.exit(main())