# Copyright 2025 Tencent wechat. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import tempfile
import os
import os.path as osp
import subprocess
import sys
import urllib.request
import hashlib
from typing import List, Dict, Any, Optional

from .base_evaluator import BaseEvaluator

EVALPLUS_IMPORT_ERROR = """
Please install evalplus use following steps:
pip install evalplus
"""

# URL for downloading HumanEvalPlus dataset from the official evalplus repository
HUMANEVAL_PLUS_URL = "https://github.com/evalplus/humanevalplus_release/raw/main/HumanEvalPlus.jsonl.gz"

# Expected SHA256 hash for integrity verification (optional, can be updated if needed)
HUMANEVAL_PLUS_SHA256 = None  # Set to hash string if verification is desired


def get_cache_dir() -> str:
    """
    Get the cache directory for storing downloaded datasets.
    Uses XDG_CACHE_HOME if available, otherwise falls back to ~/.cache
    """
    cache_home = os.environ.get("XDG_CACHE_HOME", osp.expanduser("~/.cache"))
    cache_dir = osp.join(cache_home, "humaneval_plus")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_humaneval_plus(
    url: str = HUMANEVAL_PLUS_URL,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    expected_sha256: Optional[str] = HUMANEVAL_PLUS_SHA256,
) -> str:
    """
    Download the HumanEvalPlus dataset if not already cached.

    Args:
        url: URL to download the dataset from
        cache_dir: Directory to cache the downloaded file. If None, uses default cache dir.
        force_download: If True, re-download even if file exists
        expected_sha256: Expected SHA256 hash for verification (optional)

    Returns:
        Path to the downloaded/cached dataset file
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    filename = osp.basename(url)
    filepath = osp.join(cache_dir, filename)

    # Check if file already exists and is valid
    if osp.exists(filepath) and not force_download:
        if expected_sha256 is not None:
            actual_hash = compute_sha256(filepath)
            if actual_hash != expected_sha256:
                print(f"Cached file hash mismatch, re-downloading...")
            else:
                print(f"Using cached HumanEvalPlus dataset: {filepath}")
                return filepath
        else:
            print(f"Using cached HumanEvalPlus dataset: {filepath}")
            return filepath

    # Download the file
    print(f"Downloading HumanEvalPlus dataset from {url}...")
    try:
        # Create a request with a user-agent to avoid potential blocking
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; HumanEvalEvaluator/1.0)"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            total_size = response.getheader("Content-Length")
            if total_size:
                total_size = int(total_size)
                print(f"Total size: {total_size / (1024*1024):.2f} MB")

            # Download with progress indication
            downloaded = 0
            chunk_size = 8192
            with open(filepath, "wb") as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}%", end="", flush=True)

            print()  # New line after progress

        # Verify hash if provided
        if expected_sha256 is not None:
            actual_hash = compute_sha256(filepath)
            if actual_hash != expected_sha256:
                os.remove(filepath)
                raise ValueError(
                    f"Downloaded file hash mismatch. Expected {expected_sha256}, got {actual_hash}"
                )

        print(f"Successfully downloaded HumanEvalPlus dataset to {filepath}")
        return filepath

    except Exception as e:
        # Clean up partial download
        if osp.exists(filepath):
            os.remove(filepath)
        raise RuntimeError(f"Failed to download HumanEvalPlus dataset: {e}") from e


class HumanEvalEvaluator(BaseEvaluator):
    def __init__(self, k: List[int] = [1], dataset_path: Optional[str] = None):
        """
        Initialize the HumanEval evaluator.

        Args:
            k: List of k values for pass@k metrics
            dataset_path: Optional path to the HumanEvalPlus dataset.
                         If None, the dataset will be automatically downloaded.
        """
        try:
            import evalplus
        except ImportError:
            raise ImportError(EVALPLUS_IMPORT_ERROR)

        self.k = k

        # If no dataset path provided, download automatically
        if dataset_path is None:
            self.dataset_path = download_humaneval_plus()
        else:
            self.dataset_path = dataset_path

    def _postprocess(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        fence_blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
        if not fence_blocks:
            fence_blocks = re.findall(r"~~~(.*?)~~~", text, flags=re.DOTALL)

        if fence_blocks:
            block = fence_blocks[0]
            if "\n" in block:
                first_line, rest = block.split("\n", 1)
                if (
                    first_line.strip()
                    and len(first_line.strip()) <= 20
                    and first_line.strip().isalpha()
                ):
                    text = rest
                else:
                    text = block
            else:
                text = block

            text = text.lstrip("\n").rstrip()
        else:
            raw = text
            raw = raw.split("```", 1)[0]
            raw = raw.split("~~~", 1)[0]

            kept = []
            started = False
            for ln in raw.splitlines():
                if not started and (
                    ln.lstrip().startswith("from ") or ln.lstrip().startswith("import ")
                ):
                    continue
                started = True
                kept.append(ln)
            text = "\n".join(kept).lstrip("\n").rstrip()

            lines_all = text.splitlines()
            idx = 0
            while idx < len(lines_all) and lines_all[idx].strip() == "":
                idx += 1

            def leading_spaces(s: str) -> int:
                return len(s) - len(s.lstrip(" "))

            def is_comment_or_string_start(s: str) -> bool:
                t = s.strip()
                return t.startswith("#") or t.startswith(("'", '"'))

            while idx < len(lines_all) and (
                lines_all[idx].strip() == ""
                or is_comment_or_string_start(lines_all[idx])
            ):
                idx += 1

            pre_block_lines = []
            if idx < len(lines_all):
                first_indent = leading_spaces(lines_all[idx])
                if first_indent >= 4:
                    for j in range(idx, len(lines_all)):
                        ln = lines_all[j]
                        if ln.strip() == "" or is_comment_or_string_start(ln):
                            pre_block_lines.append(ln)
                            continue
                        curr_indent = leading_spaces(ln)
                        if curr_indent < 4:
                            break
                        pre_block_lines.append(ln)
                    if any(
                        ln.strip() and not is_comment_or_string_start(ln)
                        for ln in pre_block_lines
                    ):
                        text = "\n".join(pre_block_lines).rstrip()

            if ("def " in text or text.lstrip().startswith("def")) and not (
                pre_block_lines
                and any(
                    ln.strip() and not is_comment_or_string_start(ln)
                    for ln in pre_block_lines
                )
            ):
                m = re.search(r"\bdef\b[ \t]+[A-Za-z_][A-Za-z0-9_]*\s*\(", text)
                if m:
                    seg = text[m.start() :]
                    lines = seg.splitlines()
                    paren = 0
                    body_start = None
                    for i, line in enumerate(lines):
                        paren += line.count("(") - line.count(")")
                        if line.rstrip().endswith(":") and paren == 0:
                            body_start = i + 1
                            break
                    if body_start is None:
                        body_start = 1
                    body_lines = lines[body_start:]
                    text = "\n".join(body_lines).lstrip("\n").rstrip()

        lines = [ln for ln in text.splitlines() if ln.strip() != ""]
        if not lines:
            return ""

        def leading_spaces2(s: str) -> int:
            return len(s) - len(s.lstrip(" "))

        first_code_idx = None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s == "" or s.startswith("#") or s.startswith(("'", '"')):
                continue
            first_code_idx = i
            break

        if first_code_idx is None:
            return "\n".join(lines)

        first_indent = leading_spaces2(lines[first_code_idx])
        if first_indent == 0:
            lines = [("    " + ln) if ln.strip() else ln for ln in lines]

        first_code_indent = leading_spaces2(lines[first_code_idx])
        end_idx = None
        for i, ln in enumerate(lines[first_code_idx + 1 :], start=first_code_idx + 1):
            s = ln.strip()
            if s == "" or s.startswith("#") or s.startswith(("'", '"')):
                continue
            curr_indent = leading_spaces2(ln)
            if curr_indent < first_code_indent:
                end_idx = i
                break
        if end_idx is not None:
            lines = lines[:end_idx]

        return "\n".join(lines).rstrip()

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        evalplus_samples = []
        task_id_map = {}

        for item in results:
            task_id = item.get("task_id")
            generation = item.get("generation")
            prompt = item.get("original_prompt", item.get("prompt", ""))

            if task_id is None or generation is None:
                continue

            task_id_map[task_id] = item

            processed_completion = self._postprocess(generation)

            item["extracted_code"] = processed_completion

            solution = prompt + "\n" + processed_completion

            evalplus_samples.append({"task_id": task_id, "solution": solution})

        if not evalplus_samples:
            return {f"pass@{k}": 0.0 for k in self.k}

        with tempfile.TemporaryDirectory() as tmp_dir:
            samples_file = osp.join(tmp_dir, "predictions.jsonl")

            with open(samples_file, "w", encoding="utf-8") as f:
                for entry in evalplus_samples:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            cmd = [
                sys.executable,
                "-m",
                "evalplus.evaluate",
                "--dataset",
                "humaneval",
                "--samples",
                samples_file,
            ]

            env = os.environ.copy()
            if self.dataset_path and osp.exists(self.dataset_path):
                env["HUMANEVAL_OVERRIDE_PATH"] = self.dataset_path
            else:
                # If dataset path doesn't exist, let evalplus use its default
                print(
                    f"Warning: Dataset path {self.dataset_path} not found. "
                    "Using default evalplus dataset."
                )

            print(f"Running EvalPlus evaluation on {len(evalplus_samples)} samples...")
            try:
                subprocess.run(
                    cmd,
                    env=env,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                print(f"EvalPlus evaluation failed: {e.stderr.decode()}")
                return {f"base_pass@{k}": 0.0 for k in self.k}

            results_file = samples_file.replace(".jsonl", "_eval_results.json")

            if not osp.exists(results_file):
                print(f"Error: Result file {results_file} not found.")
                return {}

            with open(results_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            eval_results = eval_data.get("eval", {})

            base_pass_count = 0
            plus_pass_count = 0
            total_tasks = len(eval_results)

            if total_tasks == 0:
                return {}

            for t_id, res_list in eval_results.items():
                if not res_list:
                    continue

                res = res_list[0]
                base_status = res.get("base_status")
                plus_status = res.get("plus_status")

                is_base_pass = base_status == "pass"
                is_plus_pass = plus_status == "pass"

                if is_base_pass:
                    base_pass_count += 1
                if is_plus_pass:
                    plus_pass_count += 1

                if t_id in task_id_map:
                    task_id_map[t_id]["is_correct"] = is_plus_pass
                    task_id_map[t_id]["is_correct_base"] = is_base_pass
                    task_id_map[t_id]["is_correct_plus"] = is_plus_pass
                    task_id_map[t_id]["evalplus_result"] = res

            scores = {}
            for k in self.k:
                if k == 1:
                    scores["humaneval_pass@1"] = base_pass_count / total_tasks
                    scores["humaneval_plus_pass@1"] = plus_pass_count / total_tasks
                else:
                    pass

            return scores