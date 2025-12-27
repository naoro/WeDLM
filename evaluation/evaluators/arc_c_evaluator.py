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

import re
from typing import List, Dict, Any

from .base_evaluator import BaseEvaluator


class ARCCEvaluator(BaseEvaluator):
    def __init__(self):
        pass

    def _extract_answer(self, generation: str, available_options: List[str]) -> str:
        if not generation:
            return ""
        text = generation.strip()
        valid_options_set = {opt.upper() for opt in available_options}

        if text and text[0].upper() in valid_options_set:
            return text[0].upper()

        patterns = [
            r"^([A-D])[.\s)]",
            r"(?:answer|Answer|ANSWER)[\s:]*([A-D])",
            r"(?:is|IS)[\s:]*([A-D])",
            r"\b([A-D])\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).upper()
                if candidate in valid_options_set:
                    return candidate

        for char in text:
            if char.upper() in valid_options_set:
                return char.upper()

        return ""

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = 0
        correct = 0
        failed_extraction = 0
        invalid_option = 0

        by_num_options = {}

        for item in results:
            if "answer" not in item or "generation" not in item:
                item["is_correct"] = False
                continue

            total += 1
            ground_truth = item["answer"].strip().upper()
            generation = item.get("generation", "")

            available_options = item.get("available_options", ["A", "B", "C", "D"])
            num_options = item.get("num_options", len(available_options))

            if num_options not in by_num_options:
                by_num_options[num_options] = {
                    "total": 0,
                    "correct": 0,
                    "failed_extraction": 0,
                    "invalid_option": 0,
                }

            by_num_options[num_options]["total"] += 1

            predicted_answer = self._extract_answer(generation, available_options)

            item["extracted_answer"] = predicted_answer
            is_correct = False

            if not predicted_answer:
                failed_extraction += 1
                by_num_options[num_options]["failed_extraction"] += 1
            elif predicted_answer not in [opt.upper() for opt in available_options]:
                invalid_option += 1
                by_num_options[num_options]["invalid_option"] += 1
            elif predicted_answer == ground_truth:
                correct += 1
                by_num_options[num_options]["correct"] += 1
                is_correct = True

            item["is_correct"] = is_correct

        accuracy = (correct / total * 100.0) if total > 0 else 0.0

        by_num_options_metrics = {}
        for num_opts, stats in by_num_options.items():
            opt_accuracy = (
                (stats["correct"] / stats["total"] * 100.0)
                if stats["total"] > 0
                else 0.0
            )
            by_num_options_metrics[f"{num_opts}_options"] = {
                "accuracy": float(f"{opt_accuracy:.2f}"),
                "correct": stats["correct"],
                "total": stats["total"],
                "failed_extraction": stats["failed_extraction"],
                "invalid_option": stats["invalid_option"],
            }

        metrics = {
            "accuracy": float(f"{accuracy:.2f}"),
            "num_total": total,
            "num_correct": correct,
            "num_failed_extraction": failed_extraction,
            "num_invalid_option": invalid_option,
        }

        if len(by_num_options) > 1:
            metrics["by_num_options"] = by_num_options_metrics

        return metrics
