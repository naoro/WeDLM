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


class HellaSwagEvaluator(BaseEvaluator):
    def __init__(self):
        pass

    def _extract_answer(self, generation: str) -> str:
        if not generation:
            return ""

        text = generation.strip()

        match = re.search(
            r"(?:the\s+)?(?:answer|option|choice)\s*(?:is)?\s*:?\s*\b([A-D])\b",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

        match = re.match(r"\s*([A-D])(?:[.\s)]|$)", text)
        if match:
            return match.group(1).upper()

        all_options = re.findall(r"\b([A-D])\b", text)
        if all_options:
            return all_options[-1].upper()

        return ""

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = 0
        correct = 0
        failed_extraction = 0

        for item in results:
            if "answer" not in item or "generation" not in item:
                item["is_correct"] = False
                continue

            total += 1
            ground_truth = item["answer"].strip().upper()
            generation = item.get("generation", "")

            predicted_answer = self._extract_answer(generation)

            is_correct = False
            if not predicted_answer:
                failed_extraction += 1
            elif predicted_answer == ground_truth:
                correct += 1
                is_correct = True

            item["extracted_answer"] = predicted_answer
            item["is_correct"] = is_correct

        accuracy = (correct / total * 100.0) if total > 0 else 0.0

        metrics = {
            "accuracy": float(f"{accuracy:.2f}"),
            "num_total": total,
            "num_correct": correct,
            "num_failed_extraction": failed_extraction,
        }

        return metrics
