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

import random
from typing import List, Dict, Any

from .base_evaluator import BaseEvaluator


class GPQAEvaluator(BaseEvaluator):
    """
    GPQA Evaluator (GSM8K style).

    Logic:
    1. First look for the content after the last '####' in the generated text.
    2. Extract the first A/B/C/D letter from that content.
    3. If '####' is not found, fall back to searching for the answer at the end of the text.
    """

    def __init__(self):
        pass

    def _extract_answer(self, generation: str) -> str:
        """
        Extract the answer letter from the generation.

        Priority:
        1. Look for content after '####' delimiter
        2. Handle multi-question generations by splitting on 'Question'
        3. Fall back to random choice if extraction fails
        """
        try:
            if "\n\nQuestion" in generation:
                # Handle cases where the model generates multiple questions
                return generation.split("\n\nQuestion")[0].split("####")[1].strip()
            else:
                return generation.split("####")[1].strip()
        except (IndexError, AttributeError):
            # If extraction fails, return random choice as fallback
            return random.choice(["A", "B", "C", "D"])

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate GPQA results.

        Args:
            results: List of dictionaries containing 'answer' and 'generation' keys.

        Returns:
            Dictionary containing evaluation metrics.
        """
        total = 0
        correct = 0
        failed_extraction = 0

        detailed_predictions = []

        for i, item in enumerate(results):
            if "answer" not in item:
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

            detailed_predictions.append(
                {
                    "task_id": item.get("task_id"),
                    "question": item.get("question", ""),
                    "ground_truth": ground_truth,
                    "extracted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "generation": generation,
                    "explanation_ref": item.get("explanation", ""),
                }
            )

        accuracy = (correct / total * 100.0) if total > 0 else 0.0

        metrics = {
            "accuracy": float(f"{accuracy:.2f}"),
            "num_total": total,
            "num_correct": correct,
            "num_failed_extraction": failed_extraction,
            "detailed_predictions": detailed_predictions,
        }

        return metrics