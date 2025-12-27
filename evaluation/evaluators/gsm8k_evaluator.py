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


class Gsm8kEvaluator(BaseEvaluator):
    def __init__(self, atol: float = 1e-6):
        self.atol = atol

    def _extract_numbers(self, text: str, extract_first: bool = False) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.replace(",", "")
        numbers = re.findall(r"-?(?:\d+\.?\d*|\.\d+)", text)
        if not numbers:
            return "NULL"
        if extract_first:
            return numbers[0]
        return numbers[-1]

    def _postprocess_reference(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        if "####" in text:
            tail = text.split("####")[-1].strip()
            num = self._extract_numbers(tail, extract_first=True)
            if num != "NULL":
                return num

        return self._extract_numbers(text)

    def _postprocess_prediction(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        text = text.split("Question:")[0]

        if "####" in text:
            tail = text.split("####")[-1].strip()
            num = self._extract_numbers(tail, extract_first=True)
            if num != "NULL":
                return num

        return self._extract_numbers(text)

    def _equal(self, pred: str, refer: str) -> bool:
        if pred == "NULL" or refer == "NULL":
            return False
        try:
            return abs(float(pred) - float(refer)) < self.atol
        except ValueError:
            return pred == refer

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {"accuracy": 0.0}

        correct = 0
        total = 0
        failed_extraction = 0

        detailed_predictions = []

        for i, item in enumerate(results):
            gen = item.get("generation", "")
            ref_raw = item.get("answer", "")

            pred = self._postprocess_prediction(gen)
            refer = self._postprocess_reference(ref_raw)

            total += 1

            is_correct = False
            extraction_failed = pred == "NULL"

            if extraction_failed:
                failed_extraction += 1
            elif self._equal(pred, refer):
                correct += 1
                is_correct = True

            item["extracted_answer"] = pred
            item["is_correct"] = is_correct
            item["extraction_failed"] = extraction_failed

            detail_item = {
                "index": i,
                "question": item.get("question", item.get("prompt", "")),
                "ground_truth": refer,
                "generation": gen,
                "extracted_answer": pred,
                "normalized_prediction": pred,
                "normalized_ground_truth": refer,
                "is_correct": is_correct,
                "extraction_failed": extraction_failed,
            }
            detailed_predictions.append(detail_item)

        accuracy = (correct / total) if total > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "num_total": total,
            "num_correct": correct,
            "num_failed_extraction": failed_extraction,
            "detailed_predictions": detailed_predictions,
        }

        return metrics
