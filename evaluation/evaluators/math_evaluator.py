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
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .base_evaluator import BaseEvaluator


def _last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    else:
        return string[idx : right_brace_idx + 1]


def _remove_boxed(s: str) -> Optional[str]:
    left = "\\boxed{"
    try:
        if s.startswith(left):
            count = 0
            start = len(left) - 1
            for i in range(start, len(s)):
                if s[i] == "{":
                    count += 1
                elif s[i] == "}":
                    count -= 1
                    if count == 0:
                        return s[len(left) : i]
            return None

        left = "\\fbox{"
        if s.startswith(left):
            count = 0
            start = len(left) - 1
            for i in range(start, len(s)):
                if s[i] == "{":
                    count += 1
                elif s[i] == "}":
                    count -= 1
                    if count == 0:
                        return s[len(left) : i]
            return None

        return None
    except Exception:
        return None


def _strip_string(string: str) -> str:
    if not isinstance(string, str):
        return ""

    string = string.replace("\n", "").replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" ", "")
    string = re.sub(r"(\d),(\d)", r"\1\2", string)

    while string.startswith("{") and string.endswith("}"):
        count = 0
        is_single_pair = True
        for i, c in enumerate(string):
            if c == "{":
                count += 1
            elif c == "}":
                count -= 1
                if count == 0 and i < len(string) - 1:
                    is_single_pair = False
                    break

        if is_single_pair and count == 0:
            string = string[1:-1]
        else:
            break

    return string


class MATHEvaluator(BaseEvaluator):
    def __init__(self):
        pass

    def _truncate_extra_problems(self, generation: str) -> str:
        if not generation:
            return ""

        patterns = ["\nProblem:", "\n\nProblem:", "\nProblem ", "\n\nProblem "]
        min_idx = len(generation)
        for pattern in patterns:
            idx = generation.find(pattern)
            if idx > 0:
                min_idx = min(min_idx, idx)

        return generation[:min_idx] if min_idx < len(generation) else generation

    def _extract_answer(self, generation: str) -> str:
        if not generation:
            return ""

        generation = self._truncate_extra_problems(generation)

        boxed_str = _last_boxed_only_string(generation)
        if boxed_str is None:
            return ""

        answer = _remove_boxed(boxed_str)
        if answer is None:
            return ""

        while answer.startswith("{") and answer.endswith("}"):
            count = 0
            is_single_wrapped_pair = True
            for i, char in enumerate(answer[:-1]):
                if char == "{":
                    count += 1
                elif char == "}":
                    count -= 1
                if count == 0:
                    is_single_wrapped_pair = False
                    break

            if is_single_wrapped_pair:
                answer = answer[1:-1]
            else:
                break

        return answer

    def _is_equiv(self, pred_str: str, gold_str: str) -> bool:
        if pred_str is None or gold_str is None:
            return False

        pred_norm = _strip_string(pred_str)
        gold_norm = _strip_string(gold_str)

        return pred_norm == gold_norm

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = 0
        correct = 0
        failed_extraction = 0

        level_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        type_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        detailed_predictions = []

        for i, item in enumerate(results):
            if "answer" not in item or "generation" not in item:
                item["is_correct"] = False
                continue

            total += 1
            ground_truth = item["answer"]
            generation = item.get("generation", "")
            level = item.get("level", "unknown")
            type_ = item.get("type", "unknown")

            predicted_answer = self._extract_answer(generation)

            is_correct = False
            extraction_failed = not bool(predicted_answer) and not self._is_equiv(
                "", ground_truth
            )

            if extraction_failed:
                failed_extraction += 1
            elif self._is_equiv(predicted_answer, ground_truth):
                correct += 1
                is_correct = True
                level_stats[level]["correct"] += 1
                type_stats[type_]["correct"] += 1

            level_stats[level]["total"] += 1
            type_stats[type_]["total"] += 1

            item["extracted_answer"] = predicted_answer
            item["is_correct"] = is_correct
            item["extraction_failed"] = extraction_failed

            detail_item = {
                "index": i,
                "question": item.get("question", ""),
                "ground_truth": ground_truth,
                "generation": generation,
                "extracted_answer": predicted_answer,
                "normalized_prediction": _strip_string(predicted_answer),
                "normalized_ground_truth": _strip_string(ground_truth),
                "is_correct": is_correct,
                "extraction_failed": extraction_failed,
                "level": level,
                "type": type_,
            }
            detailed_predictions.append(detail_item)

        accuracy = (correct / total * 100.0) if total > 0 else 0.0

        def calculate_subgroup_accuracy(stats_dict):
            acc_dict = {}
            for group, stats in stats_dict.items():
                acc = (
                    (stats["correct"] / stats["total"] * 100.0)
                    if stats["total"] > 0
                    else 0.0
                )
                acc_dict[group] = {
                    "accuracy": float(f"{acc:.2f}"),
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
            return dict(sorted(acc_dict.items()))

        per_level_accuracy = calculate_subgroup_accuracy(level_stats)
        per_type_accuracy = calculate_subgroup_accuracy(type_stats)

        metrics = {
            "accuracy": float(f"{accuracy:.2f}"),
            "num_total": total,
            "num_correct": correct,
            "num_failed_extraction": failed_extraction,
        }

        if len(per_level_accuracy) > 1 or (
            len(per_level_accuracy) == 1 and "unknown" not in per_level_accuracy
        ):
            metrics["per_level_accuracy"] = per_level_accuracy
        if len(per_type_accuracy) > 1 or (
            len(per_type_accuracy) == 1 and "unknown" not in per_type_accuracy
        ):
            metrics["per_type_accuracy"] = per_type_accuracy

        metrics["detailed_predictions"] = detailed_predictions

        return metrics
