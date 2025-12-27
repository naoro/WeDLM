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
from typing import List, Dict, Any, Optional

from .base_dataset import BaseDataset


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
        if not s.startswith(left) or not s.endswith("}"):
            left = "\\fbox{"
            if not s.startswith(left) or not s.endswith("}"):
                return None
        return s[len(left) : -1]
    except Exception:
        return None


def _extract_ground_truth(solution_str: str) -> Optional[str]:
    boxed_str = _last_boxed_only_string(solution_str)
    if boxed_str is None:
        return None
    answer = _remove_boxed(boxed_str)
    return answer


class MATHDataset(BaseDataset):
    def __init__(self):
        self.path = (
            "data/math.json"
        )

        self._fewshot_prefix = (
            "Problem:\nFind the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.\n"
            "Solution:\nThe expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $\\boxed{[2,5)}$. I hope it is correct.\n\n"
            "Problem:\nIf $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$\n"
            "Solution:\nWe have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $\\boxed{24}$. I hope it is correct.\n\n"
            "Problem:\nTerrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?\n"
            "Solution:\nIf Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \\begin{align*} 30n&=480\\\\ \\Rightarrow\\qquad n&=480/30=\\boxed{16} \\end{align*}\nFinal Answer: The final answer is $\\boxed{16}$. I hope it is correct.\n\n"
            "Problem:\nIf the system of equations: \\begin{align*} 6x-4y&=a,\\\\ 6y-9x &=b. \\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.\n"
            "Solution:\nIf we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $\\boxed{-\\frac{2}{3}}$. I hope it is correct.\n\n"
        )

    @property
    def name(self) -> str:
        return "math"

    @property
    def evaluator_name(self) -> str:
        return "math"

    def _build_prompt(self, problem: str) -> str:
        current_question = f"Problem:\n{problem}\nSolution:\n"

        return f"{self._fewshot_prefix}{current_question}"

    def load(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    
        if not isinstance(raw_data, list):
            raise ValueError(f"Expected a list in {self.path}, got {type(raw_data)}")

        skipped_count = 0
        for idx, item in enumerate(raw_data):
            if "problem" not in item or "solution" not in item:
                raise ValueError(
                    f"Missing 'problem' or 'solution' in MATH sample at index {idx}: {item}"
                )

            answer = _extract_ground_truth(item["solution"])
            if answer is None:
                skipped_count += 1
                continue

            prompt = self._build_prompt(problem=item["problem"])

            data_item = {
                "task_id": idx,
                "prompt": prompt,
                "answer": answer,
                "level": item.get("level", "unknown"),
                "type": item.get("type", "unknown"),
                "full_solution": item["solution"],
            }
            data.append(data_item)

        if skipped_count > 0:
            print(
                f"Warning: Skipped {skipped_count} samples from MATH dataset because no '\\boxed' answer was found in the solution."
            )

        if not data:
            raise ValueError(f"No valid data found in {self.path}!")

        return data

    def get_recommended_config(self) -> Optional[Dict[str, Any]]:
        return {
            "max_new_tokens": 1024,
            "temperature": 0.0,
        }
