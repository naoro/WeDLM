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


_INSTRUCTION_PREFIX = """from typing import List

def filter_positive(numbers: List[int]) -> List[int]:
    \"\"\" Return a list containing only positive numbers from the input list.
    >>> filter_positive([-1, 2, -4, 5, 0])
    [2, 5]
    >>> filter_positive([-1, -2, -3])
    []
    \"\"\"
    return [x for x in numbers if x > 0]


def reverse_string(text: str) -> str:
    \"\"\" Return the reversed version of the input string.
    >>> reverse_string("hello")
    "olleh"
    >>> reverse_string("world")
    "dlrow"
    \"\"\"
    return text[::-1]


def get_odd_numbers(numbers: List[int]) -> List[int]:
    \"\"\" Return a list containing only the odd numbers from the input list.
    >>> get_odd_numbers([1, 2, 3, 4, 5])
    [1, 3, 5]
    >>> get_odd_numbers([2, 4, 6])
    []
    \"\"\"
    return [num for num in numbers if num % 2 != 0]


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False


"""


class HumanEvalDataset(BaseDataset):
    def __init__(self):
        self.path = "data/humaneval.jsonl"

    @property
    def name(self) -> str:
        return "humaneval"

    @property
    def evaluator_name(self) -> str:
        return "humaneval"

    def load(self) -> List[Dict[str, Any]]:
        data = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        json_obj = json.loads(line)
                        if "prompt" not in json_obj:
                            raise ValueError(
                                f"Line in {self.path} is missing 'prompt' key: {line.strip()}"
                            )
                        if "task_id" not in json_obj:
                            raise ValueError(
                                f"Line in {self.path} is missing 'task_id' key: {line.strip()}"
                            )

                        original_prompt = json_obj["prompt"]

                        templated_prompt = _INSTRUCTION_PREFIX + original_prompt

                        json_obj["prompt"] = templated_prompt
                        json_obj["original_prompt"] = original_prompt

                        data.append(json_obj)
                    except json.JSONDecodeError:
                        raise ValueError(
                            f"Failed to decode JSON from line in {self.path}: {line.strip()}"
                        )
        if not data:
            raise ValueError(f"No valid data found in {self.path}!")
        return data

    def get_recommended_config(self) -> Optional[Dict[str, Any]]:
        return {
            "max_new_tokens": 256,
            "temperature": 0.0,
        }
