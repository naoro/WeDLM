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


class MBPPDataset(BaseDataset):
    def __init__(self):

        self.path = "data/mbpp.jsonl"

        self._fewshot_block = "You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n\n[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n[DONE] \n\n \nYou are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \nassert is_not_prime(10) == True \nassert is_not_prime(35) == True \n\n[BEGIN]\n 'import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result' \n[DONE] \n\n \nYou are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n\n[BEGIN]\n 'import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n[DONE] \n\n \n"

    @property
    def name(self) -> str:
        return "mbpp"

    @property
    def evaluator_name(self) -> str:
        return "mbpp"

    def _build_prompt(self, text: str, test_list: List[str]) -> str:
        test_block = "\n".join(test_list)

        tail = (
            "You are an expert Python programmer, and here is your task: "
            f"{text} Your code should pass these tests:\n\n "
            f"{test_block}  \n"
            "[BEGIN]\n"
        )
        return f"{self._fewshot_block}\n{tail}"

    def load(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Failed to decode JSON from line in {self.path}: {line.strip()}"
                    )

                if "text" not in obj:
                    raise ValueError(f"Missing 'text' in MBPP sample: {obj}")
                if "task_id" not in obj:
                    raise ValueError(f"Missing 'task_id' in MBPP sample: {obj}")
                if "test_list" not in obj or not isinstance(obj["test_list"], list):
                    raise ValueError(
                        f"Missing or invalid 'test_list' in MBPP sample: {obj}"
                    )
                if "test_setup_code" not in obj:
                    obj["test_setup_code"] = ""

                text: str = obj["text"]
                test_list: List[str] = obj["test_list"]
                prompt = self._build_prompt(text, test_list)

                item = {
                    "task_id": obj["task_id"],
                    "prompt": prompt,
                    "text": text,
                    "test_setup_code": obj.get("test_setup_code", ""),
                    "test_list": test_list,
                    "challenge_test_list": obj.get("challenge_test_list", []),
                }
                data.append(item)

        if not data:
            raise ValueError(f"No valid data found in {self.path}!")

        return data

    def get_recommended_config(self) -> Optional[Dict[str, Any]]:
        return {
            "max_new_tokens": 512,
            "temperature": 0.1,
        }
