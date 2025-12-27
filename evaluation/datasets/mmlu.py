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


class MMLUDataset(BaseDataset):
    def __init__(self):
        self.path = "data/mmlu.json"

        self._fewshot_prefix = (
            "The following are multiple choice questions (with answers) about elementary mathematics.\n\n"
            "The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?\n"
            "A. 5 thousands\n"
            "B. 5 hundreds\n"
            "C. 5 tens\n"
            "D. 5 ones\n"
            "Answer: A\n\n"
            'Olivia used the rule "Add 11" to create the number pattern shown below. 10, 21, 32, 43, 54 Which statement about the number pattern is true?\n'
            "A. The 10th number in the pattern will be an even number.\n"
            "B. The number pattern will never have two even numbers next to each other.\n"
            "C. The next two numbers in the pattern will be an even number then an odd number.\n"
            "D. If the number pattern started with an odd number then the pattern would have only odd numbers in it.\n"
            "Answer: B\n\n"
            "A total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?\n"
            "A. Add 5 to 30 to find 35 teams.\n"
            "B. Divide 30 by 5 to find 6 teams.\n"
            "C. Multiply 30 and 5 to find 150 teams.\n"
            "D. Subtract 5 from 30 to find 25 teams.\n"
            "Answer: B\n\n"
            "A store sells 107 different colors of paint. They have 25 cans of each color in storage. The number of cans of paint the store has in storage can be found using the expression below. 107 × 25. How many cans of paint does the store have in storage?\n"
            "A. 749\n"
            "B. 2,675\n"
            "C. 2,945\n"
            "D. 4,250\n"
            "Answer: B\n\n"
            "Which expression is equivalent to 5 x 9?\n"
            "A. (5 x 4) x (6 x 5)\n"
            "B. (5 x 5) + (5 x 4)\n"
            "C. (5 x 5) + (5 x 9)\n"
            "D. (5 x 9) x (6 x 9)\n"
            "Answer: B\n\n"
        )

    @property
    def name(self) -> str:
        return "mmlu"

    @property
    def evaluator_name(self) -> str:
        return "mmlu"

    def _build_prompt(
        self, question: str, option_a: str, option_b: str, option_c: str, option_d: str
    ) -> str:
        current_question = (
            f"{question}\n"
            f"A. {option_a}\n"
            f"B. {option_b}\n"
            f"C. {option_c}\n"
            f"D. {option_d}\n"
            f"Answer:"
        )

        return f"{self._fewshot_prefix}{current_question}"

    def load(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Expected a list in {self.path}, got {type(raw_data)}")

        for idx, item in enumerate(raw_data):
            required_fields = ["question", "A", "B", "C", "D", "answer"]
            for field in required_fields:
                if field not in item:
                    raise ValueError(
                        f"Missing '{field}' in MMLU sample at index {idx}: {item}"
                    )

            answer = item["answer"].strip().upper()
            if answer not in ["A", "B", "C", "D"]:
                raise ValueError(
                    f"Invalid answer '{answer}' in MMLU sample at index {idx}. Must be A, B, C, or D."
                )

            prompt = self._build_prompt(
                question=item["question"],
                option_a=item["A"],
                option_b=item["B"],
                option_c=item["C"],
                option_d=item["D"],
            )

            data_item = {
                "task_id": idx,
                "prompt": prompt,
                "answer": answer,
                "subject": item.get("subject", "unknown"),
                "question": item["question"],
            }
            data.append(data_item)

        if not data:
            raise ValueError(f"No valid data found in {self.path}!")

        return data

    def get_recommended_config(self) -> Optional[Dict[str, Any]]:
        return {
            "max_new_tokens": 4,
            "temperature": 0.0,
        }
