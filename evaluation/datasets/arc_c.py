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


class ARCCDataset(BaseDataset):
    def __init__(self):
        self.path = "data/arc-c.json"

    @property
    def name(self) -> str:
        return "arc_c"

    @property
    def evaluator_name(self) -> str:
        return "arc_c"

    def _build_prompt(self, question: str, options: Dict[str, str]) -> str:
        prompt_parts = [f"Question: {question}"]

        for key in sorted(options.keys()):
            prompt_parts.append(f"{key}. {options[key]}")

        prompt_parts.append("Answer:")

        return "\n".join(prompt_parts)

    def load(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Expected a list in {self.path}, got {type(raw_data)}")

        for idx, item in enumerate(raw_data):
            if "question" not in item:
                raise ValueError(
                    f"Missing 'question' in ARC-C sample at index {idx}: {item}"
                )
            if "answer" not in item:
                raise ValueError(
                    f"Missing 'answer' in ARC-C sample at index {idx}: {item}"
                )

            options = {}
            for option_key in ["A", "B", "C", "D"]:
                if option_key in item and item[option_key]:
                    options[option_key] = item[option_key]

            if len(options) < 2:
                raise ValueError(
                    f"ARC-C sample at index {idx} has less than 2 options: {item}"
                )

            answer = item["answer"].strip().upper()
            if answer not in options:
                raise ValueError(
                    f"Invalid answer '{answer}' in ARC-C sample at index {idx}. "
                    f"Answer must be one of the available options: {list(options.keys())}"
                )

            prompt = self._build_prompt(question=item["question"], options=options)

            data_item = {
                "task_id": idx,
                "prompt": prompt,
                "answer": answer,
                "question": item["question"],
                "num_options": len(options),
                "available_options": sorted(options.keys()),
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
