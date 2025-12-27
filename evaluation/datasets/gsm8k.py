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

_FINAL_HUMAN_SUFFIX = "Question: {question}\nAnswer:\n"


class Gsm8kDataset(BaseDataset):
    def __init__(self):
        self.path = (
            "data/gsm8k.jsonl"
        )

        self._fewshot_prefix = (
            "Solve the question step by step, the answer must be generated after ####\n"
            "Question:\nAaron and Vanessa were relay race partners on a running team. Aaron was able to run each mile twice as fast as Vanessa, but Vanessa was able to run twice as far as Aaron did. If Vanessa ran 4 miles and Aaron completed his part of the race in 16 minutes, how long in minutes did Vanessa take to complete her part?\n"
            "Answer:\nLet's break this down step by step:\n1.  **Determine Aaron's distance:** Vanessa ran 4 miles. Since Vanessa ran twice as far as Aaron, Aaron must have run half that distance.\n    Aaron's distance = 4 / 2 = <<4/2=2>>2 miles.\n2.  **Determine Aaron's pace:** Aaron ran his 2 miles in 16 minutes. To find his pace per mile, we divide the time by the distance.\n    Aaron's pace = 16 / 2 = <<16/2=8>>8 minutes per mile.\n3.  **Determine Vanessa's pace:** The problem states Aaron runs twice as fast as Vanessa. This means it takes Vanessa twice as long to run a mile as it takes Aaron.\n    Vanessa's pace = 8 * 2 = <<8*2=16>>16 minutes per mile.\n4.  **Calculate Vanessa's total time:** Vanessa ran 4 miles at a pace of 16 minutes per mile.\n    Total time = 4 * 16 = <<4*16=64>>64 minutes.\n#### 64\n\n"
            "Question:\n"
            "A factory has two machines, Machine A and Machine B. Machine A produces widgets three times as fast as Machine B. "
            "However, Machine B is scheduled to operate for twice as many hours as Machine A. "
            "If Machine A produced 90 widgets in 3 hours, how many widgets did Machine B produce in its scheduled time?\n"
            "Answer:\n"
            "Let's break this down step by step:\n"
            "1.  **Calculate Machine A's production rate:** Machine A produced 90 widgets in 3 hours. To find the rate per hour, we divide the total widgets by the hours worked.\n"
            "    Machine A's rate = 90 / 3 = <<90/3=30>>30 widgets per hour.\n"
            "2.  **Determine Machine B's production rate:** The problem states Machine A works three times as fast as Machine B. Therefore, Machine B's rate is one-third of Machine A's rate.\n"
            "    Machine B's rate = 30 / 3 = <<30/3=10>>10 widgets per hour.\n"
            "3.  **Determine Machine B's operating time:** Machine B is scheduled to work for twice as many hours as Machine A. Machine A worked for 3 hours.\n"
            "    Machine B's time = 3 * 2 = <<3*2=6>>6 hours.\n"
            "4.  **Calculate Machine B's total production:** Now we have Machine B's rate (10 widgets/hour) and its operating time (6 hours).\n"
            "    Total widgets = 10 * 6 = <<10*6=60>>60 widgets.\n"
            "#### 60\n\n"
            "Question:\nJames hires a horse-drawn carriage from 5 PM to 9 PM.  He gets 1 hour free.  The first paid hour is $15 and each hour after that is twice the cost.  How much did he pay?\n"
            "Answer:\nHe got it for 9-5=<<9-5=4>>4 hours\nHe pays for 4-1=<<4-1=3>>3 hours\nThe first hour cost 1*15=$<<1*15=15>>15\nThe other 3-1=2 hours are more expensive\nThey cost 15*2=$<<15*2=30>>30 per hour\nSo those 2 hours cost 2*30=$<<2*30=60>>60\nSo he pays 60+15=$<<60+15=75>>75 per hours\n#### 75\n\n"
        )

    @property
    def name(self) -> str:
        return "gsm8k"

    @property
    def evaluator_name(self) -> str:
        return "gsm8k"

    def load(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Failed to decode JSON from line in {self.path}: {line}"
                    )
                if "question" not in obj or "answer" not in obj:
                    raise ValueError(
                        f"Line in {self.path} missing 'question' or 'answer': {line}"
                    )

                q = obj["question"]

                prompt = self._fewshot_prefix + _FINAL_HUMAN_SUFFIX.format(question=q)

                data.append(
                    {
                        "task_id": f"gsm8k_test_{idx:04d}",
                        "prompt": prompt,
                        "question": q,
                        "answer": obj["answer"],
                    }
                )

        if not data:
            raise ValueError(f"No valid data found in {self.path}!")
        return data

    def get_recommended_config(self) -> Optional[Dict[str, Any]]:
        return {
            "max_new_tokens": 768,
            "temperature": 0.1,
        }
