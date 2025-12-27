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


class HellaSwagDataset(BaseDataset):
    def __init__(self):
        self.path = "data/hellaswag.json"

        self._fewshot_prefix = (
            "Continue the following text without adding any additional information or formatting:\n\n"
            "A woman is kneeling down talking the the camera. The woman then grabs a pair of running stilts and demonstrates them. Several people\n"
            "A) are shown running in various parts of the city during the performance.\n"
            "B) are then shown jumping into the pool from many different positions using the stilts.\n"
            "C) then go through a city park on the running stilts.\n"
            "D) are shown working out filling in the stance.\n"
            "What is the right option?\n"
            "C\n\n"
            "A person is riding on a dirt bike. He sets up for a race. He\n"
            "A) starts racing on his dirt bike.\n"
            "B) races along paved roads at full speed.\n"
            "C) races over a small hill and jumps on the green track.\n"
            "D) slows down as he approaches a finish line.\n"
            "What is the right option?\n"
            "A\n\n"
            "A woman is laying down in a chair as a man with gloves works. He places clamps on various parts of her nose. He\n"
            "A) screws out a nose check and clamps his other to make her nose bleed.\n"
            "B) puts the clamps on the surface of her nose and pulls out her nose.\n"
            "C) then inserts a rod before creating a piercing.\n"
            "D) then pumps a hose through her nose and ends by pulling a piece of fabric away from her face.\n"
            "What is the right option?\n"
            "C\n\n"
            "A man is standing inside a bathroom. He\n"
            "A) is washing his hands.\n"
            "B) uses scissors to trim his beard.\n"
            "C) starts to shave his beard off with a shaver machine.\n"
            "D) is using a mops and bucket to clean a table in front of him.\n"
            "What is the right option?\n"
            "B\n\n"
            "A man and woman are standing in a starkly white room. They lay plastic on the ground and fill a tub with white paint. They\n"
            "A) then demonstrate how to hang a strip of wallpaper onto the wall.\n"
            "B) paint the bathroom walls and sink with the paint can.\n"
            "C) use liquid adhesive to create the foil.\n"
            "D) then paint the walls black.\n"
            "What is the right option?\n"
            "A\n\n"
            "He cuts the sandwich into four slices on a board. He puts a toothpick in each section of the sandwich. He\n"
            "A) lays out cheese slices onto the sandwich.\n"
            "B) then scrapes off what's missing from the sandwich.\n"
            "C) moves the sandwich from the board onto a plate.\n"
            "D) puts ssel sprinths on the sandwich.\n"
            "What is the right option?\n"
            "C\n\n"
            "We then see men spinning on the pommel horse with title screens clipped in between. We see a man throwing his legs over the pommel horse. We\n"
            "A) see a man this way thrown off the horse.\n"
            "B) see a man in red spinning himself around the pommel horse quickly.\n"
            "C) see a man hit the ground and stand up.\n"
            "D) see the screen again.\n"
            "What is the right option?\n"
            "B\n\n"
            "A man is in front of a crowd with a midget, as they throw darts at a dart board. They\n"
            "A) take turns throwing the darts.\n"
            "B) have one in a hand and stuck on each loser picking it up.\n"
            "C) put an umbrella over the dart board.\n"
            "D) are bored and are sporting another gun in their hand.\n"
            "What is the right option?\n"
            "A\n\n"
            "Several young men play a game on a foosball table. A new player\n"
            "A) comes in, and begins shuffling the wedlms, while the dealer gives tips.\n"
            "B) passes the ball to the other one.\n"
            "C) walks up to the table, counts the balls and indicates for his score.\n"
            "D) arrives, who is a much older man amd plays a game with the younger men.\n"
            "What is the right option?\n"
            "D\n\n"
            "A male athlete prepares himself to run. He runs with a javelin over his shoulder. He\n"
            "A) then hurls it over all of his body, and lands on the shoulder of a person standing in the field behind him.\n"
            "B) throws it as hard as he can.\n"
            "C) shoots and catches the javelin.\n"
            "D) then takes long, really fast strides.\n"
            "What is the right option?\n"
            "B\n\n"
        )

    @property
    def name(self) -> str:
        return "hellaswag"

    @property
    def evaluator_name(self) -> str:
        return "hellaswag"

    def _build_prompt(
        self, question: str, option_a: str, option_b: str, option_c: str, option_d: str
    ) -> str:
        current_question = (
            f"{question}\n"
            f"A) {option_a}\n"
            f"B) {option_b}\n"
            f"C) {option_c}\n"
            f"D) {option_d}\n"
            f"What is the right option?\n"
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
                        f"Missing '{field}' in HellaSwag sample at index {idx}: {item}"
                    )

            answer = item["answer"].strip().upper()
            if answer not in ["A", "B", "C", "D"]:
                raise ValueError(
                    f"Invalid answer '{answer}' in HellaSwag sample at index {idx}. Must be A, B, C, or D."
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
